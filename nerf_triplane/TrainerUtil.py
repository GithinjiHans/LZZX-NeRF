import asyncio
import gc
import glob
import queue
import subprocess
import threading
import time

import cv2
import imageio
import tensorboardX
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import trimesh
from pydub import AudioSegment
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from .utils import *


class TrainerUtil(object):
    '''该类由于太过庞大，特将其从原utils.py中独立出来！'''

    def __init__(self,
                 name,  # 实验的名称
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # 损失函数，如果为None，则假设在train_step中内联实现
                 optimizer=None,  # 优化器
                 ema_decay=None,  # 如果使用EMA（指数移动平均），则设置衰减值
                 ema_update_interval=1000,  # 每个训练步骤更新EMA的频率
                 lr_scheduler=None,  # 学习率调度器
                 metrics=[],  # 评估指标，如果为None，则使用val_loss衡量性能，否则使用第一个指标
                 local_rank=0,  # 当前使用的GPU编号
                 world_size=1,  # 总GPU数量
                 device=None,  # 要使用的设备，通常设置为None即可（自动选择设备）
                 mute=False,  # 是否不打印所有输出
                 fp16=False,  # AMP优化级别
                 eval_interval=1,  # 每隔多少个epoch进行一次评估
                 max_keep_ckpt=2,  # 磁盘上保存的最大检查点数量
                 workspace='workspace',  # 保存日志和检查点的工作空间 workspace to save logs & ckpts
                 best_mode='min',  # 结果 越小/越大 越好 the smaller/larger result, the better
                 use_loss_as_metric=True,  # 将损失作为第一个指标
                 report_metric_at_train=False,  # 是否在训练时报告指标
                 use_checkpoint="latest",  # 初始化时使用的检查点
                 use_tensorboardX=True,  # 是否使用tensorboard进行日志记录
                 scheduler_update_every_step=False,  # 是否在每个训练步骤后调用scheduler.step()
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.flip_finetune_lips = self.opt.finetune_lips
        self.flip_init_lips = self.opt.init_lips
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1 or self.opt.finetune_lips or True:
            import lpips
            # self.criterion_lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
            self.criterion_lpips_alex = lpips.LPIPS(net='alex').to(self.device)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            self.init_workspace(self.workspace)

        '''定义一个全局的队列，用于存放最终生成结果的图像和优化图像的共享数据'''
        self.resultImgBitsQueue = queue.Queue()
        # 是否进行人脸优化
        self.faceOptimize = self.opt.face_optimize

    def __del__(self):
        if self.log_ptr and not self.log_ptr.closed:
            self.log_ptr.close()

    def init_workspace(self, workspace: str):
        os.makedirs(workspace, exist_ok=True)
        if self.workspace is None or self.workspace=='':
            self.workspace = workspace

        self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
        self.log_ptr = open(self.log_path, "a+")

        self.ckpt_path = os.path.join(workspace, 'checkpoints')
        self.best_path = f"{self.ckpt_path}/{self.name}.pth"
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {workspace}')
        # self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 6]
        face_mask = data['face_mask']  # [B, N]
        eye_mask = data['eye_mask']  # [B, N]
        lhalf_mask = data['lhalf_mask']
        eye = data['eye']  # [B, 1]
        auds = data['auds']  # [B, 29, 16]
        index = data['index']  # [B]

        if not self.opt.torso:
            rgb = data['images']  # [B, N, 3]
        else:
            rgb = data['bg_torso_color']

        B, N, C = rgb.shape

        if self.opt.color_space == 'linear':
            rgb[..., :3] = utils.srgb_to_linear(rgb[..., :3])

        bg_color = data['bg_color']

        if not self.opt.torso:
            renderResult = self.model.render(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index, staged=False,
                                             bg_color=bg_color, perturb=True, force_all_rays=False if (
                        self.opt.patch_size <= 1 and not self.opt.train_camera) else True, **vars(self.opt))
        else:
            renderResult = self.model.render_torso(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index,
                                                   staged=False, bg_color=bg_color, perturb=True,
                                                   force_all_rays=False if (
                                                           self.opt.patch_size <= 1 and not self.opt.train_camera) else True,
                                                   **vars(self.opt))

        if isinstance(renderResult, tuple):
            outputs = renderResult[0]
            times = renderResult[1]
        else:
            outputs = renderResult
        # 固定头部和训练躯干
        if not self.opt.torso:
            pred_rgb = outputs['image']
        else:
            pred_rgb = outputs['torso_color']

        # loss factor
        step_factor = min(self.global_step / self.opt.iters, 1.0)

        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # 固定头部和训练躯干，设置了参数torso=True则在此返回
        if self.opt.torso:
            loss = loss.mean()
            loss += ((1 - self.model.anchor_points[:, 3]) ** 2).mean()
            return pred_rgb, rgb, loss

        # camera optim regularization
        # if self.opt.train_camera:
        #     cam_reg = self.model.camera_dR[index].abs().mean() + self.model.camera_dT[index].abs().mean()
        #     loss = loss + 1e-2 * cam_reg

        # 使用LPIPS微调嘴唇，是运行参数中的finetune_lips，默认为True
        if not self.flip_finetune_lips:
            # 使用不确定性损失
            if self.opt.unc_loss:
                alpha = 0.2
                uncertainty = outputs['uncertainty']  # [N], abs sum
                beta = uncertainty + 1

                unc_weight = F.softmax(uncertainty, dim=-1) * N
                # print(unc_weight.shape, unc_weight.max(), unc_weight.min())
                loss *= alpha + (1 - alpha) * ((1 - step_factor) + step_factor * unc_weight.detach()).clamp(0, 10)
                # loss *= unc_weight.detach()

                beta = uncertainty + 1
                norm_rgb = torch.norm((pred_rgb - rgb), dim=-1).detach()
                loss_u = norm_rgb / (2 * beta ** 2) + (torch.log(beta) ** 2) / 2
                loss_u *= face_mask.view(-1)
                loss += step_factor * loss_u

                loss_static_uncertainty = (uncertainty * (~face_mask.view(-1)))
                loss += 1e-3 * step_factor * loss_static_uncertainty

            # patch-based rendering
            if self.opt.patch_size > 1:
                rgb = rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
                pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1,
                                                                                                  2).contiguous() * 2 - 1

                # torch_vis_2d(rgb[0])
                # torch_vis_2d(pred_rgb[0])

                # LPIPS loss ?
                loss_lpips = self.criterion_lpips_alex(pred_rgb, rgb)
                loss = loss + 0.1 * loss_lpips

        # lips finetune
        # 使用LPIPS微调嘴唇，嘴唇训练，默认为True
        # 虽然初始化函数中self.flip_finetune_lips = self.opt.finetune_lips，但此处实践发现self.flip_finetune_lips会动态变化！
        # 判断是否启用LPIPS微调嘴唇，一定要使用opt.finetune_lips来判断！
        # print(f'当前self.flip_finetune_lips：{self.flip_finetune_lips}')
        if self.opt.finetune_lips:
            # TODO
            xmin, xmax, ymin, ymax = data['rect']

            rgb = rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
            pred_rgb = pred_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1

            padding_h = max(0, (32 - rgb.shape[-2] + 1) // 2)
            padding_w = max(0, (32 - rgb.shape[-1] + 1) // 2)

            if padding_w or padding_h:
                rgb = torch.nn.functional.pad(rgb, (padding_w, padding_w, padding_h, padding_h))
                pred_rgb = torch.nn.functional.pad(pred_rgb, (padding_w, padding_w, padding_h, padding_h))

            # torch_vis_2d(rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss
            loss = loss + 0.01 * self.criterion_lpips_alex(pred_rgb, rgb)

        # flip every step... if finetune lips
        if self.flip_finetune_lips:
            self.opt.finetune_lips = not self.opt.finetune_lips

        loss = loss.mean()

        # weights_sum loss
        # entropy to encourage weights_sum to be 0 or 1.
        if self.opt.torso:
            alphas = outputs['torso_alpha'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        else:
            alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        # aud att loss (regions out of face should be static)
        if self.opt.amb_aud_loss and not self.opt.torso:
            ambient_aud = outputs['ambient_aud']
            loss_amb_aud = (ambient_aud * (~face_mask.view(-1))).mean()
            # gradually increase it
            lambda_amb = step_factor * self.opt.lambda_amb
            loss += lambda_amb * loss_amb_aud

        # eye att loss
        if self.opt.amb_eye_loss and not self.opt.torso:
            ambient_eye = outputs['ambient_eye'] / self.opt.max_steps

            loss_cross = ((ambient_eye * ambient_aud.detach()) * face_mask.view(-1)).mean()
            loss += lambda_amb * loss_cross

        # regularize
        if self.global_step % 16 == 0 and not self.flip_finetune_lips:
            xyzs, dirs, enc_a, ind_code, eye = outputs['rays']
            xyz_delta = (torch.rand(size=xyzs.shape, dtype=xyzs.dtype, device=xyzs.device) * 2 - 1) * 1e-3
            with torch.no_grad():
                sigmas_raw, rgbs_raw, ambient_aud_raw, ambient_eye_raw, unc_raw = self.model(xyzs, dirs, enc_a.detach(),
                                                                                             ind_code.detach(), eye)
            sigmas_reg, rgbs_reg, ambient_aud_reg, ambient_eye_reg, unc_reg = self.model(xyzs + xyz_delta, dirs,
                                                                                         enc_a.detach(),
                                                                                         ind_code.detach(), eye)

            lambda_reg = step_factor * 1e-5
            reg_loss = 0
            if self.opt.unc_loss:
                reg_loss += self.criterion(unc_raw, unc_reg).mean()
            if self.opt.amb_aud_loss:
                reg_loss += self.criterion(ambient_aud_raw, ambient_aud_reg).mean()
            if self.opt.amb_eye_loss:
                reg_loss += self.criterion(ambient_eye_raw, ambient_eye_reg).mean()

            loss += reg_loss * lambda_reg

        return pred_rgb, rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]

        images = data['images']  # [B, H, W, 3/4]
        auds = data['auds']
        index = data['index']  # [B]
        eye = data['eye']  # [B, 1]

        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = utils.srgb_to_linear(images[..., :3])

        # eval with fixed background color
        # bg_color = 1
        bg_color = data['bg_color']

        # 当评估模型效果时，需要将模型参数evaluate改为True
        self.model.evaluate = True
        renderResult = self.model.render(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index, staged=True,
                                         bg_color=bg_color, perturb=False, **vars(self.opt))
        self.model.evaluate = False
        outputs = renderResult[0]
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_ambient_aud = outputs['ambient_aud'].reshape(B, H, W)
        pred_ambient_eye = outputs['ambient_eye'].reshape(B, H, W)
        pred_uncertainty = outputs['uncertainty'].reshape(B, H, W)

        loss_raw = self.criterion(pred_rgb, images)
        loss = loss_raw.mean()

        return pred_rgb, pred_depth, pred_ambient_aud, pred_ambient_eye, pred_uncertainty, images, loss, loss_raw

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]

        auds = data['auds']  # [B, 29, 16]
        index = data['index']
        H, W = data['H'], data['W']

        # allow using a fixed eye area (avoid eye blink) at test
        # exp_eye：明确的控制眼睛，默认为True
        # fix_eye：固定的眼睛区域，负为禁用，设置为0-0.3为合理的眼睛，默认为-1
        # 推理时，opt.exp_eye=True,opt.fix_eye=-1
        if self.opt.exp_eye and self.opt.fix_eye >= 0:
            eye = torch.FloatTensor([self.opt.fix_eye]).view(1, 1).to(self.device)
        else:
            eye = data['eye']  # [B, 1]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        else:
            bg_color = data['bg_color']

        self.model.testing = True
        # 此处最终调用到renderer.py中的render方法！
        outputs, ts = self.model.render(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index, staged=True,
                                        bg_color=bg_color, perturb=perturb, **vars(self.opt))
        self.model.testing = False

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        # pred_depth = outputs['depth'].reshape(-1, H, W)

        # return pred_rgb, pred_depth
        return pred_rgb, None, ts

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = utils.extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:],
                                                     resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # 循环每步训练步骤
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # 执行训练的关键方法
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False,remove_old=False) # 不删除模型文件

            if self.epoch % self.eval_interval == 0:
                # 隔指定的步数的时候，就做一次模型评估
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, write_image=False, consume_save_ts=True, push_mq=False, mqInstance=None):
        '''测试，并生成结果。
        结果以每帧图片的形式生成。
        这里整合了GFPGAN项目，将每帧图片进行优化。
        采用多线程形式并行执行生成结果和优化的图像的程序。
        最终将优化好的图像合并为视频片段，再将多个片段合并为整体视频文件。
        Ajian 2023.10.20：此日期前后将推理图像直接写成了ts文件，支持hls协议进行流媒体播放，建议使用main-webui.py进行可视化推理！
        '''

        save_path = os.path.join(self.workspace, 'results')
        os.makedirs(save_path, exist_ok=True)
        consumeRuning = False

        self.log(f"==> 推理开始, save results to {save_path}")

        try:
            print(
                f'推理开始，len(loader)={len(loader)}，loader.batch_size={loader.batch_size}，总帧数：{len(loader) * loader.batch_size}')
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            self.model.eval()
            flushStep = 125
            all_preds = [i for i in range(flushStep)]
            predsIndex = 0
            # all_preds_depth = []

            # with torch.no_grad()的使用就像一个循环，其中循环内的每个张量都将requires_grad设置为False（如果需要为张量计算梯度，则为True，否则为False）。
            # 这意味着当前与当前计算图相连的任何具有梯度的张量现在都与当前图分离。我们不再能够计算关于这个张量的梯度。
            with torch.no_grad():
                for i, data in enumerate(loader):
                    begin = time.time()
                    # 推理时self.fp16=True
                    # 自动混合精度（Automatic Mixed Precision, AMP)训练，是在训练一个数值精度 FP32 的模型，一部分算子的操作时，数值精度为 FP16，其余算子的操作精度是 FP32，
                    # 而具体哪些算子用 FP16，哪些用 FP32，不需要用户关心，amp 自动给它们都安排好了。这样在不改变模型、不降低模型训练精度的前提下，可以缩短训练时间，降低存储需求
                    # `autocast(enable=True)` 可以作为上下文管理器和装饰器来使用，给算子自动安排按照 FP16 或者 FP32 的数值精度来操作
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        # 推理的工作就在test_step方法中
                        preds, preds_depth, ts = self.test_step(data)
                    if (i + 1) % 100 == 0:
                        print(f'本100次推理各环节累积用时(s):{ts}')
                    # path = os.path.join(save_path, f'{name}_{i:04d}_rgb.png')
                    # path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')

                    # self.log(f"[INFO] saving test image to {path}")

                    # color_space配置，颜色空间，默认srgb
                    # 推理时，opt.color_space='srgb'
                    if self.opt.color_space == 'linear':
                        preds = utils.linear_to_srgb(preds)
                    if consume_save_ts:
                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)

                    # 如果设置了mq，则将数据存入mq中
                    if push_mq and mqInstance is not None:
                        mqInstance.push((preds[0].detach().cpu().numpy() * 255).astype(np.uint8).tobytes())

                    # pred_depth = preds_depth[0].detach().cpu().numpy()
                    # pred_depth = (pred_depth * 255).astype(np.uint8)
                    # if write_image:
                    # imageio.imwrite(path, pred)
                    # imageio.imwrite(path_depth, pred_depth)
                    if consume_save_ts:
                        all_preds[predsIndex] = pred
                    predsIndex += 1

                    # all_preds_depth.append(pred_depth)

                    '''
                    为解决此处内存占用巨大，后续进行np.stack()操作时可能内存溢出的问题，
                    将数组以指定个数为一组单位，送入GFPGAN中进行优化，得到优化后的图像，
                    将优化后的图像直接写入磁盘mp4文件。
                    后续将这些mp4小文件进行拼合为一个整体的mp4大文件。
                    为加快运行速度，这里采用多线程并行方式执行。
                    20231021:改为固定5s生成一段ts视频，使其支持hls协议，可供浏览器直接播放
                    '''
                    if consume_save_ts and (i + 1) % flushStep == 0:
                        # 将本批次生成的图存入共享队列中
                        self.resultImgBitsQueue.put(all_preds)
                        # 启动消费者线程
                        if not consumeRuning:
                            # 寻找本次推理所使用的音频文件
                            if self.opt.asr_model == 'deepspeech':
                                audio_path = self.opt.aud.replace('.npy', '')
                            elif self.opt.asr_model == 'hubert':
                                audio_path = self.opt.aud.replace('_hu.npy', '')
                            else:
                                audio_path = self.opt.aud.replace('_eo.npy', '')
                            # 除了wav，有可能还使用其他格式的音频
                            if os.path.exists(audio_path + '.wav'):
                                audio_path += '.wav'
                            elif os.path.exists(audio_path + '.mp3'):
                                audio_path += '.mp3'
                            elif os.path.exists(audio_path + '.flac'):
                                audio_path += '.flac'
                            else:
                                audio_path += '.wma'
                            # 设置event，开启消费者线程
                            consumeEvent = threading.Event()
                            consume = threading.Thread(target=self.consume_hls_stream,
                                                       args=(consumeEvent, audio_path, save_path,))
                            consume.start()
                            consumeRuning = True
                        predsIndex = 0
                        gc.collect()

                    pbar.update(loader.batch_size)

            # write video
            # all_preds = np.stack(all_preds, axis=0)
            # all_preds_depth = np.stack(all_preds_depth, axis=0)
            # self.log(f"写入结果文件：{os.path.join(save_path, f'{name}.mp4')}...")
            # imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            # imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            if consume_save_ts:
                if all_preds:
                    # 清除最后一次没有占完位置的元素
                    del all_preds[predsIndex:len(all_preds)]
                    self.resultImgBitsQueue.put(all_preds)
                    self.log('主线程生产者已经将所有数据放入队列！')
                del all_preds
                # 阻塞等待队列被消费完
                self.log('主线程等待消费者消费完成！')
                self.resultImgBitsQueue.join()
                # 设置事件已经完成，通知消费者停止循环
                consumeEvent.set()
                # # 整体合并视频
                # self.log(f"写入结果文件：{os.path.join(save_path, f'{name}.mp4')}...")
                # full = self.concatMp4FragmentAndClean(save_path, name + '.mp4')
                # self.log('整体mp4：' + full)
                self.log('生成HLS完成，结束！')
                gc.collect()
        except Exception as e:
            self.log(f"异常了！{e}")
            e.with_traceback()

        self.log(f"==> Finished Test.")

    async def test_inference_2(self, dataDic, currentTaskIndex, audio_full_path: str, push_mq=False, mqInstance=None):
        '''进行推理<br>
            :param dataDic:当前批次加载的原始数据
            :param currentTaskIndex:当前批次任务的下标
            :param audio_full_path:本次推理驱动口型的音频帧全路径名,xxxx/xxx/xx.wav
        '''
        if len(dataDic['index']) <= currentTaskIndex:
            return
        try:
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    v = {}
                    try:
                        v['rays_o'] = dataDic['rays_o'][currentTaskIndex].unsqueeze(0)  # [B, N, 3]
                    except IndexError as e1:
                        # 下标错误，说明已经到最后几个数据，长度可能不足
                        return
                    v['rays_d'] = dataDic['rays_d'][currentTaskIndex].unsqueeze(0)  # [B, N, 3]
                    v['bg_coords'] = dataDic['bg_coords']  # [1, N, 2]
                    v['poses'] = dataDic['poses'][currentTaskIndex].unsqueeze(0)  # [B, 7]
                    v['auds'] = dataDic['auds'][currentTaskIndex]  # [B, 29, 16]
                    v['index'] = [dataDic['index'][currentTaskIndex]]
                    v['H'] = dataDic['H']
                    v['W'] = dataDic['W']
                    v['eye'] = dataDic['eye'][currentTaskIndex].unsqueeze(0)  # [B, 1]
                    v['bg_color'] = dataDic['bg_color'][currentTaskIndex].unsqueeze(0)
                    preds, _, ts = self.test_step(v)
                    if push_mq and mqInstance is not None:
                        # 将当前帧及对应的音频数据放入队列中
                        ns = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                        mqInstance.pushGenerateFramesBytes(ns.tobytes())
                        #print(f"推入队列一帧，编号：{v['index']}")
                       # import imageio
                       # imageio.imwrite('/root/test.mp4', ns)

                    return 'done'
        except Exception as e:
            self.log(f"异常了----->{e}")
            e.with_traceback()
        self.log(f"==> Finished Test.")

    async def test_with_2(self, loader, audio_full_path: str, consume_save_ts=True, push_mq=False,
                          mqInstance=None, local_save_file_name: list = None):
        pbar = tqdm.tqdm(total=loader._data.dataLoaderSetSize,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():
            # 执行推理
            if loader.batch_size > 1:
                print(f'\n\n\n★★★★★★★★★★-------推理开始执行，loader.batch_size={loader.batch_size}...-------★★★★★★★★★★')
                start1 = time.time()
                file_name = None
                mqInstance.setInferenceTotalFramesNum(loader._data.dataLoaderSetSize) # 设置总帧数
                for dic in loader:
                    # 构建任务
                    tasks = [None] * loader.batch_size
                    for j in range(loader.batch_size):
                        tasks[j] = asyncio.create_task(
                            self.test_inference_2(dic, j, audio_full_path, push_mq=True, mqInstance=mqInstance))
                    #print(f'并行执行task len：{len(tasks)}')
                    # 并行执行
                    await asyncio.gather(*tasks)
                    # 更新进度条
                    pbar.update(loader.batch_size)
                print(f'★★★★★★★★★★-------推理结束！用时:{time.time() - start1}s-------★★★★★★★★★★\n\n\n')
                # 推理完成
                if local_save_file_name is not None:
                    local_save_file_name.append(file_name)
                    print(f'local_save_file_name={local_save_file_name}')
            else:
                self.test(loader, consume_save_ts=consume_save_ts, push_mq=push_mq, mqInstance=mqInstance)
            # 推理完成，等待写本地文件进程执行完成，并恢复静默视频推流子进程
            mqInstance.pushAndSaveFrames_done()
            mqInstance.pushWaitVideoForModel(mqInstance.modelFullPath, mqInstance.rtmpConfigDict["remoteRtmpURL"])

    def writeMp4Fragment(self, all_preds, save_path, fragmentName):
        '''
        分小片段的写成mp4文件,后续将这些mp4小文件进行拼合为一个整体的mp4大文件
        '''
        all_preds = np.stack(all_preds, axis=0)
        imageio.mimwrite(os.path.join(save_path, fragmentName), all_preds, fps=25, quality=8,
                         macro_block_size=1)
        # 写入filelist.txt文件，后续使用ffmpeg拼接
        with open(os.path.join(save_path, 'filelist.txt'), 'a') as file:
            file.write(f"file '{fragmentName}'\n")
        all_preds = []

    def concatMp4FragmentAndClean(self, saveBasePath, mp4Name):
        '''
        拼接多个碎片mp4文件，返回拼接完整之后的整体mp4文件完整路径。并清理碎片文件和filelist.txt
        :param saveBasePath: mp4碎片文件存放目录
        :param mp4Name: 拼接好之后的mp4文件的名字，带后缀，如：xxx.mp4
        :return:
        '''
        full = os.path.join(saveBasePath, mp4Name)
        cmd = f"ffmpeg -f concat -safe 0 -i {os.path.join(saveBasePath, 'filelist.txt')} -c copy {full}"
        print(f'拼接mp4碎片文件，命令：【{cmd}】')
        p = subprocess.Popen(cmd, shell=True, cwd=saveBasePath)
        p.wait()
        # print(f'清理该{saveBasePath}下的碎片文件...')
        # for f in os.listdir(saveBasePath):
        #     if f.find('===') > -1 or f == 'filelist.txt':
        #         os.remove(os.path.join(saveBasePath, f))
        return full

    def optimizeResultAndWriteVideoFragmentConsume(self, save_path: str):
        '''消费者：将生成结果进行优化，并将生成结果图写入视频片段'''
        from face_optimize.gfpgan_util import GFPGANUtil
        gan = GFPGANUtil()
        while True:
            all_preds = self.resultImgBitsQueue.get()
            # 优化图像
            if self.faceOptimize:
                print(f'开始优化图像{len(all_preds)}...')
                all_preds = gan.doRestor(all_preds)
            # 图像存为短视频
            self.writeMp4Fragment(all_preds, save_path, f'{str(int(time.time()))}.mp4')
            print('视频片段存储完成！')
            gc.collect()
            self.resultImgBitsQueue.task_done()

    def consume_hls_stream(self, event, audio_full_path: str, inference_result_path: str):
        '''消费者：将生成图像封包为ts文件，以支持hls流播放
            :param event:事件，主线程通过设置来通知子线程工作完成
            :param audio_full_path:本次推理所依据的音频文件的全路径
            :param inference_result_path:本次推理结果存储目录，默认应该是trial_{model}_torso/results/
        '''
        self.log(f"==> 生成TS文件...")
        tsIndex = 0
        while True:
            self.log(f"循环等待...")
            if event.is_set():
                try:
                    self.resultImgBitsQueue.task_done()
                except:
                    pass
                self.log(f"==> 生产者完成！")
                # 生产者已经执行完成
                break
            # 取出队列中的帧图片，按视频25帧率的规格，125帧应该是5s视频文件
            frames = self.resultImgBitsQueue.get()
            self.log(f"==> 得到帧数据{len(frames)}")
            if not frames:
                try:
                    self.resultImgBitsQueue.task_done()
                except:
                    pass
                continue
            # 创建m3u8文件
            if tsIndex == 0:
                try:
                    self.log(f"==> 生成m3u8...")
                    m3u8PathName = os.path.join(inference_result_path, str(time.time()).replace('.', ''), 'video.m3u8')
                    hls_util.create_m3u8_by_totalTime(len(AudioSegment.from_wav(audio_full_path)), m3u8PathName)
                    self.log(f"==> m3u8生成完成.")
                except Exception as e:
                    print(f'm3u8生成报错：{e}')
            # 封装ts文件，默认封成512x512的视频，以加快处理速度
            self.log("==> 封装ts...")
            try:
                hls_util.create_ts_with_5sec(save_path=os.path.dirname(m3u8PathName), frame_ndarray=frames,
                                             audio_path_name=audio_full_path, ts_index=tsIndex)
            except Exception as e:
                print(f'调用ts生成方法失败：{e}')
            self.log(f"==> 封装ts完成!")
            try:
                self.resultImgBitsQueue.task_done()
            except:
                pass
            # 输出特殊字符日志，以便外部程序监听到m3u8文件产生了！
            if tsIndex == 1:  # 服务器至少已经生成了2个ts文件，才通知客户端播放
                # 高频输出视频地址，防止客户端读取不到
                self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
                print(f"##M3U8##SUCCESS:{m3u8PathName}")
                time.sleep(0.3)
                self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
                print(f"##M3U8##SUCCESS:{m3u8PathName}")
                time.sleep(0.3)
                self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
                print(f"##M3U8##SUCCESS:{m3u8PathName}")
            tsIndex += 1
        if tsIndex == 0:
            try:
                self.resultImgBitsQueue.task_done()
            except:
                pass
            # 高频输出视频地址，防止客户端读取不到
            self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
            print(f"##M3U8##SUCCESS:{m3u8PathName}")
            time.sleep(0.3)
            self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
            print(f"##M3U8##SUCCESS:{m3u8PathName}")
            time.sleep(0.3)
            self.log(f"##M3U8##SUCCESS:{m3u8PathName}")
            print(f"##M3U8##SUCCESS:{m3u8PathName}")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, auds, eye=None, index=0, bg_color=None, spp=1, downscale=1):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        if auds is not None:
            auds = auds.to(self.device)

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = utils.get_rays(pose, intrinsics, rH, rW, -1)

        bg_coords = utils.get_bg_coords(rH, rW, self.device)

        if eye is not None:
            eye = torch.FloatTensor([eye]).view(1, 1).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'auds': auds,
            'index': [index],  # support choosing index for individual codes
            'eye': eye,
            'poses': pose,
            'bg_coords': bg_coords,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3,
                                                                                                   1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = utils.linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    # [GUI] test with provided data
    def test_gui_with_data(self, data, W, H):

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                preds, preds_depth = self.test_step(data, perturb=False)

        if self.ema is not None:
            self.ema.restore()

        if self.opt.color_space == 'linear':
            preds = utils.linear_to_srgb(preds)

        # the H/W in data may be differnt to GUI, so we still need to resize...
        preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1).contiguous()
        preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> 开始训练.. {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, mininterval=1,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        # 训练的核心方法
        t1 = 0
        t2 = 0
        t3 = 0
        tindex = 0
        for data in loader:
            tindex += 1
            # update grid every 16 steps
            begin = time.time()
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # 最终调用到renderer.py中的update_extra_state方法
                    # 经过监测，此处每次约用时0.01毫秒
                    self.model.update_extra_state()
                    t1 += time.time() - begin

            begin = time.time()
            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()  # 梯度清零
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # 最终在这里执行训练
                # 经过监测，此处每次约用时0.01毫秒
                preds, truths, loss = self.train_step(data)
                t2 += time.time() - begin

            begin = time.time()
            # 经过监测，此部分以下代码每次执行约需0.1毫秒，执行比较耗时
            # 经过优化，此部分代码已不是最耗时部分
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

            # 当前GPU编号，默认为0
            if self.local_rank == 0:
                # 是否报告训练的指标值，默认为False
                if self.report_metric_at_train:
                    # self.metrics默认为[]
                    # 当参数--test=True时，即在推理的时候，self.metrics=[PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]
                    for metric in self.metrics:
                        # update方法即为[PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]类的方法
                        metric.update(preds, truths)

                # 为加快速度，禁用了日志写入
                # if self.use_tensorboardX:
                # self.writer.add_scalar("train/loss", loss_val, self.global_step)
                # self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                # 是否在每个步骤之后调用scheduler.step()方法，默认为False
                # 当嘴型训练时这里被设置了为True
                # 为加快速度，这里也直接禁用！
                # if self.scheduler_update_every_step:
                #     pbar.set_description(
                #         f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                # else:
                #     pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)
            t3 += time.time() - begin

            if tindex % 100 == 0:
                print(f'本100次用时ms：t1={t1},t2={t2},t3={t3}')

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            print(f'当前是评估模型效果，load batch_size=={loader.batch_size}...')
            for data in loader:
                self.local_step += 1
                #print(f'当前data：{data}')
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, pred_ambient_aud, pred_ambient_eye, pred_uncertainty, truths, loss, loss_raw = self.eval_step(
                        data)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')
                    # save_path_error = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_errormap.png')
                    save_path_ambient_aud = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_aud.png')
                    save_path_ambient_eye = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_eye.png')
                    save_path_uncertainty = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_uncertainty.png')
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = utils.linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    # loss_raw = loss_raw[0].mean(-1).detach().cpu().numpy()
                    # loss_raw = (loss_raw - np.min(loss_raw)) / (np.max(loss_raw) - np.min(loss_raw))
                    pred_ambient_aud = pred_ambient_aud[0].detach().cpu().numpy()
                    pred_ambient_aud /= np.max(pred_ambient_aud)
                    pred_ambient_eye = pred_ambient_eye[0].detach().cpu().numpy()
                    pred_ambient_eye /= np.max(pred_ambient_eye)
                    # pred_ambient = pred_ambient / 16
                    # print(pred_ambient.shape)
                    pred_uncertainty = pred_uncertainty[0].detach().cpu().numpy()
                    # pred_uncertainty = (pred_uncertainty - np.min(pred_uncertainty)) / (np.max(pred_uncertainty) - np.min(pred_uncertainty))
                    pred_uncertainty /= np.max(pred_uncertainty)

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if not self.opt.torso:
                        cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                        # cv2.imwrite(save_path_error, (loss_raw * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_aud, (pred_ambient_aud * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_eye, (pred_ambient_eye * 255).astype(np.uint8))
                        cv2.imwrite(save_path_uncertainty, (pred_uncertainty * 255).astype(np.uint8))
                        # cv2.imwrite(save_path_gt, cv2.cvtColor((linear_to_srgb(truths[0].detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                if self.best_mode == 'min':
                    print('1结果越小越好')
                else:
                    print('1结果越大越好')
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss
                print('1使用最小loss作为本次结果...')

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        state['mean_count'] = self.model.mean_count
        state['mean_density'] = self.model.mean_density
        state['mean_density_torso'] = self.model.mean_density_torso

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                # always save new as best... (since metric cannot really reflect performance...)
                if True:

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            print(f'从该目录下加载模型：{self.ckpt_path}/{self.name}_ep*.pth')
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded bare model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if 'mean_count' in checkpoint_dict:
            self.model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            self.model.mean_density = checkpoint_dict['mean_density']
        if 'mean_density_torso' in checkpoint_dict:
            self.model.mean_density_torso = checkpoint_dict['mean_density_torso']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
