import argparse
import time

from nerf_triplane.TrainerUtil import TrainerUtil
from nerf_triplane.network import NeRFNetwork
from nerf_triplane.provider import NeRFDataset
from nerf_triplane.utils import *

# torch.autograd.set_detect_anomaly(True)
# Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=200000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096 * 16,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--bg_img', type=str, default='', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1,
                        help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8,
                        help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0,
                        help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01,
                        help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2,
                        help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='',
                        help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000,
                        help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true',
                        help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    parser.add_argument('--asr_model', type=str, default='deepspeech')
    # parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    parser.add_argument('--face_optimize', action='store_true',
                        help="是否对生成的视频人脸使用GFPGAN进行优化，配置即为True，不配置即为False")

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    if opt.test and False:
        opt.smooth_path = True
        opt.smooth_eye = True
        opt.smooth_lips = True

    opt.cuda_ray = True
    # assert opt.cuda_ray, "Only support CUDA ray mode."

    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    # if opt.finetune_lips:
    #     # do not update density grid in finetune stage
    #     opt.update_extra_interval = 1e9

    print(f'参数：{opt}')

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'驱动device：{device}')

    start1 = time.time()
    model = NeRFNetwork(opt)
    print(f'初始化NeRFNetwork用时：{time.time() - start1}')

    # 执行推理时的参数：Namespace(path='data/ajian1016/', O=True, test=True, test_train=True,
    #                      data_range=[0, -1], workspace='trial_ajian1016_torso/', seed=0, iters=200000, lr=0.01,
    #                      lr_net=0.001, ckpt='latest', num_rays=65536, cuda_ray=True, max_steps=16, num_steps=16,
    #                      upsample_steps=0, update_extra_interval=16, max_ray_batch=4096, warmup_step=10000, amb_aud_loss=1,
    #                      amb_eye_loss=1, unc_loss=1, lambda_amb=0.0001, fp16=True, bg_img='', fbg=False, exp_eye=True,
    #                      fix_eye=-1, smooth_eye=False, torso_shrink=0.8, color_space='srgb', preload=0, bound=1, scale=4,
    #                      offset=[0, 0, 0], dt_gamma=0.00390625, min_near=0.05, density_thresh=10, density_thresh_torso=0.01,
    #                      patch_size=1, init_lips=False, finetune_lips=False, smooth_lips=False, torso=True, head_ckpt='', gui=False,
    #                      W=450, H=450, radius=3.35, fovy=21.24, max_spp=1, att=2, aud='./data/ajian1016/1697958002734092.npy', emb=False,
    #                      ind_dim=4, ind_num=10000, ind_dim_torso=8, amb_dim=2, part=False, part2=False, train_camera=False, smooth_path=True,
    #                      smooth_path_window=7, asr=False, asr_wav='', asr_play=False, asr_model='deepspeech', face_optimize=False,
    #                      asr_save_feats=False, fps=25, l=10, m=50, r=10)
    # manually load state dict for head
    # 推理时，opt.torso=True,opt.head_ckpt=''
    if opt.torso and opt.head_ckpt != '':

        model_dict = torch.load(opt.head_ckpt, map_location='cpu')['model']

        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")

            # freeze these keys
        for k, v in model.named_parameters():
            if k in model_dict:
                print(f'[INFO] freeze {k}, {v.shape}')
                v.requires_grad = False

    print('模型加载完成!')

    criterion = torch.nn.MSELoss(reduction='none')

    # 推理时，opt.test=True
    if opt.test:
        if opt.gui:
            metrics = []  # use no metric in GUI for faster initialization...
        else:
            # metrics = [PSNRMeter(), LPIPSMeter(device=device)]
            # 加载了这三个Meter对象
            metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

        trainer = TrainerUtil(name='ngp',  # 实验的名称
                              opt=opt,  # 参数
                              model=model,  # 模型
                              device=device,  # 设备
                              workspace=opt.workspace,  # 保存日志和检查点的工作空间
                              criterion=criterion,  # 损失函数，如果为None，则假设在train_step中内联实现
                              fp16=opt.fp16,  # AMP优化级别，推理时opt.fp16=True
                              metrics=metrics,  # 评估指标，如果为None，则使用val_loss衡量性能，否则使用第一个指标
                              use_checkpoint=opt.ckpt,  # 初始化时使用的检查点
                              # 以下为新加的配置参数
                              use_tensorboardX=False, # 推理时不使用tensorboard进行日志记录
                              )
        # 推理时opt.test_train=True
        if opt.test_train:
            start1 = time.time()
            test_set = NeRFDataset(opt, device=device, type='train')
            print(f'初始化NeRFDataset用时s:{time.time()-start1}')
            # a manual fix to test on the training dataset
            test_set.training = False
            test_set.num_rays = -1

            start1 = time.time()
            test_loader = test_set.dataloader()
            print(f'加载推理dataloader用时s:{time.time() - start1}')
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        # temp fix: for update_extra_states
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        # if opt.gui:
        #     from nerf_triplane.gui import NeRFGUI
        #
        #     # we still need test_loader to provide audio features for testing.
        #     with NeRFGUI(opt, trainer, test_loader) as gui:
        #         gui.render()

        #else:
        ### test and save video (fast)
        # 这是最终推理时进入的方法
        print('\n\n\n★★★★★★★★★★-------推理开始执行...-------★★★★★★★★★★')
        start1 = time.time()
        trainer.test(test_loader)
        print(f'★★★★★★★★★★-------推理结束！用时:{time.time()-start1}s-------★★★★★★★★★★\n\n\n')

        # Ajian 20231024:将此屏蔽
        # ### evaluate metrics (slow)
        # if test_loader.has_gt:
        #     trainer.evaluate(test_loader)



    else:
        print('进入执行！')

        optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr, opt.lr_net), betas=(0, 0.99), eps=1e-8)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        assert len(
            train_loader) < opt.ind_num, f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to this number!"

        # temp fix: for update_extra_states
        model.aud_features = train_loader._data.auds
        model.eye_area = train_loader._data.eye_area
        model.poses = train_loader._data.poses

        # decay to 0.1 * init_lr at last iter step
        if opt.finetune_lips:
            scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                            lambda iter: 0.05 ** (iter / opt.iters))
        else:
            scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                            lambda iter: 0.5 ** (iter / opt.iters))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        eval_interval = max(1, int(5000 / len(train_loader)))
        trainer = TrainerUtil('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                              criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                              scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                              eval_interval=eval_interval)
        with open(os.path.join(opt.workspace, 'opt.txt'), 'a') as f:
            f.write(str(opt))
        if opt.gui:
            # with NeRFGUI(opt, trainer, train_loader) as gui:
            #     gui.render()
            print('没启用原GUI，代码被屏蔽了！')
            pass

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epochs = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            print(f'[INFO] 开始训练，最大步骤={max_epochs}')
            trainer.train(train_loader, valid_loader, max_epochs)
            print('训练完成！')

            # free some mem
            del train_loader, valid_loader
            torch.cuda.empty_cache()

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

