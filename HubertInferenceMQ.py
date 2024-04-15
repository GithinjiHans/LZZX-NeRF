import asyncio
import time

from munch import DefaultMunch
from torch.utils.data import DataLoader

from nerf_triplane.TrainerUtil import TrainerUtil
from nerf_triplane.network import NeRFNetwork
from nerf_triplane.utils import *


class HubertInferenceMQ:

    def __init__(self):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError as e:
            print('Info. This pytorch version is not support with tf32.')

        # 这三个参数是具体的推理方法调用到的时候才传入的，在初始化时需要注意不要使用！
        # "path": './data/kanghui-hubert',
        # "workspace": 'trial_kanghui-hubert_torso'
        # "aud": './data/kanghui-hubert/16983227490532022_hu.npy',
        opt = {"H": 450,
               "O": True,
               "W": 450,
               "amb_aud_loss": 1,
               "amb_dim": 2,
               "amb_eye_loss": 1,
               "asr": False,
               "asr_model": 'hubert',
               "asr_play": False,
               "asr_save_feats": False,
               "asr_wav": '',
               "att": 2,
               "bg_img": '',
               "bound": 1,
               "ckpt": 'latest',
               "color_space": 'srgb',
               "cuda_ray": True,
               "data_range": [0, -1],
               "density_thresh": 10,
               "density_thresh_torso": 0.01,
               "dt_gamma": 0.00390625,
               "emb": False,
               "exp_eye": True,
               "face_optimize": False,
               "fbg": False,
               "finetune_lips": False,
               "fix_eye": -1,
               "fovy": 21.24,
               "fp16": True,
               "fps": 25,
               "gui": False,
               "head_ckpt": '',
               "ind_dim": 4,
               "ind_dim_torso": 8,
               "ind_num": 10000,
               "init_lips": False,
               "iters": 200000,
               "l": 10,
               "lambda_amb": 0.0001,
               "lr": 0.01,
               "lr_net": 0.001,
               "m": 50,
               "max_ray_batch": 4096,
               "max_spp": 1,
               "max_steps": 16,
               "min_near": 0.05,
               "num_rays": 65536,
               "num_steps": 16,
               "offset": [0, 0, 0],
               "part": False,
               "part2": False,
               "patch_size": 1,
               "preload": 0,
               "r": 10,
               "radius": 3.35,
               "scale": 4,
               "seed": 0,
               "smooth_eye": False,  # 平滑眼睛，这个效果待研究，到底是True好还是False好
               "smooth_lips": True,  # 平滑嘴唇，这个效果待研究，到底是True好还是False好
               "smooth_path": True,
               "smooth_path_window": 14, # 平滑度2~15，加大可防止头部剧烈抖动
               "test": True,
               "test_train": True,
               "torso": False, #使用身体推理，则为True，不使用身体则为False，默认不使用
               "torso_shrink": 0.8,
               "train_camera": False,
               "unc_loss": 1,
               "update_extra_interval": 16,
               "upsample_steps": 0,
               "warmup_step": 10000,
               "audFromNdarray": True  # 新增参数，表示音频特征向量直接从ndarray中获取，不从npy文件中获取！
               }

        opt = DefaultMunch.fromDict(opt)
        print(f'参数：{opt}')
        if opt.patch_size > 1:
            assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

        seed_everything(opt.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'驱动device：{self.device}')

        self.model = NeRFNetwork(opt)
        print(f'初始化NeRFNetwork完成...')

        # 推理时，opt.test=True
        self.trainer = None
        if opt.test:
            # 加载了这三个Meter对象
            metrics = [PSNRMeter(), LPIPSMeter(device=self.device), LMDMeter(backend='fan')]
            criterion = torch.nn.MSELoss(reduction='none')
            trainer = TrainerUtil(name='ngp',  # 实验的名称
                                  opt=opt,  # 参数
                                  model=self.model,  # 模型
                                  device=self.device,  # 设备
                                  workspace=None,  # 保存日志和检查点的工作空间，初始化时，先设置为None，待正式运行推理时，再设置具体的workspace
                                  criterion=criterion,  # 损失函数，如果为None，则假设在train_step中内联实现
                                  fp16=opt.fp16,  # AMP优化级别，推理时opt.fp16=True
                                  metrics=metrics,  # 评估指标，如果为None，则使用val_loss衡量性能，否则使用第一个指标
                                  use_checkpoint=opt.ckpt,  # 初始化时使用的检查点
                                  # 以下为新加的配置参数
                                  use_tensorboardX=False,  # 推理时不使用tensorboard进行日志记录
                                  )
            self.trainer = trainer
            print('初始化trainer完成...')
        self.opt = opt

    def do_inference(self, dataLoader: DataLoader, mqInstance, audio_full_path: str)->str:
        '''使用hubert执行推理
            :param dataLoader:数据加载器
            :param mqInstance:推理之后数据存放的位置
            :param audio_full_path:推理时驱动口型的音频文件全路径
        '''
        fileName = []
        asyncio.run(self.trainer.test_with_2(loader=dataLoader,
                                             audio_full_path=audio_full_path,
                                             consume_save_ts=False,
                                             push_mq=True,
                                             mqInstance=mqInstance,
                                             local_save_file_name=fileName))
        dataLoader._data.dataLoaderSetSize = 0
        if fileName is None or len(fileName)<=0:
            return ''
        return fileName[0]
