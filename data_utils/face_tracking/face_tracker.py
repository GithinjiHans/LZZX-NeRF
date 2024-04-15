import os
import sys
import cv2
import argparse
from pathlib import Path
import torch
import numpy as np
from data_loader import load_dir
from ajian_util import AjianUtil
from facemodel import Face_3DMM
from util import *
import gc

# torch.autograd.set_detect_anomaly(True)

dir_path = os.path.dirname(os.path.realpath(__file__))


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="obama/ori_imgs", help="idname of target person"
)
parser.add_argument("--img_h", type=int, default=512, help="image height")
parser.add_argument("--img_w", type=int, default=512, help="image width")
parser.add_argument("--frame_num", type=int, default=11000, help="image number")
parser.add_argument("--arg_focal", type=int, help="焦距拟合中的arg_focal参数值，有此值则不重复进行fit_焦距拟合方法计算！")
parser.add_argument("--arg_landis", type=float,
                    help="焦距拟合中的arg_landis参数值，有此值则不重复进行fit_焦距拟合方法计算！")
args = parser.parse_args()
print(f'运行参数：{args}')

start_id = 0
end_id = args.frame_num

lms, img_paths = load_dir(args.path, start_id, end_id)
num_frames = lms.shape[0]
print(f'lms文件个数：{num_frames}')
h, w = args.img_h, args.img_w
cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()
id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
model_3dmm = Face_3DMM(
    os.path.join(dir_path, "3DMM"), id_dim, exp_dim, tex_dim, point_num
)

# only use one image per 40 to do fit the focal length
sel_ids = np.arange(0, num_frames, 40)
sel_num = sel_ids.shape[0]
arg_focal = 1600
arg_landis = 1e5

ajianUtil = AjianUtil()
print(f'[INFO] fitting focal length...')
if not args.arg_focal or not args.arg_landis:
    arg_focal, arg_landis = ajianUtil.fit_焦距拟合(lms=lms, sel_ids=sel_ids, cxy=cxy, id_dim=id_dim, sel_num=sel_num,
                                                   exp_dim=exp_dim,
                                                   model_3dmm=model_3dmm)
else:
    arg_focal = args.arg_focal
    arg_landis = args.arg_landis

print(f'[INFO] coarse fitting...')
light_para, tex_para, euler_angle, trans, exp_para, id_para, focal_length = ajianUtil.fit_粗略拟合(arg_focal=arg_focal,
                                                                                                   lms=lms,
                                                                                                   id_dim=id_dim,
                                                                                                   num_frames=num_frames,
                                                                                                   exp_dim=exp_dim,
                                                                                                   tex_dim=tex_dim,
                                                                                                   cxy=cxy,
                                                                                                   model_3dmm=model_3dmm,
                                                                                                   h=h, w=w)
ajianUtil.init_3DMM_render(arg_focal, h, w)
print('加载3DMM模型topology_info.npy完成。')

print(f'[INFO] fitting light...')
# 根据参数判断是否需要重新运行，如果指定了参数pt文件，则从文件中直接读取参数结果
track_fitting_light_returns_file = os.path.join(os.path.dirname(args.path), "track_fitting_light_returns.pt")
if os.path.exists(track_fitting_light_returns_file):
    fit_光栅_result = torch.load(track_fitting_light_returns_file)
    if fit_光栅_result:
        exp_para = fit_光栅_result['exp_para'].cuda()
        euler_angle = fit_光栅_result['euler_angle'].cuda()
        trans = fit_光栅_result['trans'].cuda()
        light_para = fit_光栅_result['light_para'].cuda()
else:
    exp_para, euler_angle, trans, light_para = ajianUtil.fit_光栅(light_para=light_para, num_frames=num_frames,
                                                                  img_paths=img_paths,
                                                                  lms=lms, tex_para=tex_para, euler_angle=euler_angle,
                                                                  trans=trans, exp_para=exp_para, id_para=id_para,
                                                                  model_3dmm=model_3dmm, focal_length=focal_length,
                                                                  cxy=cxy)
    # 将光栅操作的返回参数记录下来，方便下次运行时直接从存储的参数运行，避免每次都重新运行！
    torch.save(
        {
            "exp_para": exp_para,
            "euler_angle": euler_angle,
            "trans": trans,
            "light_para": light_para
        },
        os.path.join(os.path.dirname(args.path), "track_fitting_light_returns.pt"),
    )
gc.collect()

print(f'[INFO] fine frame-wise fitting...')
id_para, exp_para, euler_angle, trans, light_para = ajianUtil.fit_精细拟合(num_frames=num_frames, img_paths=img_paths,
                                                                           lms=lms, exp_para=exp_para, exp_dim=exp_dim,
                                                                           euler_angle=euler_angle,
                                                                           trans=trans, light_para=light_para,
                                                                           id_para=id_para, tex_para=tex_para,
                                                                           model_3dmm=model_3dmm,
                                                                           focal_length=focal_length, cxy=cxy,
                                                                           ori_img_path=args.path
                                                                           )
gc.collect()

torch.save(
    {
        "id": id_para.detach().cpu(),
        "exp": exp_para.detach().cpu(),
        "euler": euler_angle.detach().cpu(),
        "trans": trans.detach().cpu(),
        "focal": focal_length.detach().cpu(),
    },
    os.path.join(os.path.dirname(args.path), "track_params.pt"),
)
gc.collect()

print("params saved")
