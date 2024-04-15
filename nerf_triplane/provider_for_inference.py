import gc
import numba
import os
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import tqdm
import trimesh
from numba import njit
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from .utils import get_audio_features, get_rays, get_bg_coords


# ti.init()


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50

# @njit
def load_ori_imgs_lms_byNumbaJIT(root_path: str, img_id: str, exp_eye: bool, finetune_lips: bool, H=1080, W=1080):
    lms = np.loadtxt(os.path.join(root_path, 'ori_imgs', str(img_id) + '.lms'))  # [68, 2]
    lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())  # actually lower half area
    xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
    ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
    o3 = None
    if exp_eye:
        xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
        ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
        o3 = list([xmin, xmax, ymin, ymax])
    o4 = None
    if finetune_lips:
        lips = slice(48, 60)
        xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
        ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
        # padding to H == W
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        l = max(xmax - xmin, ymax - ymin) // 2
        xmin = max(0, cx - l)
        xmax = min(H, cx + l)
        ymin = max(0, cy - l)
        ymax = min(W, cy + l)
        o4 = list([xmin, xmax, ymin, ymax])

    return list([xmin, xmax, ymin, ymax]), list([lh_xmin, lh_xmax, ymin, ymax]), o3, o4


@njit
def load_ori_imgs_allLms_byNumbaJIT(allLms: numba.typed.List, allImgIds: numba.typed.List,
                                    exp_eye: bool, finetune_lips: bool, H=1080, W=1080):
    returnR1 = numba.typed.List()
    returnR1.append([0, 0, 0, 0])
    returnR2 = numba.typed.List()
    returnR2.append([0, 0, 0, 0])
    returnR3 = numba.typed.List()
    returnR3.append([0, 0, 0, 0])
    returnR4 = numba.typed.List()
    returnR4.append([0, 0, 0, 0])

    # aIndex = 0
    # for lms in allLms:
    #     lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())  # actually lower half area
    #     xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
    #     ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
    #     returnR1[aIndex] = [xmin, xmax, ymin, ymax]
    #     returnR2[aIndex] = [lh_xmin, lh_xmax, ymin, ymax]
    #     if exp_eye:
    #         xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
    #         ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
    #         returnR3[aIndex] = [xmin, xmax, ymin, ymax]
    #     if finetune_lips:
    #         lips = slice(48, 60)
    #         xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
    #         ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
    #         # padding to H == W
    #         cx = (xmin + xmax) // 2
    #         cy = (ymin + ymax) // 2
    #         l = max(xmax - xmin, ymax - ymin) // 2
    #         xmin = max(0, cx - l)
    #         xmax = min(H, cx + l)
    #         ymin = max(0, cy - l)
    #         ymax = min(W, cy + l)
    #         returnR4[aIndex] = [xmin, xmax, ymin, ymax]

    return returnR1, returnR2, returnR3, returnR4


def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


#nerf_matrix_to_ngp_returnType = ti.types.matrix(4, 4, ti.f32)


# @ti.kernel
# def nerf_matrix_to_ngp_byTaichi(pose: ti.types.ndarray(), scale: float,
#                                 offset: ti.types.ndarray()) -> nerf_matrix_to_ngp_returnType:
#     return nerf_matrix_to_ngp_returnType([
#         [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
#         [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
#         [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
#         [0, 0, 0, 1],
#     ])


# returnMatrix = ti.ndarray(dtype=nerf_matrix_to_ngp_returnType, shape=10000)


@njit
def nerf_matrix_to_ngp_forAllFrames_byNumbaJIT(transformArrays: numba.typed.List, scale: float,
                                               offset: numba.typed.List):
    pose = transformArrays[0]
    returnR = numba.typed.List()
    returnR.append(np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0., 0., 0., 1.]
    ], dtype=np.float32))

    pindex = 0
    for pose in transformArrays:
        if pindex > 0:
            returnR.append(np.array([
                [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
                [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
                [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
                [0., 0., 0., 1.]
            ], dtype=np.float32))
        pindex += 1
    return returnR


def smooth_camera_path(poses, kernel_size=5):
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array

    N = poses.shape[0]
    K = kernel_size // 2

    trans = poses[:, :3, 3].copy()  # [N, 3]
    rots = poses[:, :3, :3].copy()  # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses


def polygon_area(x, y):
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    print(f'[INFO] visualize poses: {poses.shape}')

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # 0 = disk, 1 = cpu, 2 = gpu
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16

        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        # load nerf-compatible format data.

        # load all splits (train/valid/test)
        if type == 'all':
            transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
            transform = None
            for transform_path in transform_paths:
                with open(transform_path, 'r') as f:
                    tmp_transform = json.load(f)
                    if transform is None:
                        transform = tmp_transform
                    else:
                        transform['frames'].extend(tmp_transform['frames'])
        # load train and val split
        elif type == 'trainval':
            with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                transform = json.load(f)
            with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                transform_val = json.load(f)
            transform['frames'].extend(transform_val['frames'])
        # only load one specified split
        else:
            # no test, use val as test
            # 推理时，直接读取的data/xxx/下的transforms_train.json
            with open(os.path.join(self.root_path, f'transforms_{"val" if type == "test" else type}.json'), 'r') as f:
                transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = int(transform['cy']) * 2 // downscale
            self.W = int(transform['cx']) * 2 // downscale

        # read images
        frames = transform["frames"]

        # use a slice of the dataset
        if self.end_index == -1:  # abuse...
            self.end_index = len(frames)

        frames = frames[self.start_index:self.end_index]

        # use a subset of dataset.
        if type == 'train':
            if self.opt.part:
                frames = frames[::10]  # 1/10 frames
            elif self.opt.part2:
                frames = frames[:375]  # first 15s
        elif type == 'val':
            frames = frames[:100]  # first 100 frames for val

        print(f'加载frames，type={type},frames type={frames.__class__}')
        print(f'[INFO] load {len(frames)} {type} frames.')

        # only load pre-calculated aud features when not live-streaming
        aud_features = None
        try:
            self.opt.audFromNdarray
        except AttributeError:
            self.opt.audFromNdarray = False

        if not self.opt.asr:
            # empty means the default self-driven extracted features.
            if self.opt.aud == '' and self.opt.audFromNdarray == False:
                if 'esperanto' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_eo.npy'))
                elif 'deepspeech' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_ds.npy'))
                    print(f"加载了deepspeech音频：{os.path.join(self.root_path, 'aud_ds.npy')}")
                # elif 'hubert_cn' in self.opt.asr_model:
                #     aud_features = np.load(os.path.join(self.root_path, 'aud_hu_cn.npy'))
                elif 'hubert' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_hu.npy'))
                    print(f"加载了hubert音频：{os.path.join(self.root_path, 'aud_hu.npy')}")
                else:
                    aud_features = np.load(os.path.join(self.root_path, 'aud.npy'))
            # cross-driven extracted features.
            elif self.opt.aud != '' and self.opt.audFromNdarray == False:
                aud_features = np.load(self.opt.aud)

            # 当设置了audFromNdarray=True时，表示音频特征数据直接传入的是一个ndarray，不再从npy文件中读取，该ndarray在初始化dataset时可能不会传入!
            if aud_features is not None:
                aud_features = self.init_aud_features(aud_features)

        # load action units
        import pandas as pd
        au_blink_info = pd.read_csv(os.path.join(self.root_path, 'au.csv'))
        au_blink = au_blink_info[' AU45_r'].values

        framesLen = len(frames)
        print(f'源视频总帧长度：{framesLen}')
        # 推导式预分配list的大小既可以节约内存又可以提高list存入数据的速度
        self.torso_img = list(0 for i in range(framesLen))
        self.images = list(0 for i in range(framesLen))

        self.poses = list(0 for i in range(framesLen))
        self.exps = []

        self.auds = list(0 for i in range(framesLen))
        self.face_rect = list(0 for i in range(framesLen))
        self.lhalf_rect = list(0 for i in range(framesLen))
        self.lips_rect = list(0 for i in range(framesLen))
        self.eye_area = list(0 for i in range(framesLen))
        self.eye_rect = list(0 for i in range(framesLen))
        #taichiOffset = np.array(self.offset)

        print('-----------此处开始将每帧图像提前进行矩阵计算，以提高加载每帧图像的速度---------')
        begin = time.time()
        transformList = numba.typed.List()
        transformList.append(np.array(frames[0]['transform_matrix'], dtype=np.float32))
        findex = 0
        for f in frames:
            if findex > 0:
                transformList.append(np.array(f['transform_matrix'], dtype=np.float32))
            findex += 1
        offset = numba.typed.List()
        offset.append(self.offset[0])
        offset.append(self.offset[1])
        offset.append(self.offset[2])

        allFramesPoses = nerf_matrix_to_ngp_forAllFrames_byNumbaJIT(transformArrays=transformList, scale=self.scale,
                                                                    offset=offset)
        print(f'----------结束每帧图像矩阵计算，用时:{time.time() - begin}s,allFramesPoses length:{len(allFramesPoses)}')

        findex = 0
        for f in tqdm.tqdm(frames, desc=f'加载每一帧图像，type={type}'):
            f_path = os.path.join(self.root_path, 'gt_imgs', str(f['img_id']) + '.jpg')
            if not os.path.exists(f_path):
                print('[WARN]', f_path, 'NOT FOUND!')
                continue

            # 加载pose姿势？
            # pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            # begin = time.time()
            # r = nerf_matrix_to_ngp_byTaichi(pose=pose, scale=self.scale, offset=taichiOffset)
            # pose = np.array(r.to_numpy(), dtype=np.float32)
            # print(f'nerf_matrix_to_ngp计算用时ms:{time.time() - begin}')

            self.poses[findex] = allFramesPoses[findex]
            # del r
            # del pose

            # preload=0：从磁盘加载，1：从CPU加载，2：从GPU加载
            # 加载gt_imgs目录下jpg图片
            if self.preload > 0:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                self.images[findex] = image
            else:
                self.images[findex] = f_path

            # load frame-wise bg

            # 加载torso_imgs下躯干png图片
            torso_img_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '.png')
            if self.preload > 0:
                torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                torso_img = torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                self.torso_img[findex] = torso_img
            else:
                self.torso_img[findex] = torso_img_path

            # find the corresponding audio to the image frame
            # 推理时，opt.asr=False,self.opt.aud在self.opt.audFromNdarray=True时，等于具体的音频npy路径，否则也为空
            if not self.opt.asr and self.opt.aud == '' and self.opt.audFromNdarray == False:
                aud = aud_features[min(f['aud_id'], aud_features.shape[0] - 1)]  # 这里即是取出每一帧视频对应的这帧声音的特征张量，存放在auds中
                self.auds[findex] = aud

            # 加载ori_imgs目录下对应图片的lms文件
            # load lms and extract face
            # lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms')) # [68, 2]
            # lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            # xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            # ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            # self.lhalf_rect.append([lh_xmin, lh_xmax, ymin, ymax])
            begin = time.time()
            o1, o2, o3, o4 = load_ori_imgs_lms_byNumbaJIT(self.root_path, f['img_id'], self.opt.exp_eye,
                                                          self.opt.finetune_lips, self.H, self.W)
            # print(f'load_ori_imgs_lms_byNumbaJIT用时ms：{time.time()-begin}')
            self.face_rect[findex] = o1
            self.lhalf_rect[findex] = o2

            # 是否需要明确控制眼睛，默认为True
            if self.opt.exp_eye:
                # eyes_left = slice(36, 42)
                # eyes_right = slice(42, 48)

                # area_left = polygon_area(lms[eyes_left, 0], lms[eyes_left, 1])
                # area_right = polygon_area(lms[eyes_right, 0], lms[eyes_right, 1])

                # # area percentage of two eyes of the whole image...
                # area = (area_left + area_right) / (self.H * self.W) * 100

                # action units blink AU45
                area = au_blink[f['img_id']]
                area = np.clip(area, 0, 2) / 2
                # area = area + np.random.rand() / 10
                self.eye_area[findex] = area
                self.eye_rect[findex] = o3

            # 使用LPIPS and landmarks微调嘴唇，默认为True
            if self.opt.finetune_lips:
                self.lips_rect[findex] = o4

            findex += 1
            # 循环结束

        # load pre-extracted background image (should be the same size as training image...)
        # 加载背景图，默认使用bc.jpg
        if self.opt.bg_img == 'white':  # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black':  # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else:  # load from file
            # default bg
            if self.opt.bg_img == '':
                self.opt.bg_img = os.path.join(self.root_path, 'bc.jpg')
            bg_img = cv2.imread(self.opt.bg_img, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255  # [H, W, 3/4]

        self.bg_img = bg_img

        self.poses = np.stack(self.poses, axis=0)

        # smooth camera path...
        # 平滑相机位置，可使画面不剧烈晃动
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)

        self.poses = torch.from_numpy(self.poses)  # [N, 4, 4]

        if self.preload > 0:
            self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]
            self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0))  # [N, H, W, C]
        else:
            self.images = np.array(self.images)
            self.torso_img = np.array(self.torso_img)

        if self.opt.asr:
            # live streaming, no pre-calculated auds
            self.auds = None
        else:
            # auds corresponding to images
            # 当设置了audFromNdarray=True时，表示音频特征数据直接传入的是一个ndarray，不再从npy文件中读取，该ndarray在初始化dataset时可能不会传入!
            if self.opt.aud == '' and self.opt.audFromNdarray == False:
                self.auds = torch.stack(self.auds, dim=0)  # [N, 32, 16]
                # 存放音频特征张量的对象就可以删除了
                del aud_features
            # auds is novel, may have a different length with images
            else:
                self.auds = aud_features

        self.bg_img = torch.from_numpy(self.bg_img)

        if self.opt.exp_eye:
            self.eye_area = np.array(self.eye_area, dtype=np.float32)  # [N]
            print(f'[INFO] eye_area: {self.eye_area.min()} - {self.eye_area.max()}')

            if self.opt.smooth_eye:

                # naive 5 window average
                ori_eye = self.eye_area.copy()
                for i in range(ori_eye.shape[0]):
                    start = max(0, i - 1)
                    end = min(ori_eye.shape[0], i + 2)
                    self.eye_area[i] = ori_eye[start:end].mean()

            self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1)  # [N, 1]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        # print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())
        # 推理时preload=0
        if self.preload > 1:
            self.poses = self.poses.to(self.device)

            if self.auds is not None:
                self.auds = self.auds.to(self.device)

            self.bg_img = self.bg_img.to(torch.half).to(self.device)

            self.torso_img = self.torso_img.to(torch.half).to(self.device)
            self.images = self.images.to(torch.half).to(self.device)

            if self.opt.exp_eye:
                self.eye_area = self.eye_area.to(self.device)

        # load intrinsics
        if 'focal_len' in transform:
            fl_x = fl_y = transform['focal_len']
        elif 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # directly build the coordinate meshgrid in [-1, 1]^2
        self.bg_coords = get_bg_coords(self.H, self.W, self.device)  # [1, H*W, 2] in [-1, 1]

        # 新增的一个自定义字段，用于设置当前dataset的总长度，方便后续程序获取
        self.dataLoaderSetSize = 0

    def init_aud_features(self, aud_features):
        aud_features = torch.from_numpy(aud_features)
        # support both [N, 16] labels and [N, 16, K] logits
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] --> [N, 29, 16]
            if self.opt.emb:
                print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                aud_features = aud_features.argmax(1)  # [N, 16]
        else:
            assert self.opt.emb, "aud only provide labels, must use --emb"
            aud_features = aud_features.long()
        print(f'[INFO] 加载音频特征向量完成: {aud_features.shape}')
        self.auds = aud_features  # 注意，此处将音频向量放到了auds中，必须要等音频向量加载完成之后，才能调用dataloader()
        return aud_features

    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def collate(self, index):
        B = len(index)  # a list of length 1
        # assert B == 1

        results = {}

        # audio use the original index
        if self.auds is not None:
            if isinstance(index, list) and len(index) > 1:
                auds = []
                for i in index:
                    auds.append(get_audio_features(self.auds, self.opt.att, i).to(self.device))
                results['auds'] = auds
            else:
                auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
                results['auds'] = auds

        # head pose and bg image may mirror (replay --> <-- --> <--).
        for i,val in enumerate(index):
            index[i] = self.mirror_index(val)

        poses = self.poses[index].to(self.device)  # [B, 4, 4]

        # 当type='val'时，self.training=False
        if self.training and self.opt.finetune_lips:
            rect = self.lips_rect[index[0]]
            results['rect'] = rect
            rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, rect=rect)
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size)

        results['index'] = index  # for ind. code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        # get a mask for rays inside rect_face
        if self.training:
            xmin, xmax, ymin, ymax = self.face_rect[index[0]]
            face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax)  # [B, N]
            results['face_mask'] = face_mask

            xmin, xmax, ymin, ymax = self.lhalf_rect[index[0]]
            lhalf_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax)  # [B, N]
            results['lhalf_mask'] = lhalf_mask

        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device)  # [1]
            if self.training:
                results['eye'] += (np.random.rand() - 0.5) / 10
                xmin, xmax, ymin, ymax = self.eye_rect[index[0]]
                eye_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax)  # [B, N]
                results['eye_mask'] = eye_mask

        else:
            results['eye'] = None

        # load bg
        bg_torso_img = self.torso_img[index]
        # 训练时，按每一个index加载
        if not self.opt.torso or self.training:
            if B == 1:
                if self.preload == 0:  # on the fly loading
                    bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                    bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
                    bg_torso_img = bg_torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                    bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
                bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (
                            1 - bg_torso_img[..., 3:])
                bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)
            else:
                # 当index有多个的时候，批次加载
                bg_torso_imgs = [None] * B
                for i,ii in enumerate(index):
                    bg_torso_img = self.torso_img[ii]
                    if self.preload == 0:  # on the fly loading
                        bg_torso_img = cv2.imread(bg_torso_img, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                        bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
                        bg_torso_img = bg_torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                        bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
                    bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (
                                1 - bg_torso_img[..., 3:])
                    bg_torso_img = bg_torso_img.view(1, -1, 3).to(self.device)
                    bg_torso_imgs[i]=bg_torso_img

        if not self.opt.torso:
            if B==1:
                bg_img = bg_torso_img
            else:
                bg_img = bg_torso_imgs
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)

        if self.training:
            bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3]

        results['bg_color'] = bg_img

        # 推理时self.training=False
        if self.opt.torso and self.training:
            bg_torso_img = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3]
            results['bg_torso_color'] = bg_torso_img

        if not self.opt.test or not self.opt.audFromNdarray:
            images = self.images[index]  # [B, H, W, 3/4]
            if self.preload == 0:
                images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3]
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                images = images.astype(np.float32) / 255  # [H, W, 3]
                images = torch.from_numpy(images).unsqueeze(0)
            images = images.to(self.device)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['images'] = images

        if self.training:
            bg_coords = torch.gather(self.bg_coords, 1, torch.stack(2 * [rays['inds']], -1))  # [1, N, 2]
        else:
            bg_coords = self.bg_coords  # [1, N, 2]
        results['bg_coords'] = bg_coords

        # results['poses'] = convert_poses(poses) # [B, 6]
        # results['poses_matrix'] = poses # [B, 4, 4]
        results['poses'] = poses  # [B, 4, 4]

        return results

    def dataloader(self, aud_size=None) -> DataLoader:
        print(f'self.training={self.training},type={self.type}')
        if self.training:
            # training len(poses) == len(auds)
            size = self.poses.shape[0]
        else:
            # test with novel auds, then use its length
            if self.auds is not None:
                print(f'self.auds.shape={self.auds.shape}')
                size = self.auds.shape[0]
            # live stream test, use 2 * len(poses), so it naturally mirrors.
            else:
                # 当没有音频输入的时候，这里决定了视频推理的最长长度为2倍原视频长度
                size = 2 * self.poses.shape[0]

        # 训练、测试时，使用batch_size=1
        if self.type=='val' or self.training:
            loader = DataLoader(dataset=list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                                num_workers=0)
            print(f'此时，loader batch_size={loader.batch_size}')
        else:
            # 推理时，加大batch_size
            loader = DataLoader(dataset=list(range(size)), batch_size=32, collate_fn=self.collate, shuffle=self.training,
                                num_workers=0)
            print(f'推理时，loader batch_size={loader.batch_size},待生成的总帧数：{size}')
        self.dataLoaderSetSize = size  # 新增的一个自定义字段，用于设置当前dataset的总长度，方便后续程序获取
        loader._data = self  # an ugly fix... we need poses in trainer.
        # do evaluate if has gt images and use self-driven setting
        loader.has_gt = (self.opt.aud == '' and self.opt.audFromNdarray == False)

        return loader
