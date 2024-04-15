import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np

def extract_audio(path, out_path, sample_rate=16000):
    '''用ffmpeg命令从源视频中分离音频'''
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    '''
    -f wav:强制输入或输出文件格式。通常会自动检测输入文件的格式，并根据输出文件的文件扩展名猜测格式，因此在大多数情况下不需要此选项
    -ar 16000:设置音频采样频率。对于输出流，它默认设置为相应输入流的频率。对于输入流，此选项仅对音频抓取设备和原始解复用器有意义，并映射到相应的解复用器选项
    '''
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path, mode='wav2vec'):
    '''提取音频的特征标签'''

    print(f'[INFO] ===== extract audio labels for {path} =====')
    if mode == 'wav2vec':
        cmd = f'python nerf/asr.py --wav {path} --save_feats'
    elif mode=='deepspeech': # deepspeech
        cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
    else: #hubert 按项目作者描述，中文语音，使用hubert模型貌似效果更好！但是使用hubert模型后，就不能使用-emb选项
        cmd = f'python data_utils/hubert.py --wav {path}'

    print(f"调用【{cmd}】提取音频的特征标签开始...")
    os.system(cmd)
    print(f'[INFO] ===== 提取音频的特征标签完成！ =====')



def extract_images(path, out_path, fps=25):
    '''利用ffmpeg命令提取视频图片'''
    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    '''
    -vf fps=25: 设置帧率
    -qmin 1:未知
    -q:v 1：即-aq <q> (o)：指定音频品质（VBR）
    -start_number 0:未知，貌似是指定视频转图开始位置
    '''
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir,faceParseType='m2fp'):
    '''对指定的图片进行背景、头、脖、体的分割分离'''

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    '''
    根据其代码推测，这里使用的是face-parsing.PyTorch组件，
    其模型为该项目自带的79999_iter.pth模型文件（https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812），
    代码仓库见：https://github.com/zllrunning/face-parsing.PyTorch。
    后期优化时可以考虑更换为训练更好的BiSeNet模型文件。
    --------------------------------------------------
    20230926：更换为基于PaddleSeg的pp_liteseg模型来解析人脸
    20231017：更换为效果更佳的m2fp模型来解析人体
    '''
    print('PaddleSeg解析人脸图像开始...')
    if faceParseType == 'paddleseg':
        cmd = f'python data_utils/face_parsing_by_paddleseg/seg.py --input_dir={ori_imgs_dir} --save_dir={parsing_dir} --modelYml=data_utils/face_parsing_by_paddleseg/pp_liteseg_CelebAMask-HQ_512x512_30k.yml --model_path=data_utils/face_parsing_by_paddleseg/model.pdparams'
    elif faceParseType == 'bisenet':
        cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    elif faceParseType == 'm2fp':
        cmd = f'python data_utils/face_parsing_by_m2fp/m2fp.py --input_dir={ori_imgs_dir} --save_dir={parsing_dir}'

    print(f'解析命令：【{cmd}】')
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_landmarks(ori_imgs_dir):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')

    import face_alignment
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths):
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')


def extract_background(base_dir, ori_imgs_dir):
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')


def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        #头 这是获取的红色部分
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        #脖子 绿色部分
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        #躯干 这是获取的蓝色部分
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        #背景 这是获取的白色部分
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # get gt image
        #将每一张原始图的背景都替换为指定的的背景图
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        #将替换了背景的图片再将其头部分替换为背景的头的部分，这样就得到一张没有头的图片，即躯干图片
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        #对设置好的躯干图片进行垂直方向的像素修复
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None
            

        # neck part "vertical" in-painting...
        #对颈部图片进行垂直方向的修复
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0) # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        #对修复区域进行模糊处理，避免出现垂直线
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'), np.concatenate([torso_image, torso_alpha], axis=-1))

    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir,arg_focal=None,arg_landis=None):

    print(f'[INFO] ===== perform face tracking =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]
    print(f'得到图片的宽：{w},高：{h}')

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'
    if arg_focal and arg_landis:
        cmd += f' --arg_focal={arg_focal} --arg_landis={arg_landis}'
    print(f'执行命令：{cmd}')

    tmp_image=None
    del tmp_image
    image_paths=None
    del image_paths

    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====')


def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    import torch

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


    # train_val_split = int(valid_num*0.5)
    # train_val_split = valid_num - 25 * 20 # take the last 20s as valid set.
    train_val_split = int(valid_num * 10 / 11)

    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--asr', type=str, default='deepspeech', help="wav2vec or deepspeech or hubert")
    parser.add_argument('--face_parse', type=str, default='m2fp', help="人脸解析模型，默认使用m2fp解析，可选值：paddleseg|bisenet|m2fp")
    parser.add_argument("--arg_focal", type=int,
                        help="焦距拟合中的arg_focal参数值，有此值则不重复进行fit_焦距拟合方法计算！")
    parser.add_argument("--arg_landis", type=float,
                        help="焦距拟合中的arg_landis参数值，有此值则不重复进行fit_焦距拟合方法计算！")

    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)

    #从源视频中分离出来的音频存储目录
    wav_path = os.path.join(base_dir, 'aud.wav')
    #从源视频中切分的每一帧图像存储目录
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    #每一帧图像分离背景，标记头、脖、身体的图像存储目录
    parsing_dir = os.path.join(base_dir, 'parsing')
    #看起来还是源视频的每一帧图像，暂未明白其意义
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    #裁切图像头部之后剩下的躯干图像存储目录
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)


    # extract audio
    if opt.task == -1 or opt.task == 1:
        #用ffmpeg命令从源视频中分离音频，存储为aud.wav文件
        extract_audio(opt.path, wav_path)
        print('源视频音频aud.wav提取完成！')

    # extract audio features
    if opt.task == -1 or opt.task == 2:
        #提取音频的特征标签
        '''
        mode:wav2vec|deepspeech
            wav2vec:是Interspeech 2019 收录的文章，论文中，作者用无监督预训练的卷积神经网络来改善语音识别任务，
                提出了一种噪声对比学习二分类任务(noise contrastive binary classification task)，
                从而使得wav2vec可以在大量未标注的数据上进行训练。
                实验结果表明wav2vec预训练得到的speech representation超越了帧级别的音素分类任务并且可以显著提升ASR模型的表现，
                同时，完全卷积架构与使用的递归模型相比，可以在硬件上并行计算。
            deepspeech:利用python_speech_features库，用DeepSpeech模型提取音频的高维特征。
        '''
        extract_audio_features(wav_path, mode=opt.asr)

    # extract images
    if opt.task == -1 or opt.task == 3:
        print('提取视频每帧图片')
        extract_images(opt.path, ori_imgs_dir)

    # face parsing
    if opt.task == -1 or opt.task == 4:
        print('对指定的图片进行背景、头、脖、体的分割分离')
        #此步骤分割的质量好坏，将极大影响最终视频合成效果
        extract_semantics(ori_imgs_dir, parsing_dir,faceParseType=opt.face_parse)

    # extract bg
    if opt.task == -1 or opt.task == 5:
        print('解析背景')
        extract_background(base_dir, ori_imgs_dir)

    # extract torso images and gt_images
    if opt.task == -1 or opt.task == 6:
        print('提取替换了背景的图及躯干图片')
        extract_torso_and_gt(base_dir, ori_imgs_dir)

    # extract face landmarks
    if opt.task == -1 or opt.task == 7:
        print('提取人脸特征点')
        extract_landmarks(ori_imgs_dir)

    # face tracking
    if opt.task == -1 or opt.task == 8:
        print('执行人脸跟踪')
        face_tracking(ori_imgs_dir,opt.arg_focal,opt.arg_landis)

    # save transforms.json
    if opt.task == -1 or opt.task == 9:
        print('执行动作迁移')
        save_transforms(base_dir, ori_imgs_dir)

