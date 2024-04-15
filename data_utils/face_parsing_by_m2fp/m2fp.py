import os

import cv2
import configargparse
import numpy as np
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
from cv2 import MORPH_ELLIPSE, MORPH_RECT


def get_item(masks, labels, label):
    if label in labels:
        iis = [i for i, x in enumerate(labels) if x == label]
        arrs = [None] * len(iis)
        for i, v in enumerate(iis):
            arrs[i] = masks[v] * 255
        return arrs
    else:
        return []


def replace_bgArray_item(bgArray, targets, newArray):
    '''将指定的背景二维数组中寻找到的值不为0的位置的子数组，替换为newArray数组'''
    if targets is None:
        return bgArray
    for tar in targets:
        indexs = np.where(tar != 0)
        for index in zip(*indexs):  # index的值为(1,1,1)，表示第一行第一列的第一个元素
            bgArray[(index[0], index[1])] = newArray
    return bgArray


def predict(input_dir, save_dir):
    '''使用达摩M2FP多人人体解析模型进行人体语义解析'''

    img_lists = os.listdir(input_dir)

    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    print("M2FP模型开始人体解析...")
    bgImg = None
    kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (3, 3))  # 膨胀算法核，椭圆形，3x3的矩阵
    kernel2 = cv2.getStructuringElement(MORPH_RECT, (3, 3))  # 膨胀算法核，矩形，3x3的矩阵
    for t in tqdm(img_lists, desc='解析进度'):
        if not t.endswith('.jpg') and not t.endswith('.png'):
            continue
        # 读取一张图片，根据其宽高创建背景图
        if bgImg is None:
            w, h = Image.open(os.path.join(input_dir, t)).size
            # 构建一个w行h列有3个值的二维数组，每个子数组的值为(255, 255, 255)，即填充白色，作为背景图
            bgImg = np.full((w, h, 3), (255, 255, 255))
        # copy一份背景，作为本次解析的背景图
        myBgArray = bgImg.copy()

        # 进行语义分割
        result = segmentation_pipeline(os.path.join(input_dir, t))
        # 得到语义结果
        labels = result[OutputKeys.LABELS]
        masks = result['masks']
        # 获取人脸
        face = masks[labels.index('Face')] * 255
        faceArray = np.array(face)
        face = cv2.dilate(faceArray, kernel, 1)  # 进行膨胀，避免拼接后有白色空隙
        # 获取头发
        hairs = get_item(masks, labels, 'Hair')
        hArray = []  # 头发部分也进行膨胀，避免拼接后与脸部有空隙
        for h in hairs:
            hArray.append(cv2.dilate(np.array(h), kernel, 1))
        hairs = hArray
        # 获取脖子
        neck = masks[labels.index('Torso-skin')] * 255
        neck = cv2.dilate(np.array(neck), kernel, 1) # 脖子也膨胀一下
        # 获取衣服
        clothes = get_item(masks, labels, 'UpperClothes')
        clothes = cv2.dilate(np.array(clothes), kernel2, 1) # 衣服使用矩形卷积核膨胀
        # 手臂，不一定有
        LeftArm = get_item(masks, labels, 'Left-arm')
        RightArm = get_item(masks, labels, 'Right-arm')
        # 外套，不一定有
        Coats = get_item(masks, labels, 'Coat')
        # 太阳镜，不一定有
        Sunglasses = get_item(masks, labels, 'Sunglasses')
        # 围巾，不一定有
        Scarf = get_item(masks, labels, 'Scarf')
        # 裙子，不一定有
        Skirt = get_item(masks, labels, 'Skirt')
        # 裤子，不一定有
        Pants = get_item(masks, labels, 'Pants')
        # 连衣裙，不一定有
        Dress = get_item(masks, labels, 'Dress')

        # 在背景上绘制蓝色人脸
        myBgArray = replace_bgArray_item(myBgArray, [face], [0, 0, 255])
        # 头发 蓝色
        myBgArray = replace_bgArray_item(myBgArray, hairs, [0, 0, 255])
        # 脖子 绿色
        myBgArray = replace_bgArray_item(myBgArray, [neck], [0, 255, 0])
        # 获取衣服等等 以下均为红色
        myBgArray = replace_bgArray_item(myBgArray, clothes, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, LeftArm, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, RightArm, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Coats, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Sunglasses, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Scarf, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Skirt, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Pants, [255, 0, 0])
        myBgArray = replace_bgArray_item(myBgArray, Dress, [255, 0, 0])

        # 将背景数组myBgArray存储为png
        bg = Image.fromarray(np.uint8(myBgArray))
        bg.save(os.path.join(save_dir, t.replace('.jpg', '.png')))
    print(f"解析完成，结果存储于：{save_dir}")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputImgs', help='待解析的图片目录路径')
    parser.add_argument('--save_dir', type=str, default='./outputImgs/', help='解析完成后的图片目录存储路径')
    args = parser.parse_args()
    predict(input_dir=args.input_dir, save_dir=args.save_dir)

    #predict(input_dir=r'E:\测试m2fp', save_dir=r'E:\测试m2fp\gan')
