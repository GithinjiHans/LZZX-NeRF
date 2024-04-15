# 设置GPU(没有GPU则为CPU)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device', device)

# 创建一个全白的背景图像
bgImg = np.full((1080, 1080, 3), (255, 255, 255))

# 读取图片
image = Image.open(r"C:\Users\thinkpad\Downloads\hair.jpg")   # 读取图片
img_data = np.array(image)

# 找出img_data中所有元素值不为0的值的位置
non_zero_indices = np.where(img_data != 0)
# 将这些位置在bgImg中的对应值替换为[0,0,255]
#index的值为(1,1,1)，表示第一行第一列的第一个元素
for index in zip(*non_zero_indices):
    bgImg[(index[0],index[1])]=[0,0,255]

newImage = Image.fromarray(np.uint8(bgImg))
newImage.save(r'e:\\testbg.png')

