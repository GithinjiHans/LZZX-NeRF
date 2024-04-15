import json
import os
import sys
from argparse import ArgumentParser

from PIL import Image

parser = ArgumentParser()
parser.add_argument('--video', type=str, help='源视频路径,/xx/xxx/xxx.mp4')
parser.add_argument('--cropX', type=int, help='截取位置距离最左边的距离')
parser.add_argument('--cropY', type=int, help='截取位置距离最顶边的距离')
parser.add_argument('--cropW', type=int, help='截取范围的宽度，需要确保为偶数')
parser.add_argument('--cropH', type=int, help='截取范围的高度，需要确保为偶数')
args = parser.parse_args()
# 在计算机图形中，像素通常是按照2的倍数进行排列的，所以这里的长宽尺寸必须是一个偶数！
if args.cropW is None or args.cropW % 2 != 0 or args.cropH is None or args.cropH % 2 != 0:
    print('裁切的长宽值不能为空，且必须为偶数！！！')
    sys.exit()

# 截取的图片的长宽
targetW = args.cropW
targetH = args.cropH

# 截取图片的起始位置x、y，由自行在ps中测量好之后通过参数传入
targetX = args.cropX
targetY = args.cropY

# 目标视频
targetVideo = args.video
# 截图存放位置
tmpFrame = os.path.join(os.path.dirname(targetVideo), 'targetFrame.jpg')
cropSavePath = os.path.join(os.path.dirname(targetVideo), '待手动处理为纯背景的图片.jpg')

# 截取一帧图片
os.system(f'ffmpeg -i "{targetVideo}" -y -f image2 -ss 3 -vframes 1 {tmpFrame}')
# 裁切截图，从指定位置裁切指定长宽
img = Image.open(tmpFrame)
cropImg = img.crop((targetX, targetY, targetX + targetW, targetY + targetH))
cropImg.save(cropSavePath)
os.remove(tmpFrame)
# 裁切视频，从指定位置裁切指定长宽的视频作为训练用视频
trainVideo = os.path.join(os.path.dirname(targetVideo), "trainVideo.mp4")
os.system(f'ffmpeg -y -colorspace bt709 -i "{targetVideo}" -vf crop={targetW}:{targetH}:{targetX}:{targetY} -c:v libx264 -c:a aac -r 25 "{trainVideo}"')

# 等待手动处理完成
print(f'将【{cropSavePath}】处理好之后，输入y继续...')
while True:
    tip = input('处理好了吗？(y/n)')
    if tip == 'y':
        break
readyVideo = os.path.join(os.path.dirname(targetVideo), 'readySourceVideo.mp4')
# 将裁切的图贴到原视频对应位置上去，并去掉声音，制作为待拼接的视频
os.system(
    f'ffmpeg -y -colorspace bt709 -i "{targetVideo}" -i "{cropSavePath}" -c:v libx264 -c:a aac -r 25 -an -filter_complex overlay={targetX}:{targetY} -colorspace bt709 {readyVideo}')

# 将距离参数记录到json文件中，方便训练完成之后读取参数进行视频拼接
parameter = {'x': targetX, 'y': targetY, 'w': targetW, 'h': targetH}
with open(os.path.join(os.path.dirname(targetVideo), 'video_crop_parameter.json'), 'w') as f:
    json.dump(parameter, f)
# 将处理好的背景图作为训练用的背景图
os.rename(cropSavePath, os.path.join(os.path.dirname(cropSavePath), 'bc.jpg'))

print(f'操作完成，请使用【{trainVideo}】进行训练！！')
