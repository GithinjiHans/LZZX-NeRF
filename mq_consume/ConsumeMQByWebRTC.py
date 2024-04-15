import json
import multiprocessing
import os
import signal
import subprocess
import sys
import wave
from datetime import datetime
from multiprocessing import freeze_support

import cv2


def get_audio_duration(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        frames = audio_file.getnframes()
        frame_rate = audio_file.getframerate()
        duration = frames / float(frame_rate)
    return duration


def get_udp_port():
    import random
    random_number = random.randint(11001, 99999)
    return random_number


def formatPath(path: str) -> str:
    if path is None or len(path) == 0:
        return path
    if sys.platform.startswith('win'):
        path = path.replace('\\', '/')
    path = path.replace('/./', '/')
    return path


def runSubProcess(cmd: str):
    if cmd is None or len(cmd) == 0:
        return None
    command = cmd.split(' ')
    cmds = [c for c in command if c.strip() != '']
    p = None
    if sys.platform.startswith('linux'):
        print(f"启动进程，命令：{' '.join(cmds)}")
        p = subprocess.Popen(' '.join(cmds), shell=True, preexec_fn=os.setsid)
    elif sys.platform.startswith('win'):
        p = subprocess.Popen(cmds, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def stopSubProcess(p: subprocess.Popen):
    if p is not None:
        try:
            if sys.platform.startswith('linux'):
                p.terminate()
                p.kill()
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                os.system("ps -ef | grep ffmpeg | grep -v grep | awk '{print $2}' | xargs kill -9")
            elif sys.platform.startswith('win'):
                os.system(f"taskkill /t /f /pid {p.pid}")
        except Exception as e:
            print(f'kill pid fail:{e}')
        p = None


class ConsumeMQByWebRTC:
    def __init__(self, sessionId: str, rtmpIp: str):
        multiprocessing.freeze_support()
        self.rtmpConfigDict = dict()
        self.rtmpConfigDict["remoteRtmpURL"] = None  # 推流地址
        self.rtmpConfigDict["PushFlag"] = None  # 推流标志，为YES时开始推流
        self.rtmpConfigDict["videoW"] = 512  # 视频宽，默认值
        self.rtmpConfigDict["videoH"] = 512  # 视频高，默认值
        self.rtmpConfigDict["audio"] = None  # 推理时使用的驱动嘴型的音频文件全路径地址
        self.rtmpConfigDict["totalFramesNum"] = 9999999  # 推理生成的图片帧总帧数
        self.rtmpConfigDict["sessionId"] = sessionId  # 当前sessionId，推流时的key就是这个
        self.rtmpConfigDict["udpPort"] = get_udp_port()  # 当前推流udp使用的端口
        # 记录当前在推送静默视频的进程
        self.waitVideoProcess = None
        # 记录当前推理使用的模型名
        self.modelFullPath = ''
        # 记录向外推送RTMP流的进程
        self.pushRTMPProcess: subprocess.Popen = None
        self.readQueueWorkerProcess: subprocess.Popen = None
        self.rtmpStream = ' -f flv rtmp://' + rtmpIp + '/live/av_' + sessionId

    def pushGenerateFramesBytes(self, bytesVal):
        '''将生成好的图片帧二进制数据发送到流媒体中，并存储为本地文件'''
        self.rtmpConfigDict["PushFlag"] = 'YES'
        # 停掉等待视频推流
        self.stopWaitVideoForModel()
        if self.readQueueWorkerProcess is None:
            print(f'音频文件：{self.rtmpConfigDict["audio"]}')
            modelBaseDir = os.path.dirname(self.rtmpConfigDict["audio"])
            if sys.platform.startswith('win'):
                modelBaseDir = modelBaseDir.replace('\\', '/')

            cropCfg = os.path.join(modelBaseDir, 'video_crop_parameter.json')
            if not os.path.exists(cropCfg):
                command = f'ffmpeg -y -re -f image2pipe -f rawvideo -pix_fmt rgb24 \
                                                                    -s {self.rtmpConfigDict["videoW"]}x{self.rtmpConfigDict["videoH"]} -r 25 -thread_queue_size 1024 -i - \
                                                                    -thread_queue_size 1024 -i {self.rtmpConfigDict["audio"]} -c:v libx264 -c:a aac \
                                                                    -map 0:v:0 -map 1:a:0 -pix_fmt yuv420p -ac 2 -g 25 \
                                                                    -threads 2 -max_muxing_queue_size 4096 -colorspace bt709  -f mp4 {self.rtmpConfigDict["file"]} \
                                                                     -c:v libx264 -c:a aac -s {self.rtmpConfigDict["videoW"]}x{self.rtmpConfigDict["videoH"]} \
                                                                    -tune zerolatency -b:v 1500k -maxrate 1500k -minrate 1500k \
                                                                    -bufsize 50k -nal-hrd cbr -sc_threshold 0 -bsf:v h264_mp4toannexb \
                                                                    -r 25 -keyint_min 48  \
                                                                     -colorspace bt709 -pix_fmt yuv420p {self.rtmpStream} '
            else:
                audioLen = get_audio_duration(self.rtmpConfigDict["audio"])
                with open(cropCfg) as j:
                    param = json.load(j)
                if param is None or param['x'] is None or param['y'] is None:
                    param = {'x': 0, 'y': 0}
                readySourceVideo = formatPath(os.path.join(modelBaseDir, "readySourceVideo.mp4"))
                color = ' -color_primaries bt470bg -color_trc smpte170m -colorspace smpte170m '
                command = f'ffmpeg -y -re -f image2pipe -f rawvideo -pix_fmt rgb24 \
                                    -s {self.rtmpConfigDict["videoW"]}x{self.rtmpConfigDict["videoH"]} -r 25 \
                                    -thread_queue_size 1024 -i - \
                                    -thread_queue_size 1024 {color} -i "{readySourceVideo}" \
                                    -thread_queue_size 1024 -i "{self.rtmpConfigDict["audio"]}" -c:v libx264 -c:a aac {color} -profile:v main -preset ultrafast \
                                    -filter_complex "[1:v]trim=duration={audioLen},loop=100[a];[0:v]trim=duration={audioLen}[b];[a][b]overlay={param["x"]}:{param["y"]},split=2[out1][out2]" \
                                    -map [out1] -map 2:a:0  \
                                    -threads 4 -max_delay 300 -b:v 2M -maxrate 2M -bufsize 1M {color} -pix_fmt yuv420p {self.rtmpStream} \
                                    -map [out2] -map 2:a:0 {color} -pix_fmt yuv420p \
                                    -threads 4 -f mp4 "{self.rtmpConfigDict["file"]}" '
            command = command.split(' ')
            cmds = [c for c in command if c.strip() != '']
            cmds = ' '.join(cmds)
            print(f'\n\n【存储本地MP4并推流命令：{cmds}】\n\n')
            self.readQueueWorkerProcess = subprocess.Popen(cmds, stdin=subprocess.PIPE, shell=True)
        self.readQueueWorkerProcess.stdin.write(bytesVal)

    def setInferenceTotalFramesNum(self, num: int):
        '''设置本次推理总帧数'''
        self.rtmpConfigDict["totalFramesNum"] = num

    def pushAndSaveFrames_init(self, infer_mp4_save_path: str, audio_full_path: str) -> str:
        if not os.path.exists(infer_mp4_save_path):
            os.makedirs(infer_mp4_save_path)
        # 文件名
        file_name = os.path.basename(self.modelFullPath) + '-' + datetime.now().strftime('%Y%m%d%H%M%S') + '.mp4'
        self.rtmpConfigDict["file"] = formatPath(os.path.join(infer_mp4_save_path, file_name))
        # 从裁切记录文件video_crop_parameter.json中获取推理出来的视频宽高
        cropCfg = os.path.join(self.modelFullPath, 'video_crop_parameter.json')
        if os.path.exists(cropCfg):
            with open(cropCfg) as j:
                param = json.load(j)
                self.rtmpConfigDict["videoW"] = param['w']
                self.rtmpConfigDict["videoH"] = param['h']
        # 设置音频文件路径
        self.rtmpConfigDict["audio"] = formatPath(audio_full_path)
        self.rtmpConfigDict["PushFlag"] = 'YES'
        freeze_support()
        return file_name

    def pushAndSaveFrames_done(self):
        print('等待执行完成...')
        self.readQueueWorkerProcess.communicate()
        self.readQueueWorkerProcess.wait()
        stopSubProcess(self.readQueueWorkerProcess)
        self.readQueueWorkerProcess = None
        self.rtmpConfigDict["PushFlag"] = None
        print('执行完了！')

    def pushWaitVideoForModel(self, modelFullPath: str,
                              remoteRtmpServerURL: str = None) -> dict:
        '''为指定名称的model推送其默认的等待静默视频流数据，要求model对应文件夹下要有名为"wait.mp4"的文件。'''
        self.modelFullPath = modelFullPath.replace('/./', '/')
        video = formatPath(os.path.join(modelFullPath, 'wait.mp4'))
        print('wait video:' + video)
        # 停掉之前的进程
        self.stopWaitVideoForModel()
        # 设置推地址信息
        self.rtmpConfigDict["remoteRtmpURL"] = remoteRtmpServerURL
        if os.path.exists(video):
            # 读取视频基本信息
            cap = cv2.VideoCapture(video)
            self.rtmpConfigDict["videoW"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.rtmpConfigDict["videoH"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            # 循环不间断推送等待静默视频流
            cmd = f'ffmpeg -stream_loop -1 -re -i {video} \
                        -c:a aac -c:v libx264 -profile:v main -preset ultrafast -r 25 \
                        -colorspace bt709 -max_delay 100 -g 5 -b:v 900000 {self.rtmpStream}'
            self.waitVideoProcess = runSubProcess(cmd)
        else:
            print(f'模型[{modelFullPath}]下无wait.mp4文件！')

        if os.path.exists(video):
            return 'success'
        else:
            return None

    def stopWaitVideoForModel(self):
        if self.waitVideoProcess is not None:
            stopSubProcess(self.waitVideoProcess)
            self.waitVideoProcess = None
