import concurrent
import os
import random
import re
import shutil
import time
from multiprocessing import freeze_support

import uvicorn
import flask
from flask_cors import CORS
from flask import request, jsonify
from starlette.staticfiles import StaticFiles
from gevent.pywsgi import WSGIServer

from GradioSession import GradioSession
from HubertInferenceMQ import HubertInferenceMQ
from data_utils.HubertBean import HubertBean
from mq_consume.ConsumeMQByWebRTC import ConsumeMQByWebRTC
from nerf_triplane.provider_for_inference import NeRFDataset


app = flask.Flask(__name__)
CORS(app)

modelBasePath = r'./data'
models = ['--请选择--']
for m in os.listdir(modelBasePath):
    if os.path.isdir(os.path.join(modelBasePath, m)) and not m.startswith('.'):
        models.append(m)
StreamType = 'webrtc'  # 配置播放是用webrtc还是flv
PublicHttpDomain = 'realtime-lipsync.bhuman.ai'
PrivateIpDomain = '127.0.0.1'
BasePath = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def get_jsplayer_src():
    if StreamType == 'webrtc':
        return 'location.href + "static/jswebrtc.min.js"'
    elif StreamType == 'flv':
        return 'location.href+"static/mpegts.js"'
    elif StreamType == 'hls':
        return 'https://cdn.bootcdn.net/ajax/libs/hls.js/1.4.12/hls.min.js'


def get_mqInstance_byStreamType(sessionId):
    if StreamType == 'webrtc':
        return ConsumeMQByWebRTC(sessionId,PrivateIpDomain)

def get_jsplayer_url(sessionId):
    if StreamType == 'webrtc':
        return 'webrtc://'+PublicHttpDomain+'/live/av_' + sessionId
    elif StreamType == 'flv':
        return 'https://'+PublicHttpDomain+':8080/live/av_' + sessionId

hubertInstance: HubertBean = None
session: GradioSession = None
inferenceFileName: str = None

def log_out(new_log):
    # 要把控制台格式化转义字符替换掉，不然显示乱码，千万注意，天坑！！！
    new_log = re.sub(r'\x1b\[\d*(;\d+)*m', '', new_log)
    if new_log.startswith('##SUCCESS##'):
        return new_log
    if new_log.startswith('##PLAY##'):
        return new_log
    return '[INFO] ' + new_log


@app.route('/api/inference', methods=['GET'])
def action():
    '''hubert音频驱动，整体处理方法'''
    start1 = time.time()
    # yield log_out('数据集准备中......')
    # 加载loader
    loader = session.nerfDataSetInstance.dataloader()
    # temp fix: for update_extra_states
    session.inferenceInstance.model.aud_features = loader._data.auds
    session.inferenceInstance.model.eye_areas = loader._data.eye_area
    # yield log_out('数据集准备完毕!')
    print(f'加载推理dataloader用时s:{time.time() - start1}')
    # 使用hubert时，用初始化好的对象进行推理
    # yield log_out('##PLAY##' + get_jsplayer_url(session.sessionId))

    # 执行推理
    def doInfer():
        session.inferenceInstance.do_inference(loader, session.mqInstance, session.audio_full_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(doInfer)
        runN = 0
        line_str = '...'
        while not future.done():
            runN += 1
            time.sleep(1)

    global inferenceFileName
    if inferenceFileName:
        line_str = '...'
        while True:
            time.sleep(1)
            line_str += '...'
            if session.mqInstance.rtmpConfigDict["PushFlag"] is None:
                break

    response = flask.make_response("")
    response.status_code = 200
    return response

@app.route('/api/audio_upload', methods=['POST'])
def audioUploaded():
    '''音频文件上传完成，会触发此处的回调，在上传完成之后，立即进行音频预处理！'''
    if request.files is None:
        return
    print('request.files:', request.files['audio']) 
    audio = request.files['audio']
    print('audio:', audio)
    print('audio:', audio.filename)
    # save audio in tmp folder
    audio.save(audio.filename)
    if session.selectModelName is None or session.selectModelName == '--请选择--' or not isinstance(
            session.selectModelName, str):
        print('请先选择模型再上传音频！')
        return
    # 处理音频
    try:
        session.hubertNpy = hubertInstance.get_aud_features(wav_path=audio.filename)
        audio_full_path = os.path.join(BasePath, modelBasePath, session.selectModelName,
                                        str(time.time()).replace('.', '') + '.wav')
        audio_full_path = audio_full_path.replace('/./', '/')
        src_file=audio.filename
        os.system(f'ffmpeg -i {src_file} -ac 1 -ar 16000 {audio_full_path}')
        session.audio_full_path = audio_full_path
        session.nerfDataSetInstance.init_aud_features(session.hubertNpy)
        global inferenceFileName
        inferenceFileName = session.mqInstance.pushAndSaveFrames_init(
            infer_mp4_save_path=os.path.join(BasePath, 'static/generate-mp4/'),
            audio_full_path=audio_full_path)
        print(f'生成文件名 ：{inferenceFileName}')
        return '{"status":"success"}'
    except Exception as e:
        print(f'音频处理失败！{e}')
        import logging
        logging.exception(e)
        return '{"status":"fail"}'

@app.route('/api/model_select', methods=['POST'])
def modelSelectedShowVideoStream():
    remoteRtmpServerURL=None
    '''选择模型之后，就启动推流端，开始推送静默视频流'''
    data=request.get_json()
    model = data['model']
    if model is None or model == '--请选择--':
        return '{"status":"fail"}'
    # 关停之前在进行的静默视频推流
    if remoteRtmpServerURL is None or remoteRtmpServerURL == "":
        remoteRtmpServerURL = None
    session.mqInstance.stopWaitVideoForModel()
    # 初始化推理环境，为推理做准备
    # 补齐参数
    session.inferenceInstance.opt.path = os.path.join(modelBasePath, model)
    session.inferenceInstance.opt.workspace = f'trial_{model}'
    session.inferenceInstance.opt.torso = False
    session.inferenceInstance.trainer.workspace = session.inferenceInstance.opt.workspace
    # 初始化trainer的workspace信息
    session.inferenceInstance.trainer.init_workspace(session.inferenceInstance.opt.workspace)
    # 初始化dataset
    session.nerfDataSetInstance = NeRFDataset(session.inferenceInstance.opt,
                                                device=session.inferenceInstance.device, type='train')
    session.nerfDataSetInstance.training = False
    session.nerfDataSetInstance.num_rays = -1
    try:
        # 开始推送模型默认的视频流
        session.selectModelName = model
        r = session.mqInstance.pushWaitVideoForModel(
            os.path.join(BasePath, modelBasePath, model).replace('/./', '/'),
            remoteRtmpServerURL=remoteRtmpServerURL)
        # 返回前端
        if r is not None:
            if StreamType=='webrtc':
                msg = '{"status":"success","rtc":"'+get_jsplayer_url(session.sessionId)+'"}'
            elif StreamType=='flv':
                msg = '{"status":"success","flv":"'+get_jsplayer_url(session.sessionId)+'"}'
            print('推流开始，返回：' + msg)
        else:
            msg = '{"status":"fail"}'

        return msg
    except Exception as e:
        print(f'模型推流失败:{e}')
        return '{"status":"fail"}'


if __name__ == '__main__':
    print(f'UI启动参数：{BasePath}')
    freeze_support()
    # 初始化参数
    hubertInstance = HubertBean()
    session = GradioSession(str(random.randint(10000, 99999999)))
    session.mqInstance = get_mqInstance_byStreamType(session.sessionId)
    session.inferenceInstance = HubertInferenceMQ()
    # 启动web服务
    http_server = WSGIServer(('0.0.0.0', 7860), app)
    http_server.serve_forever()
