import argparse
import os
import re
import shutil
import time
from typing import List

import gradio as gr

from GradioSession import GradioSession
from HubertInferenceMQ import HubertInferenceMQ
from data_utils.HubertBean import HubertBean
from mq_consume.ConsumeMQ import ConsumeMQ
from nerf_triplane.provider import NeRFDataset

modelBasePath = r'data'
models = ['--请选择--']
for m in os.listdir(modelBasePath):
    if os.path.isdir(os.path.join(modelBasePath, m)) and not m.startswith('.'):
        models.append(m)
_script = '''
    async()=>{
        //生成sessionId
        let id = localStorage.getItem('GRADIO_SESSION_ID')
        if(!id){
            id = new Date().getTime().toString()
            localStorage.setItem('GRADIO_SESSION_ID',id)
        }
        //设置sessionId的值到组件中
        window.gradio_config.components.forEach(e=>{
            if(e.type=="label" && e.props.label==='GRADIO_SESSION_ID'){
                e.props.value.label=id
                console.log(e)
            }
        })
        //添加播放器
        const player = document.createElement("script");
        player.onload = () =>  console.log("js load!") ;
        //player.src = "https://cdn.bootcdn.net/ajax/libs/flv.js/1.6.2/flv.min.js";
        player.src = location.href+"static/mpegts.js";
        //player.src = "https://cdn.bootcdn.net/ajax/libs/hls.js/1.4.12/hls.min.js";
        document.head.appendChild(player)
        //添加一个video
        /* flv使用*/
        document.querySelector("#resultVideoDiv .empty").innerHTML=
            `<div style="width:100%;height:512px;">
                <video style="width:100%;height:100%;" preload="auto" autoplay controls type="rtmp/flv" id="videoDom">
                    <source src="">
                </video>
            </div>`
        // hls使用
        /*document.querySelector("#resultVideoDiv .empty").innerHTML=
            `<div style="width:100%;height:512px;">
                <video style="width:100%;height:100%;" preload="auto" autoplay controls id="videoDom">
                    <source src="">
                </video>
            </div>`
        */
        //播放hls
        function playHLS(url){
            let v = document.getElementById("videoDom");
            if (Hls.isSupported()) {
                var hls = new Hls();
                hls.loadSource(url);
                hls.attachMedia(v);
            } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
                video.src = url;
            }
            //显示视频界面
            document.querySelector("#resultVideoDiv").style.display='block';
        }
        //播放flv方法
        function playFLV(flvURL){
            if (mpegts.getFeatureList().mseLivePlayback) {
                var videoElement = document.getElementById('videoDom');
                var player = mpegts.createPlayer({
                    type: 'flv',  // could also be mpegts, m2ts, flv
                    isLive: true,
                    url: flvURL
                });
                player.attachMediaElement(videoElement);
                player.load();
                player.play();
                //显示视频界面
                document.querySelector("#resultVideoDiv").style.display='block';
            }
        }
        //监控页面提交事件，提交之后监控日志输出及显示
        const btn=document.getElementById("submitBtn");
        if(btn){
            let clickCount=0
            btn.addEventListener("click",()=>{
                let output = document.querySelector("#logDivText .border-none");
                if(!output){
                    return false;
                }
                clickCount += 1;
                let show = document.querySelector('#logShowDiv .container')
                show.style.height='200px'
                show.style.overflowY='scroll'
                show.innerHTML=""
                if(clickCount==1){
                    Object.defineProperty(output, "value", {
                        set:  function (log) {
                            console.log('推理日志：',log)
                            if(log && log.startsWith('##SUCCESS##')){
                                let n = log.replace('##SUCCESS##','')
                                let url = top.location.href+'static/'+n
                                log = `<a href="${url}" target="_blank">点击下载本次生成的视频</a>`
                                show.innerHTML = show.innerHTML+'<br>'+log
                                show.scrollTop=show.scrollHeight
                                // 直接播放mp4
                                /*document.querySelector("#resultVideoDiv .empty").innerHTML=
                                    `<div style="width:100%;height:512px;">
                                        <video style="width:100%;height:100%;" preload="auto" autoplay controls id="videoDom">
                                            <source src="${url}" type="video/mp4"></source>
                                        </video>
                                    </div>`
                                document.querySelector("#resultVideoDiv").style.display='block'
                                */
                            }else if(log && log.startsWith('##PLAY##')){
                                //如果视频没有播放则设置播放
                                if(document.querySelector("#resultVideoDiv").style.display!='block'){
                                    //playHLS(log.replace('##PLAY##','')) 视频切换时，效果不佳！
                                    playFLV(log.replace('##PLAY##',''))
                                }
                            }else{
                                show.innerHTML = show.innerHTML+'<br>'+log
                                show.scrollTop=show.scrollHeight
                            }
                            return this.textContent = log;
                        }
                    });
                }
                //点击了一次提交之后，就不可再提交
                document.querySelector("#submitBtn").disabled=true
            })
        }
        //监控页面flv播放地址元素，有值时及时设置页面播放器开始直播
        Object.defineProperty(document.querySelector("#flvStramDomInput .border-none"), "value", {
            set:  function (msg) {
                console.log('监控到flv视频流地址：',msg)
                const d = JSON.parse(msg)
                if(d.status==="success" && d.hls!=""){
                    //playHLS(d.hls) 视频切换时效果不佳!
                    playFLV(d.flv)
                }else{
                    document.querySelector("#resultVideoDiv").style.display='none';
                }
                return this.textContent = msg;
            }
        })
        //提交按钮置为不可用
        const submitBtn = document.querySelector("#submitBtn");
        submitBtn.disabled=true
        //监控页面上传事件，上传完成提交按钮才可用
        let audio = document.querySelector("#audioUploadSuccess .border-none");
        Object.defineProperty(audio, "value", {
            set:  function (log) {
                console.log('上传log：'+log)
                if(log==='upload begin'){
                    submitBtn.disabled=true;
                    submitBtn.innerText='音频校验中...';
                }
                if(log==='upload success'){
                    submitBtn.innerText='提交';
                    submitBtn.disabled=false;
                }
                if(log==='upload fail'){
                    submitBtn.innerText='提交';
                    submitBtn.disabled=true;
                }
                return this.textContent = log;
            }
        });
        
    }
'''

hubertInstance: HubertBean = None
AllGradioSessions: List[GradioSession] = list()


def getSession(currentSession, sessionId) -> GradioSession:
    session: GradioSession = None
    for s in currentSession:
        if s.sessionId == sessionId:
            session = s
            break
    if session is None:
        for s in AllGradioSessions:
            if s.sessionId == sessionId:
                session = s
                currentSession.append(s)
                break
    if session is None:
        session = GradioSession(sessionId)
        session.mqInstance = ConsumeMQ(sessionId)
        session.inferenceInstance = HubertInferenceMQ()
        currentSession.append(session)
        AllGradioSessions.append(session)
        print(f'初始化session：{AllGradioSessions}')
    return session


def log_out(new_log):
    # 要把控制台格式化转义字符替换掉，不然显示乱码，千万注意，天坑！！！
    new_log = re.sub(r'\x1b\[\d*(;\d+)*m', '', new_log)
    if new_log.startswith('##SUCCESS##'):
        return new_log
    if new_log.startswith('##PLAY##'):
        return new_log
    return '[INFO] ' + new_log


with gr.Blocks(title="LZZX") as page:
    # 当前用户的个人参数
    currentSession = gr.State([])


    def action(model, currentSession, sessionIdLabel):
        '''hubert音频驱动，整体处理方法'''
        sessionId = sessionIdLabel['label']
        session = getSession(currentSession, sessionId)
        if session.hubertNpy is None:
            yield log_out(f'音频未处理...')
            return
        yield log_out(f'准备执行推理...')

        if session.nerfDataSetInstance is None:
            yield log_out(f'模型初始化中...')
            # 补齐参数
            session.inferenceInstance.opt.path = os.path.join(modelBasePath, model)
            session.inferenceInstance.opt.workspace = f'trial_{model}_torso'
            session.inferenceInstance.opt.torso = True
            session.inferenceInstance.trainer.workspace = session.inferenceInstance.opt.workspace
            # 初始化trainer的workspace信息
            session.inferenceInstance.trainer.init_workspace(session.inferenceInstance.opt.workspace)
            # 初始化dataset
            session.nerfDataSetInstance = NeRFDataset(session.inferenceInstance.opt,
                                                      device=session.inferenceInstance.device, type='train')
            yield log_out(f'模型加载完成...')
            # a manual fix to test on the training dataset
            session.nerfDataSetInstance.training = False
            session.nerfDataSetInstance.num_rays = -1

        # 初始化本次推理的音频
        session.nerfDataSetInstance.init_aud_features(
            session.hubertNpy)  # 原来的aud参数指向音频特征文件npy的路径，这里改成audArray，直接传入音频特征数据ndarray
        yield log_out(f'音频加载完成...')
        start1 = time.time()
        # 加载loader
        loader = session.nerfDataSetInstance.dataloader()
        # temp fix: for update_extra_states
        session.inferenceInstance.model.aud_features = loader._data.auds
        session.inferenceInstance.model.eye_areas = loader._data.eye_area
        yield log_out('数据集加载器准备完毕...')
        print(f'加载推理dataloader用时s:{time.time() - start1}')
        # 使用hubert时，用初始化好的对象进行推理
        yield log_out('视频生成中...')
        print("Session is: ",session)
        print("Session.sessionId== ",session.sessionId)
        yield log_out(
            '##PLAY##' + (runtimeParam.autodl_url + f'/live?port=1935&app=live&stream=av_{session.sessionId}').replace(
                '//live', '/live'))
        # yield log_out(
        #     '##PLAY##' + (runtimeParam.autodl_url + f'/hls/av_{session.sessionId}.m3u8').replace(
        #         '//hls', '/hls'))
        fileName = session.inferenceInstance.do_inference(loader, session.mqInstance, session.audio_full_path)
        if fileName:
            yield log_out('正在存储...')
            # 循环检测状态，等待文件存储完成
            while True:
                if session.mqInstance.rtmpConfigDict["PushFlag"] is None:
                    session.mqInstance.pushAndSaveFrames_done()
                    session.mqInstance.pushWaitVideoForModel(session.mqInstance.modelFullPath,
                                                             session.mqInstance.rtmpConfigDict["remoteRtmpURL"])
                    break
            yield log_out('##SUCCESS##' + fileName)

        yield log_out('生成完成！')

    with gr.Row():
        with gr.Column():
            def audioUploaded(files, currentSession, sessionIdLabel):
                '''音频文件上传完成，会触发此处的回调，在上传完成之后，立即进行音频预处理！'''
                print("sessionId..: ",sessionIdLabel)
                sessionId = sessionIdLabel['label']
                if files is None:
                    return
                yield 'upload begin'
                audio = files
                if isinstance(files, list):
                    audio = files[0]
                # session
                session = getSession(currentSession, sessionId)
                if session.selectModelName is None or session.selectModelName=='--请选择--':
                    yield 'upload fail'
                    print('请先选择模型再上传音频！')
                    return
                # 处理音频
                try:
                    session.hubertNpy = hubertInstance.get_aud_features(wav_path=audio.name)
                    audio_full_path = os.path.join(runtimeParam.base_path, modelBasePath,session.selectModelName,
                                                   str(time.time()).replace('.', '') + '.wav')
                    audio_full_path = audio_full_path.replace('/./', '/')
                    shutil.copy2(audio.name, audio_full_path)
                    session.audio_full_path = audio_full_path
                    # 音频处理完成，让前端页面提交按钮置为可用状态
                    yield 'upload success'
                except Exception as e:
                    print(f'音频处理失败！{e}')
                    yield 'upload fail'


            def modelSelectedShowVideoStream(model, remoteRtmpServerURL, currentSession: List[GradioSession],
                                             sessionIdLabel):
                print("SessionId: ",sessionIdLabel)
                sessionId = sessionIdLabel['label']
                print(f'当前sessionId：{sessionId}')
                '''选择模型之后，就启动推流端，开始推送静默视频流'''
                if model is None or model == '--请选择--':
                    return '{"status":"fail"}'
                # 初始化session
                session = getSession(currentSession, sessionId)
                # 开始推送模型默认的视频流
                if remoteRtmpServerURL is None or remoteRtmpServerURL == "":
                    remoteRtmpServerURL = None
                session.mqInstance.stopWaitVideoForModel()
                session.nerfDataSetInstance = None
                try:
                    session.selectModelName = model
                    r = session.mqInstance.pushWaitVideoForModel(
                        os.path.join(runtimeParam.base_path, modelBasePath, model),
                        remoteRtmpServerURL=remoteRtmpServerURL)
                    # 返回前端
                    if r is not None:
                        msg = '{"status":"success","flv":"' + (
                                runtimeParam.autodl_url + f'/live?port=1935&app=live&stream=av_{session.sessionId}').replace(
                            '//live',
                            '/live') + '","hls":"' + (
                                          runtimeParam.autodl_url + f'/hls/av_{session.sessionId}.m3u8').replace(
                            '//hls', '/hls') + '"}'
                        print('推流开始，返回：' + msg)
                    else:
                        msg = '{"status":"fail"}'

                    return msg
                except Exception as e:
                    print(f'模型推流失败:{e}')
                    return '{"status":"fail"}'


            sessionIdLabel = gr.Label(label='GRADIO_SESSION_ID', elem_id="GRADIO_SESSION_ID", visible=False, value="")
            remoteRtmpServerURL = gr.Textbox(label="直播间rtmp地址(需要时才配置)",
                                             placeholder="支持b站、抖音、快手等直播间，格式如：rtmp://xxxx.xxx?streamname=xxx，需要推流到直播间时才配置")
            model = gr.Dropdown(
                choices=models, value=models[0], label="选择模型", elem_id="modelSelectDom"
            )
            flvStramDom = gr.Dropdown(visible=False, label='接收服务端推流的flv播放地址，隐藏',
                                      elem_id="flvStramDomInput", allow_custom_value=True)
            model.change(modelSelectedShowVideoStream,
                         [model, remoteRtmpServerURL, currentSession, sessionIdLabel], flvStramDom)
            with gr.Tab('上传音频'):
                audioUploadInput = gr.Dropdown(visible=False, label='接收音频文件上传成功的状态，隐藏',
                                               elem_id="audioUploadSuccess", allow_custom_value=True)
                audio2 = gr.File(label='上传录音文件', file_types=['.wav'])
                audio2.change(audioUploaded, [audio2, currentSession, sessionIdLabel], audioUploadInput)
            btn = gr.Button("提交", variant="primary", elem_id="submitBtn")
        with gr.Column():
            msg = gr.Dropdown(visible=False, label='接收日志文本，隐藏', elem_id="logDivText", allow_custom_value=True)
            logCom = gr.Label(label='运行状态', elem_id="logShowDiv", value='')
            gr.Label(label='实时视频', elem_id="resultVideoDiv", visible=False)

    btn.click(
        action,
        inputs=[
            model, currentSession, sessionIdLabel
        ],
        outputs=[msg],
    )
    page.load(_js=_script)

parser = argparse.ArgumentParser()
parser.add_argument('--autodl_url', type=str, help="autodl平台的自定义服务中的完整url")
parser.add_argument('--base_path', type=str, default='/root/ER-NeRF/', help="项目根路径，如：/root/ER-NeRF/")
parser.add_argument('--torso', type=bool, default=False, help="是否使用身体进行推理")
runtimeParam = parser.parse_args()
print(f'参数：{runtimeParam}')

# 启动配置好的nginx
#os.system("/usr/local/nginx/sbin/nginx -s stop")
#os.system("/usr/local/nginx/sbin/nginx")
# 初始化hubert
hubertInstance = HubertBean()
# 初始化nerf
# 启动gradio
port = 7860
#share = True
if runtimeParam.autodl_url:
 #   share = False
    print(f'webui访问地址：{runtimeParam.autodl_url}')
page.queue().launch(server_name="0.0.0.0", server_port=port)
