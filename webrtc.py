import concurrent
import os
import random
import re
import shutil
import time
from multiprocessing import freeze_support

import gradio as gr
import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from GradioSession import GradioSession
from HubertInferenceMQ import HubertInferenceMQ
from data_utils.HubertBean import HubertBean
from mq_consume.ConsumeMQByWebRTC import ConsumeMQByWebRTC
from nerf_triplane.provider_for_inference import NeRFDataset

modelBasePath = r'./data'
models = ['--请选择--']
for m in os.listdir(modelBasePath):
    if os.path.isdir(os.path.join(modelBasePath, m)) and not m.startswith('.'):
        models.append(m)
StreamType = 'webrtc'  # 配置播放是用webrtc还是flv
PublicHttpDomain = '3.140.140.83'
PrivateIpDomain = '127.0.0.1'

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
        return 'http://'+PublicHttpDomain+':8080/live/av_' + sessionId

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
        player.src = ''' + get_jsplayer_src() + ''';
        document.head.appendChild(player)
        //添加一个video
        document.querySelector("#resultVideoDiv .empty").innerHTML=
            `<div style="width:100%;height:512px;">
                <video style="width:100%;height:100%;" preload="auto" autoplay controls defaultMuted=false muted=false id="videoDom">
                </video>
            </div>`
        //播放hls
        function playHLS(url){
            let v = document.getElementById("videoDom");
            if (typeof Hls=='object' && Hls.isSupported()) {
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
            if (typeof mpegts=='object' && mpegts.getFeatureList().mseLivePlayback) {
                var videoElement = document.getElementById('videoDom');
                var player = mpegts.createPlayer({
                    type: 'flv',  // could also be mpegts, m2ts, flv
                    isLive: true,
                    url: flvURL
                },{
                    liveBufferLatencyChasing: false,   //自动追帧
                    liveBufferLatencyMaxLatency: 0.9, // 最大缓存时间
                    liveBufferLatencyMinRemain: 0.1, // 最小缓存时间
                });
                player.on(mpegts.Events.ERROR, e=> {
                    console.log('flv播放发生异常！')
                    player.unload();
                    player.load();
                    player.play();
                });
                player.on(mpegts.Events.LOADING_COMPLETE, (e) => {
                  console.log("直播已结束");
                });
                player.on(mpegts.Events.STATISTICS_INFO, (e) => {
                  console.log("解码帧：",e.decodedFrames); // 已经解码的帧数
                });
                player.attachMediaElement(videoElement);
                player.load();
                player.play();
                document.getElementById("videoDom").muted = false;
                //显示视频界面
                document.querySelector("#resultVideoDiv").style.display='block';
            }
        }
        //播放webrtc
        function playWEBRTC(url){
            if(typeof JSWebrtc=='object'){
                new JSWebrtc.Player(url, { video: document.getElementById('videoDom'), autoplay: true});
                document.querySelector("#resultVideoDiv").style.display='block';
                document.getElementById("videoDom").muted = false;
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
                                let url = top.location.href+'static/generate-mp4/'+n
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
                                    let s = log.replace('##PLAY##','')
                                    if(s.startsWith('http')){
                                        //playHLS(s) 视频切换时，效果不佳！
                                        playFLV(s)
                                    }
                                    if(s.startsWith('webrtc')){
                                        playWEBRTC(s)
                                    }
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
                if(d.status==="success"){
                    if(d.hls!=""){
                        //playHLS(d.hls) 视频切换时效果不佳!
                    }
                    if(d.flv!=""){
                        playFLV(d.flv)
                    }
                    if(d.rtc!=""){
                        playWEBRTC(d.rtc)
                    }
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
                    submitBtn.innerText='音频上传中...';
                }
                if(log==='upload check'){
                    submitBtn.disabled=true;
                    submitBtn.innerText='音频校验中...';
                }
                if(log==='upload success'){
                    submitBtn.innerText='提交';
                    submitBtn.disabled=false;
                }
                if(log==='upload fail'){
                    alert('音频处理失败了！请换一个音频或者重试一次吧！');
                    submitBtn.innerText='提交';
                    submitBtn.disabled=true;
                }
                return this.textContent = log;
            }
        });
        
    }
'''

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


with gr.Blocks(title="Ajian Digital Human Show") as page:
    def action(model):
        '''hubert音频驱动，整体处理方法'''
        yield log_out('数据集准备中......')
        start1 = time.time()
        # 加载loader
        loader = session.nerfDataSetInstance.dataloader()
        # temp fix: for update_extra_states
        session.inferenceInstance.model.aud_features = loader._data.auds
        session.inferenceInstance.model.eye_areas = loader._data.eye_area
        yield log_out('数据集准备完毕!')
        print(f'加载推理dataloader用时s:{time.time() - start1}')
        # 使用hubert时，用初始化好的对象进行推理
        yield log_out('##PLAY##' + get_jsplayer_url(session.sessionId))

        # 执行推理
        def doInfer():
            session.inferenceInstance.do_inference(loader, session.mqInstance, session.audio_full_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(doInfer)
            runN = 0
            line_str = '...'
            while not future.done():
                runN += 1
                if runN == 1:
                    yield log_out('推理模型准备中......')
                elif runN == 2:
                    yield log_out('推理素材加载中......')
                elif runN == 3:
                    yield log_out('推理音频加载中......')
                elif runN == 4:
                    yield log_out('推理开始......')
                elif runN == 5:
                    yield log_out(line_str)
                elif runN == 6:
                    yield log_out(line_str + '...')
                elif runN == 7:
                    yield log_out('流媒体服务器开始推流...')
                else:
                    yield log_out(line_str)
                    line_str += '...'
                time.sleep(1)

        global inferenceFileName
        if inferenceFileName:
            yield log_out('视频文件存储中...')
            # 循环检测状态，等待文件存储完成
            line_str = '...'
            while True:
                time.sleep(1)
                yield log_out(line_str)
                line_str += '...'
                if session.mqInstance.rtmpConfigDict["PushFlag"] is None:
                    # session.mqInstance.pushAndSaveFrames_done()
                    # session.mqInstance.pushWaitVideoForModel(session.mqInstance.modelFullPath,
                    #                                          session.mqInstance.rtmpConfigDict["remoteRtmpURL"])
                    break
            yield log_out('##SUCCESS##' + inferenceFileName)

        yield log_out('本次推理视频生成完成！')


    with gr.Row():
        with gr.Column():
            def audioUploaded(files):
                '''音频文件上传完成，会触发此处的回调，在上传完成之后，立即进行音频预处理！'''
                if files is None:
                    return
                yield 'upload begin'
                audio = files
                if isinstance(files, list):
                    audio = files[0]
                if session.selectModelName is None or session.selectModelName == '--请选择--' or not isinstance(
                        session.selectModelName, str):
                    yield 'upload fail'
                    print('请先选择模型再上传音频！')
                    return
                # 处理音频
                try:
                    session.hubertNpy = hubertInstance.get_aud_features(wav_path=audio.name)
                    audio_full_path = os.path.join(BasePath, modelBasePath, session.selectModelName,
                                                   str(time.time()).replace('.', '') + '.wav')
                    audio_full_path = audio_full_path.replace('/./', '/')
                    shutil.copy2(audio.name, audio_full_path)
                    session.audio_full_path = audio_full_path
                    yield 'upload check'
                    # 加载音频特征向量
                    session.nerfDataSetInstance.init_aud_features(session.hubertNpy)
                    # 开启推流进程
                    global inferenceFileName
                    inferenceFileName = session.mqInstance.pushAndSaveFrames_init(
                        infer_mp4_save_path=os.path.join(BasePath, 'static/generate-mp4/'),
                        audio_full_path=audio_full_path)
                    print(f'生成文件名 ：{inferenceFileName}')
                    # 等待推理子进程准备完毕
                    # while True:
                    #     if session.mqInstance.rtmpConfigDict["PushFlag"] == 'Ready':
                    #         session.mqInstance.rtmpConfigDict["PushFlag"] = 'Yes'
                    #         break
                    # 音频处理完成，让前端页面提交按钮置为可用状态
                    yield 'upload success'
                    # 等待视频
                    # session.mqInstance.pushWaitVideoForModel(session.mqInstance.modelFullPath,
                    #     remoteRtmpServerURL=session.mqInstance.rtmpConfigDict["remoteRtmpURL"])
                except Exception as e:
                    print(f'音频处理失败！{e}')
                    import logging
                    logging.exception(e)
                    yield 'upload fail'


            def modelSelectedShowVideoStream(model, remoteRtmpServerURL):
                '''选择模型之后，就启动推流端，开始推送静默视频流'''
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


            sessionIdLabel = gr.Label(label='GRADIO_SESSION_ID', elem_id="GRADIO_SESSION_ID", visible=False, value="")
            remoteRtmpServerURL = gr.Textbox(label="直播间rtmp地址(需要时才配置)",
                                             placeholder="支持b站、抖音、快手等直播间，格式如：rtmp://xxxx.xxx?streamname=xxx，需要推流到直播间时才配置")
            model = gr.Dropdown(
                choices=models, value=models[0], label="选择模型", elem_id="modelSelectDom"
            )
            flvStramDom = gr.Dropdown(visible=False, label='接收服务端推流的flv播放地址，隐藏',
                                      elem_id="flvStramDomInput", allow_custom_value=True)
            model.change(modelSelectedShowVideoStream,
                         [model, remoteRtmpServerURL], flvStramDom)
            with gr.Tab('上传音频'):
                audioUploadInput = gr.Dropdown(visible=False, label='接收音频文件上传成功的状态，隐藏',
                                               elem_id="audioUploadSuccess", allow_custom_value=True)
                audio2 = gr.File(label='上传录音文件', file_types=['.wav'])
                audio2.change(audioUploaded, [audio2], audioUploadInput)
            btn = gr.Button("提交", variant="primary", elem_id="submitBtn")
        with gr.Column():
            msg = gr.Dropdown(visible=False, label='接收日志文本，隐藏', elem_id="logDivText", allow_custom_value=True)
            logCom = gr.Label(label='运行状态', elem_id="logShowDiv", value='')
            gr.Label(label='实时视频', elem_id="resultVideoDiv", visible=False)

    btn.click(
        action,
        inputs=[model],
        outputs=[msg],
    )
    page.load(_js=_script)

BasePath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI()
app.mount('/static', StaticFiles(directory=os.path.join(BasePath, 'static')), 'static')
if __name__ == '__main__':
    print(f'UI启动参数：{BasePath}')
    freeze_support()
    # 初始化参数
    hubertInstance = HubertBean()
    session = GradioSession(str(random.randint(10000, 99999999)))
    session.mqInstance = get_mqInstance_byStreamType(session.sessionId)
    session.inferenceInstance = HubertInferenceMQ()
    # 启动web
    page2 = page.queue()
    app = gr.mount_gradio_app(app, page2, path='/')

    uvicorn.run(app=app, host="0.0.0.0", port=7860)
