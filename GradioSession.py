
class GradioSession():

    def __init__(self, sessionId: str):
        self.sessionId = sessionId
        self.mqInstance = None,  # 当前用户的mq实例
        self.inferenceInstance = None,  # 当前用户的推理主进程对象
        self.nerfDataSetInstance = None,  # 当前用户的推理dataSet对象
        self.hubertNpy = None,  # 当前用户本次推理使用的hubert音频特征向量
        self.selectModelName = None,  # 当前用户选择使用的模型名称
        self.audio_full_path = None,  # 当前用户本次推理使用的音频文件全路径
