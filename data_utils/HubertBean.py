import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Processor


class HubertBean:
    @torch.no_grad()
    def __init__(self):
        start = time.time()
        self.device = "cuda:0"
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facebook/")
            self.wav2vec2_processor = Wav2Vec2FeatureExtractor.from_pretrained(path)
            self.hubert_model = HubertModel.from_pretrained(path)
            self.hubert_model.to(self.device)
        except Exception as e:
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert_model.to(self.device)
        print(f'加载模型用时：{time.time() - start}')

    def get_hubert_from_16k_wav(self, wav_16k_name):
        speech_16k, _ = sf.read(wav_16k_name)
        hubert = self.get_hubert_from_16k_speech(speech_16k)
        return hubert

    @torch.no_grad()
    def get_hubert_from_16k_speech(self, speech):
        # print(f'显卡编号：{torch.cuda.current_device()}')
        # print(f'GPU数量：{torch.cuda.device_count()}')
        # print(f'GPU名称：{torch.cuda.get_device_name(0)}')
        # print(f'GPU是否可用：{torch.cuda.is_available()}')
        if speech.ndim == 2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
        input_values_all = self.wav2vec2_processor(speech, return_tensors="pt",
                                                   sampling_rate=16000).input_values  # [1, T]
        input_values_all = input_values_all.to(self.device)
        # For long audio sequence, due to the memory limitation, we cannot process them in one run
        # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
        # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
        # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
        # We have the equation to calculate out time step: T = floor((t-k)/s)
        # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
        # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
        res_lst = [None] * num_iter
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = self.hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
            res_lst[i] = hidden_states[0]
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
        # if input_values.shape[1] != 0:
        if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
            hidden_states = self.hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])
        ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
        # assert ret.shape[0] == expected_T
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
        else:
            ret = ret[:expected_T]
        return ret

    def make_even_first_dim(self, tensor):
        size = list(tensor.size())
        if size[0] % 2 == 1:
            size[0] -= 1
            return tensor[:size[0]]
        return tensor

    def get_aud_features(self, wav_path: str) -> np.ndarray:
        '''提取音频特征数据，返回特征数组ndarray'''
        # 读取音频文件，预处理到16k
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            try:
                speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
                speech = speech_16k
            except Exception as e:
                print(f'音频特征提取-转换16k异常：{e}')
                speech = speech.T
                speech_16k = librosa.resample(np.asarray(speech), orig_sr=sr, target_sr=16000)
                speech = speech_16k

            print("SR: {} to {}".format(sr, 16000))
        # 提取音频特征
        npy = self.get_aud_features_by_float32(speech)
        print(f'音频特征提取完成，维度：{npy.shape}')
        return npy

    def get_aud_features_by_float32(self, float32Vals: np.ndarray) -> np.ndarray:
        hubert_hidden = self.get_hubert_from_16k_speech(float32Vals)
        hubert_hidden = self.make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
        npy = hubert_hidden.detach().numpy()
        return npy


