# -*- coding: utf-8 -*-

#该文件完成从视频输入，到输出说话人日志结果
import os
import argparse
import numpy as np
from  mytools  import voice2id_final
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
from powerset_segments_final import Powerset_segments 


# 加载语音增强模型
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')

#加载语音活动检测模型
model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")


parser = argparse.ArgumentParser(description = "demo")
parser.add_argument('--videoName', type=str, default="",   help='Demo video name')

args = parser.parse_args()

def run_diarization(fn):

    print("start-----------------------------------")
    a=0.7
    num_test = 8
    save_path = '/home/yourpath/demo/'+str(fn)+'/pyavi/'   #注意这里要用绝对路径
    print(fn)
    os.system('python Light-ASD/Ego4d_global_demo_final.py --videoName '+str(fn) +' --videoFolder /home/rx/yourpath/demo  --videos_Path /home/yourpath/demo')
    Powerset_segments(fn)
    id,segments = voice2id_final.voice2id_pyannote2_final(save_path,num_test,a,ans,model)

    print(id)
    print(segments)

    return id,segments


if __name__ == "__main__":
    fn = args.videoName    #默认存储在./demo文件夹里的视频
    run_diarization(fn)
    