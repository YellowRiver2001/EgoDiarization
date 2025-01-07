#该文件完成从视频输入，到输出说话人日志结果
import os
import numpy as np
#from Ego4d_global_demo_test import *
from  mytools  import voice2id_final
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
from powerset_segments_final import Powerset_segments 


# #加载模型
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')

model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

#计算错误分数



#实验一，啥也不加   
#实验二，只增强
#实验三，只聚类辅助
#实验四，只视听向量
#实验五，增强+聚类辅助   
#实验六，增强+视听向量
#实验七，聚类数辅助+视听向量
#实验八，啥都加

print("start-----------------------------------")
a=0.7
num_test = 8
fn = 'meeting'    #默认存储在./demo文件夹里的
save_path = '/home/rx/audio-visual/Light-ASD/demo/'+str(fn)+'/pyavi/'   #注意这里要用绝对路径
print(fn)
os.system('python Ego4d_global_demo_final.py --videoName '+str(fn))

Powerset_segments(fn)
#main()  #计算light-ASD的初步说话人检测

id,segments = voice2id_final.voice2id_pyannote2_final(save_path,num_test,a,ans,model)

print(id)
print(segments)