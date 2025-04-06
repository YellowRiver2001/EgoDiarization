#微调过后的消融实验



#该文件完成从视频输入，到语音段和语音段对于人物id的输出,并计算DER
import os
import numpy as np
import sys
#from Ego4d_global_demo_test import *
import pickle
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import statistics
from  mytools  import voice2id3
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from funasr import AutoModel



with open('../v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

#data_set = sys.argv[1] # 'test' or 'val'
data_set = 'val'

video_nums = np.loadtxt('../'+data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

def get_intervals(t):
    intervals = []
    a = 0
    b = 0
    while (a < t.shape[0] and b < t.shape[0]):
        if (t[a] == 0 and t[b] == 0):  #开始都为0，则左右指针向前移动1
            a = a + 1
            b = b + 1
            continue
    
        if (t[a] == 1 and t[b] == 1):#做指针为1，右指针也为1，则右指针向前移动1个
            b = b + 1
            continue
    
        if (t[a] == 1 and t[b] == 0):   #一段说话段找寻结束
            intervals.append([a, b-1])  
            a = b
            continue
    
    if (a != t.shape[0]):    #这里说明在末尾帧，也都是1，所有将这一段也添加进去
        intervals.append([a, t.shape[0]-1])

    return intervals  


# #加载模型
# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='damo/speech_frcrn_ans_cirm_16k')

# model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")


#实验一，啥也不加   
#实验二，只增强
#实验三，只聚类辅助
#实验四，只视听向量
#实验五，增强+聚类辅助   
#实验六，增强+视听向量
#实验七，聚类数辅助+视听向量
#实验八，啥都加



# num_tests  = [6]  #

# num_tests  = [11,6,7,5,4,3,2,1]  #
num_tests  = [3,2,1]  #


# for n_frame in [40,100,200,77,90,46,60,80,150]:
# for a in range(10,11,1):
#     print(a/10.0)
for num_test in num_tests:
    a=0.9
    #计算错误分数
    ders = []
    all_ders = []
    miss_rates = []
    fa_rates = []
    speaker_error_rates = []
    for file_no in video_nums:
        fn = filenames[file_no].split('.')[0]
        save_path = './demo/'+str(fn)+'/pyavi/'
        print(fn)

        id,segments = voice2id3.voice2id_pyannote2_final_fine_tuned_fusion(save_path,num_test,a)

        gt_dir = '../utils/ground_truth/headbox_wearer_speaker'
        with open(gt_dir + '/'  + str(fn) + '_s.pickle', 'rb') as handle:
            speech_truth = pickle.load(handle)

        trj_truth = {}
        for frame_num in speech_truth:
            for pid in speech_truth[frame_num]:
                if pid not in trj_truth:
                    trj_truth[pid] = []
                trj_truth[pid].append(frame_num)    #这里应该是将ground_true以每个人的pid为下标，存储其说话的帧序列


        # ground_truth_speaker_num.append(len(trj_truth))

        hypothesis = Annotation() # 创建一个 Annotation 实例，用于存储假设的语音活动区间及其对应的说话人标签
        #segments_path = save_path+'final_id_no2.txt'
        #segments_path = save_path + 'final_clean6_segments.txt'
        # segments_path = save_path + 'segments_pyannote3.1.txt'
        # segments = np.loadtxt(segments_path)
        for i in range(len(segments)):
            l = int(segments[i][0]*30/1000)
            r = int(segments[i][1]*30/1000)
            #print(l,r)
            hypothesis[Segment(l, r)] = str(id[i])
            #hypothesis[Segment(l, r)] = 0
        reference = Annotation()
        ground_truth = []

        for k in trj_truth:
            if k < 0:
                continue
            g = np.zeros(9000).astype('int')
            for n in range(9000):
                if (n in trj_truth[k]):
                    g[n] = 1   #得到当前id为k的人的说话帧序列，说了为1，没说为0

            g_intervals = get_intervals(g)
            for n in range(len(g_intervals)):
                #print(g_intervals[n][0],g_intervals[n][1])
                reference[Segment(g_intervals[n][0], g_intervals[n][1])] = str(k)
                #reference[Segment(g_intervals[n][0], g_intervals[n][1])] = 0
                #ground_truth.append([int(g_intervals[n][0]/30*1000), int(g_intervals[n][1]/30*1000)])

        #print(ground_truth)

        # 计算参考数据的总时长
        total_duration = sum((segment.end - segment.start) for segment in reference.get_timeline())
        print("total_duration:",total_duration)

        diarizationErrorRate = DiarizationErrorRate()
        detailed_der = diarizationErrorRate(reference, hypothesis, uem=Segment(0, 9000), detailed=True)   #
        # 输出详细结果
        fa_rate = detailed_der['false alarm'] /total_duration
        miss_rate = detailed_der['missed detection'] /total_duration
        speaker_error_rate = detailed_der['confusion'] /total_duration
        der =  fa_rate + miss_rate + speaker_error_rate

        print("DER:",der)
        print(f"False Alarm Rate (FA) Proportion: {fa_rate:.2%}")
        print(f"Miss Rate (MISS) Proportion: {miss_rate:.2%}")
        print(f"Speaker Error Rate (SPK) Proportion: {speaker_error_rate:.2%}")
        np.savetxt(save_path+'DER_Powerset_Ego4d_segments_final_study'+str(num_test)+'.txt',[der,fa_rate,miss_rate,speaker_error_rate],fmt='%.4f')
        # der,fa_rate,miss_rate,speaker_error_rate = np.loadtxt(save_path+"DER_Powerset_pretrained_segments_final_study12.txt")

        all_ders.append([der,fa_rate,miss_rate,speaker_error_rate])
        ders.append(der)
        miss_rates.append(miss_rate)
        fa_rates.append(fa_rate)
        speaker_error_rates.append(speaker_error_rate)
        print(file_no, der)


    print('mean DER = ', statistics.mean(ders))  
    print('mean miss rate = ',statistics.mean(miss_rates)  )
    # #np.savetxt('./demo/MEAN_DER.txt',[statistics.mean(ders)],fmt='%.4f')
    np.savetxt('./demo/1MEAN_DER_Powerset_Ego4d_segments_final_'+str(a)+'_study'+str(num_test)+'.txt',[statistics.mean(ders),statistics.mean(fa_rates),statistics.mean(miss_rates),statistics.mean(speaker_error_rates)],fmt='%.4f')
    # #np.savetxt('./demo/EGO4d_VAL_DERS.txt',ders,fmt='%.4f')
    np.savetxt('./demo/1EGO4d_VAL_DERS_Powerset_Ego4d_segments_final_'+str(a)+'_study'+str(num_test)+'.txt',all_ders,fmt='%.4f')








