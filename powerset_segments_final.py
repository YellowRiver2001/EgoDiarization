from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.core.inference import Inference
import numpy as np

# MODEL_PATH="VAD/Ego4D.ckpt"   #在Ego4d上微调后的预训练模型
MODEL_PATH = "VAD/powerset_pretrained.ckpt"  #没有微调的预训练模型
model : SpeakerDiarization = Model.from_pretrained(MODEL_PATH)


def is_integer_after_decimal(num):
    # 计算小数部分
    decimal_part = num - int(num)
    # 判断小数部分是否为零
    return decimal_part == 0.0

def find_and_merge_speaking_intervals(speech_matrix):
    num_speakers = speech_matrix.shape[1]
    intervals = []

    for speaker in range(num_speakers):
        start = None
        for i, speaking in enumerate(speech_matrix[:, speaker]):
            if speaking == 1 and start is None:
                start = i  # 记录开始说话的时间
            elif speaking == 0 and start is not None:
                intervals.append((start/58.6, i/58.6))  # 记录结束说话的时间      模型每5s一次检测，5s里有293帧数据，58.6帧/s
                start = None
        if start is not None:
            intervals.append((start/58.6, len(speech_matrix)/58.6))  # 处理如果最后一个时间段在数组末尾结束

    # 按开始时间排序
    intervals.sort(key=lambda x: x[0])

    # 合并首尾相连的时间段
    merged_intervals = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start == current_end and is_integer_after_decimal(start):  # 仅当首尾相连时合并
            current_end = end
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end

    merged_intervals.append((current_start, current_end))

    return merged_intervals

def save_time_intervals(time_intervals, file_name):
    with open(file_name, 'w') as file:
        for interval in time_intervals:
            file.write(f"{int(interval[0]*1000)} {int(interval[1]*1000)}\n")

def read_filenames_from_txt(input_txt_file):
    # 初始化一个空列表存储文件名
    filenames = []
    
    # 打开并读取txt文件
    with open(input_txt_file, 'r') as f:
        # 逐行读取文件名，并去除换行符
        filenames = [line.strip() for line in f]
    
    return filenames

# fn = 'mytest4'

def Powerset_segments(fn):
    save_path = './demo/'+str(fn)+'/pyavi/'
    inference = Inference(model, step=5.0)
    WAV_FILE = save_path + 'audio.wav'
    # 进行推理
    result = inference(WAV_FILE)
    data =  result.data.reshape((result.data.shape[0]*result.data.shape[1],result.data.shape[2]))
    intervals = find_and_merge_speaking_intervals(data)
    print(intervals)
    # 将时间区间存储到文件中
    save_time_intervals(intervals, save_path+'pyannote_powerset_pretrained_segments.txt')


# Powerset_segments(fn)
