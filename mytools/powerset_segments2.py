from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.core.inference import Inference
import numpy as np
import os


def is_integer_after_decimal(num):
    # 计算小数部分
    decimal_part = num - int(num)
    # 判断小数部分是否为零
    return decimal_part == 0.0


def find_and_merge_speaking_intervals(speech_matrix, n=58.6):
    num_speakers = speech_matrix.shape[1]
    intervals = []

    for speaker in range(num_speakers):
        start = None
        for i, speaking in enumerate(speech_matrix[:, speaker]):
            if speaking == 1 and start is None:
                start = i  # 记录开始说话的时间
            elif speaking == 0 and start is not None:
                intervals.append((start/n, i/n))  # 记录结束说话的时间
                start = None
        if start is not None:
            intervals.append((start/n, len(speech_matrix)/n))  # 处理如果最后一个时间段在数组末尾结束

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


def getseg(model_path, filenames_path, dataset_path, exp_path, audio_path):
    """
    处理音频文件并保存说话时间段
    
    参数:
    model_path: 模型路径
    filenames_path: 包含文件名的txt文件路径
    dataset_path: 数据集路径(val.txt或test.txt)
    exp_path: 实验结果保存的根路径
    """
    model = Model.from_pretrained(model_path)
    
    with open(filenames_path) as f:
        content = f.readlines()
    filenames = [x.strip() for x in content]
    
    video_nums = np.loadtxt(dataset_path)
    video_nums = video_nums.astype('int').tolist()
    
    for file_no in video_nums:
        
        fn = filenames[file_no].split('.')[0]
        print(fn)
        save_path = os.path.join(exp_path,"mid")
        save_path = os.path.join(save_path, str(fn))
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, 'powerset_ego4d_segments.txt')
        if os.path.exists(output_file):
            print("Output file already exists. Exiting loop.")
            continue
        inference = Inference(model, step=5.0)
        
        # 进行推理
        wav_path = os.path.join(audio_path, fn + ".wav")
        result = inference(wav_path)

        data = result.data.reshape((result.data.shape[0]*result.data.shape[1], result.data.shape[2]))
        intervals = find_and_merge_speaking_intervals(data)
        save_time_intervals(intervals, output_file)


if __name__ == "__main__":
    MODEL_PATH = "weight/Ego4D.ckpt"   # 或者其他模型路径
    FILENAMES_PATH = 'dataset/v.txt'
    DATASET_PATH = f'dataset/val.txt'
    EXP_PATH = "./exp"
    AUDIOPATH = "egowav"
    
    getseg(MODEL_PATH, FILENAMES_PATH, DATASET_PATH, EXP_PATH, AUDIOPATH)





