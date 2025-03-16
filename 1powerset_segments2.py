from pyannote.audio import Model
from pyannote.audio.core.inference import Inference
import numpy as np
import os

def is_integer_after_decimal(num):
    decimal_part = num - int(num)
    return decimal_part == 0.0


def find_and_merge_speaking_intervals(speech_matrix, n=58.6):
    num_speakers = speech_matrix.shape[1]
    intervals = []

    for speaker in range(num_speakers):
        start = None
        for i, speaking in enumerate(speech_matrix[:, speaker]):
            if speaking == 1 and start is None:
                start = i 
            elif speaking == 0 and start is not None:
                intervals.append((start/n, i/n))  
                start = None
        if start is not None:
            intervals.append((start/n, len(speech_matrix)/n)) 

    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start == current_end and is_integer_after_decimal(start): 
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
    filenames = []
    with open(input_txt_file, 'r') as f:
        filenames = [line.strip() for line in f]
    
    return filenames


def process_audio_files(model_path, filenames_path, dataset_path, exp_path, audio_path):
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
        print(file_no)
        fn = filenames[file_no].split('.')[0]
        save_path = os.path.join(exp_path, str(fn))
        os.makedirs(save_path, exist_ok=True)
        print(fn)
        inference = Inference(model, step=5.0)
        
        # 进行推理
        wav_path = os.path.join(audio_path, fn + ".wav")
        result = inference(wav_path)
        print(result.data.shape)
        data = result.data.reshape((result.data.shape[0]*result.data.shape[1], result.data.shape[2]))
        
        intervals = find_and_merge_speaking_intervals(data)
        # 将时间区间存储到文件中
        output_file = os.path.join(save_path, 'powerset_ego4d_segments.txt')
        save_time_intervals(intervals, output_file)


if __name__ == "__main__":
    MODEL_PATH = "weight/Ego4D.ckpt"   # 或者其他模型路径
    FILENAMES_PATH = 'dataset/v.txt'
    DATASET_PATH = f'dataset/val.txt'
    EXP_PATH = "./exp"
    AUDIOPATH = "egowav"
    
    process_audio_files(MODEL_PATH, FILENAMES_PATH, DATASET_PATH, EXP_PATH, AUDIOPATH)





