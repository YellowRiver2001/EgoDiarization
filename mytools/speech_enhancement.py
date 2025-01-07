from pydub import AudioSegment
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np




#对segments按照1.5s进行切割，为了让增强去噪效果更好
def split_segments(segments, split_duration=1500):
    split_segments = []
    
    for start, end in segments:
        current_start = start
        while current_start < end:
            current_end = min(current_start + split_duration, end)
            split_segments.append([current_start, current_end])
            current_start = current_end
    
    return split_segments

#segments时间单位为ms
def clean_audio(input_audio_path, segments, output_dir,clean_dir,choice,ans):
    # ans = pipeline(
    # Tasks.acoustic_noise_suppression,

    # model='/home/rx/audio-visual/Light-ASD/mytools/models/speech_frcrn_ans_cirm_16k.bin')
    #model='damo/speech_frcrn_ans_cirm_16k')

    # 加载音频文件
    audio = AudioSegment.from_file(input_audio_path)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保输出目录存在
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    audio_segments = []
    # 分割并保存音频片段
    for i, (start_time, end_time) in enumerate(segments):
        # 转换时间范围为毫秒
        start_time_ms = start_time 
        end_time_ms = end_time 

        # 分割音频
        audio_segment = audio[start_time_ms:end_time_ms]

        # 保存音频片段
        subaudio_path = os.path.join(output_dir, f"segment_{i+1}.wav")
        audio_segment.export(subaudio_path, format="wav")
        
        print(f"Saved segment {i+1} from {start_time}ms to {end_time}ms as {subaudio_path}")
        sub_clean_path = clean_dir+f"clean_segment_{i+1}.wav"
        result = ans(subaudio_path,output_path=sub_clean_path)
        clean_segment = AudioSegment.from_file(sub_clean_path)
        audio_segments.append(clean_segment)

    # Concatenate all segments into one audio
    merged_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        merged_audio = merged_audio + segment
    if choice=="split":
        merged_audio.export(clean_dir+"clean_audio.wav", format="wav")
    if choice=="no_split":
        merged_audio.export(clean_dir+"clean_audio_nosplit.wav", format="wav")


def expand_time_segments(input_segments, total_range,choice):
    # 对输入的时间区间列表按起始时间排序
    input_segments.sort()

    full_segments = []
    current_start, current_end = total_range[0], total_range[1]

    # 处理输入的时间区间列表
    for start, end in input_segments:
        if start > current_end:
            # 如果当前时间段的起始时间大于当前总时间范围的结束时间，结束循环
            break
        
        if start > current_start:
            # 添加不在输入时间区间但在总时间范围内的时间段
            if choice=='split':
                no_sub = split_segments([[current_start, start]])
                full_segments = full_segments + no_sub 
            if choice=='no_split':
                full_segments.append([current_start, start])
            
        full_segments.append([start,end])

        # 更新当前总时间范围的起始时间
        current_start = max(end, current_start)

    # 添加剩余的时间范围
    if current_start < current_end:
        full_segments.append([current_start, current_end])
    
    return full_segments

def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Step 1: Sort the intervals by the starting time
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start < current_end:
            # There is an overlap, so we merge the intervals
            current_end = max(current_end, end)
        else:
            # No overlap, so add the current interval to the merged list
            merged.append([current_start, current_end])
            # Move to the next interval
            current_start, current_end = start, end
    
    # Don't forget to add the last interval
    merged.append([current_start, current_end])
    
    return merged

def Enhance(input_audio_path,segments,split_dir,clean_dir,total_range,choice,ans):
    segments = expand_time_segments(segments, total_range,choice)
    segments = merge_intervals(segments)
    clean_audio(input_audio_path, segments, split_dir,clean_dir,choice,ans)



# def main():
#     # 示例用法
#     input_audio_path = "audio.wav"
#     segments = np.loadtxt('sub_vad2_segments.txt',dtype=int, delimiter=' ').tolist()

#     total_range = [0,300000]

#     segments = expand_time_segments(segments, total_range)
#     print(segments)
