import os
import numpy as np
import sys
#from Ego4d_global_demo_test import *
import pickle
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import statistics
from mytools import voice2id3
from mytools import powerset_segments2
from mytools import copyasd
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from funasr import AutoModel
import argparse

# Specify the local model folder path
local_model_path = "/home/rx/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt"

# Check if the local path exists
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"Local model path {local_model_path} does not exist!")

# model = AutoModel(model = local_model_path, init_param = 1)

# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='/home/rx/.cache/modelscope/hub/damo/speech_frcrn_ans_cirm_16k/')



def evaluate_diarization(args, segments, id, fn, file_no, ders, miss_rates, fa_rates, speaker_error_rates):
    """
    Evaluate diarization performance for a single file
    
    Parameters:
    args: arguments object containing train_gt_path and collar
    segments, id: the expriement output

    val sets:
    fn: filename
    file_no: file number

    ders: list to store DER values
    miss_rates: list to store miss rates
    fa_rates: list to store false alarm rates
    speaker_error_rates: list to store speaker error rates
    
    Returns:
    dict: containing calculated metrics
    """

    # get experiment output
    hypothesis = Annotation() 
    for i in range(len(segments)):
        l = int(segments[i][0]*30/1000)
        r = int(segments[i][1]*30/1000)
        hypothesis[Segment(l, r)] = str(id[i])

    # Load ground truth
    gt_path = os.path.join(args.train_gt_path, f"{fn}_s.pickle")
    with open(gt_path, 'rb') as handle:
        speech_truth = pickle.load(handle)

    # Process ground truth
    trj_truth = {}
    for frame_num in speech_truth:
        for pid in speech_truth[frame_num]:
            if pid not in trj_truth:
                trj_truth[pid] = []
            trj_truth[pid].append(frame_num)

    reference = Annotation()
    for k in trj_truth:
        if k < 0:
            continue
        g = np.zeros(9000).astype('int')
        for n in range(9000):
            if n in trj_truth[k]:
                g[n] = 1
        g_intervals = get_intervals(g)
        for interval in g_intervals:
            reference[Segment(interval[0], interval[1])] = str(k)

    total_duration = sum((seg.end - seg.start) for seg in reference.get_timeline())
    collar_value = args.collar if (hasattr(args, 'collar') and args.collar is not None) else 0.0
    diarizationErrorRate = DiarizationErrorRate(collar=collar_value)
    detailed_der = diarizationErrorRate(reference, hypothesis, uem=Segment(0, 9000), detailed=True)
    
    fa_rate = detailed_der['false alarm'] / total_duration
    miss_rate = detailed_der['missed detection'] / total_duration
    speaker_error_rate = detailed_der['confusion'] / total_duration
    der = fa_rate + miss_rate + speaker_error_rate

    print(f"\nFile {file_no} - {fn}")
    print(f"DER: {der:.4f}")
    print(f"False Alarm Rate: {fa_rate:.2%}")
    print(f"Miss Rate: {miss_rate:.2%}")
    print(f"Speaker Error Rate: {speaker_error_rate:.2%}")
    print("-----------------------------------")

    ders.append(der)
    miss_rates.append(miss_rate)
    fa_rates.append(fa_rate)
    speaker_error_rates.append(speaker_error_rate)

    return ders, miss_rates, fa_rates, speaker_error_rates

def get_intervals(t):
    intervals = []
    a = 0
    b = 0
    while (a < t.shape[0] and b < t.shape[0]):
        if (t[a] == 0 and t[b] == 0):
            a = a + 1
            b = b + 1
            continue
        if (t[a] == 1 and t[b] == 1):
            b = b + 1
            continue
        if (t[a] == 1 and t[b] == 0): 
            intervals.append([a, b-1])  
            a = b
            continue
    if (a != t.shape[0]):    
        intervals.append([a, t.shape[0]-1])
    return intervals  


# Calculate error scores
# Experiment 1: Voiceprint averaging only          √
# Experiment 2: Clustering number assisted + voiceprint averaging  √
# Experiment 3: Denoising + average voiceprint         √
# Experiment 4: Direct clustering              √
# Experiment 5: Clustering number assisted          √
# Experiment 6: Denoising + voiceprint averaging + clustering number assisted √
# 11 means all the experiments

# num_tests  = [11,7,6,5,4,3,2,1]  

def main():
    parser = argparse.ArgumentParser(description='Process diarization and compute DER metrics.')
    #dataset
    parser.add_argument('--train_name_path', type=str, default='dataset/v.txt',                 help='Path to train set filenames')
    parser.add_argument('--train_gt_path',   type=str, default='dataset/headbox_wearer_speaker',help='Path to train set ground truth')
    parser.add_argument('--val_index_path',  type=str, default='dataset/val.txt',               help='Path to validation set indices')
    
    # exp set
    parser.add_argument('--exp', type=str, default='exp/test1',help='Path to output directory')
    parser.add_argument('--modal', type=str, default='weight/Ego4D.ckpt',help='Path to output directory')
    # use the exp data and dont use the pyannote output

    parser.add_argument('--copy', type=str, default='dataset/asd',help='Path to output directory')
    args = parser.parse_args()
   
    with open(args.train_name_path) as f:
        content = f.readlines()
    filenames = [x.strip() for x in content]
    video_nums = np.loadtxt(args.val_index_path)
    video_nums = video_nums.astype('int').tolist()

    # a is hyperparameter and 0.9 is the best output
    # for a in [0,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    ders = []
    miss_rates = []
    fa_rates = []
    speaker_error_rates = []
    for a in [0.9]:
        # num_test means Ablation study
        num_test = 11
        for file_no in video_nums:
            fn = filenames[file_no].split('.')[0]
            save_path = args.exp
            
            # #get audio segments
            # powerset_segments2.getseg(args.modal, args.train_name_path, args.val_index_path, save_path, wav_path)

            #(copy)get speaker num matrix
            # save_path = args.exp
            # wav_path = os.path.join(args.audio_path, f"{fn}.wav")
            # sourcebase = args.copy
            # targetbase = os.path.join(args.exp, "mid")
            # copyasd.copy_matched_files(sourcebase, targetbase)

            #get output of pipeline
            id,segments = voice2id3.voice2id_pyannote2_final_fine_tuned_hyper(fn, save_path, num_test, a)
            
            #id,segments = voice2id3.voice2id_pyannote2_final_fine_tuned_hyper(wav_path, fn, save_path, num_test, a, ans,model)
            evaluate_diarization(args, segments, id, fn, file_no, ders, miss_rates, fa_rates, speaker_error_rates)
    
        mean_der = statistics.mean(ders)
        mean_fa = statistics.mean(fa_rates)
        mean_miss = statistics.mean(miss_rates)
        mean_ser = statistics.mean(speaker_error_rates)

        print(f"Number of files processed: {len(ders)}")
        print(f"Mean DER: {mean_der:.4f}")
        print(f"Mean FA: {mean_fa:.4f}")
        print(f"Mean MISS: {mean_miss:.4f}")
        print(f"Mean SER: {mean_ser:.4f}")
        der_path = os.path.join(args.exp, "der.txt")
        output_string = (
            f"Number of files processed: {len(ders)}\n"
            f"Mean DER: {mean_der:.4f}\n"
            f"Mean FA: {mean_fa:.4f}\n"
            f"Mean MISS: {mean_miss:.4f}\n"
            f"Mean SER: {mean_ser:.4f}\n"
        )
        with open(der_path, "w") as f:
            f.write(output_string)

if __name__ == "__main__":
    main()


