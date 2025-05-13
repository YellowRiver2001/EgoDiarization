import numpy as np
import torch
import os
from camplus import CAMPPlus
import torchaudio.compliance.kaldi as Kaldi

#from pathlib import Path
from collections import OrderedDict
import torchaudio
from spectral_cluster import cluster,cluster_probability
import soundfile
from moviepy.editor import *
#from PIL import Image, ImageDraw, ImageFont
import numpy as np


#from PIL import Image, ImageDraw, ImageFont

# import gc 


def compute_fbank(wav, num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  sample_frequency=16000,
                  subseg_cmn=True):
    wav = wav.unsqueeze(0) * (1 << 15)
    feat = Kaldi.fbank(
        wav, num_mel_bins=num_mel_bins, frame_length=frame_length, frame_shift=frame_shift, dither=dither, sample_frequency=sample_frequency, window_type='hamming', use_energy=False)
    if not subseg_cmn:
        feat = feat - torch.mean(feat, dim=0)
    return feat.cpu().numpy()

def read_segments(segments):
    utt_to_segments = OrderedDict()
    segments = segments.split('\n')
    for line in segments:
        if line == '':
            continue
        seg, utt, begin, end = line.strip().split()
        begin, end = float(begin), float(end)
        if utt not in utt_to_segments:
            utt_to_segments[utt] = [(seg, begin, end)]
        else:
            utt_to_segments[utt].append((seg, begin, end))
    return utt_to_segments

def get_speech_segments(utt_to_segments, wav_path):
    speech_segments_id = []
    speech_segments = []
    audio_name = wav_path.split('/')[-1][:-4]
    segments = utt_to_segments[audio_name]
    signal, sr = torchaudio.load(wav_path)
    print(signal.shape)
    print(sr)
    signal = signal.squeeze()
    print(signal.shape)
    for seg, begin, end in segments:
        print(seg, begin, end)
        if end-begin > 0.4:
            speech_segments_id.append(seg)
            print(seg)
            speech_segments.append(signal[int(begin * sr): int(end * sr)])
            print(signal[int(begin * sr): int(end * sr)])
        else:
            speech_segments_id.append(seg)
            speech_segment = signal[int(begin * sr): int(end * sr)]
            times =int( 0.4/(end-begin) + 1 )
            new_speech_segment = torch.cat([speech_segment] * times, dim=0)
            speech_segments.append(new_speech_segment)
    print(speech_segments)
    print(len(speech_segments))
    #print(speech_segments[0].shape)
    return speech_segments_id, speech_segments

def get_asvmodel(checkpoint_path="./camplus.ckpt"):
    model = CAMPPlus(feat_dim=80,
                 embed_dim=192,
                 pooling_func='TSTP',
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu')
  
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def get_embedding(fbank_feature):
    '''
    根据活性检测的结果,对每个音频段计算embedding
    '''
    # model 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    embedding_model = get_asvmodel()
    embedding_model = embedding_model.to(device)
    fbank_feature = fbank_feature.to(device)
    embedding = embedding_model(fbank_feature)
    return embedding

# def get_vad_result(audio_path='./audio.wav'):
#     vad = fsmn_vad.FSMNVad()
#     #vad = fsmn_vad.FSMNVadOnline()
#     segments = vad.segments_offline(audio_path)
#     vad_result = ""
#     utt = audio_path.split('\\')[-1][:-4]
#     for item in segments:
#         vad_result += "{}-{:08d}-{:08d} {} {:.3f} {:.3f}\n".format(utt, int(item[0]), int(item[1]), utt, int(item[0]) / 1000, int(item[1]) / 1000)
#     return vad_result


#基于whisperX方法的语音活动检测
def get_vad_result(segments,audio_path):
    vad_result = ""
    utt = audio_path.split('/')[-1][:-4]
    for item in segments:
        vad_result += "{}-{:08d}-{:08d} {} {:.3f} {:.3f}\n".format(utt, int(item[0]), int(item[1]), utt, int(item[0])/1000 , int(item[1])/1000 )
    
    return vad_result



def subsegment(fbank, seg_id, window_fs, period_fs, frame_shift):
    subsegs = []
    subseg_fbanks = []
    seg_begin, seg_end = seg_id.split('-')[-2:]
    seg_length = (int(seg_end) - int(seg_begin)) // frame_shift
    num_frames, feat_dim = fbank.shape
    if seg_length <= window_fs:
        subseg = seg_id + "-{:08d}-{:08d}".format(0, seg_length)
        subseg_fbank = np.resize(fbank, (window_fs, feat_dim))


        subsegs.append(subseg)
        subseg_fbanks.append(subseg_fbank)
        
    else:
        max_subseg_begin = seg_length - window_fs + period_fs
        for subseg_begin in range(0, max_subseg_begin, period_fs):
            subseg_end = min(subseg_begin + window_fs, seg_length)
            subseg = seg_id + "-{:08d}-{:08d}".format(subseg_begin, subseg_end)
            subseg_fbank = np.resize(fbank[subseg_begin:subseg_end],
                                     (window_fs, feat_dim))

            subsegs.append(subseg)
            subseg_fbanks.append(subseg_fbank)
    return subsegs, subseg_fbanks

def extract_embedding(fbanks, batch_size, subseg_cmn):
    fbanks_array = np.stack(fbanks)
    if subseg_cmn:
        fbanks_array = fbanks_array - np.mean(fbanks_array, axis=1, keepdims=True)
    embeddings = []
    for i in range(0, fbanks_array.shape[0], batch_size):
        batch_feats = fbanks_array[i:i + batch_size]
        batch_embs = get_embedding(torch.from_numpy(batch_feats))
        embeddings.append(batch_embs.detach().cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return embeddings

#声纹检测并聚类划分到人物id
def VocalPrint_Detect(audio_path,segments):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    #print(vad_result)
    utt_to_segments = read_segments(vad_result)
    
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    #print(speech_segments)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)

     
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    

    embeddings = extract_embedding(subseg_fbanks, batch_size=1, subseg_cmn=True)

    subseg_i = {}
    emb_dict = OrderedDict()

    for i, subseg_id in enumerate(subsegs):
        subseg_i[subseg_id] = embeddings[i]
    
    for sub_seg_id, emb in subseg_i.items():
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {}
            emb_dict[utt]['sub_seg'] = []
            emb_dict[utt]['embs'] = []
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)
    
    subsegs_list = []
    embeddings_list = []
    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))

    #print(embeddings_list[0])
    
    labels_list = cluster(embeddings_list[0],num_spks=4)
    
    
    
    with open("output.txt", 'w') as f:
        for (subsegs, labels) in zip(subsegs_list, [labels_list]):
            [print(subseg, label, file=f) for (subseg, label) in zip(subsegs, labels)]

    # make rttm
    utt_subseg_labels = OrderedDict()
    for line in open("output.txt", 'r'):
        subseg, label = line.strip().split()
        utt, begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
        begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
        end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0
        if utt not in utt_subseg_labels:
            utt_subseg_labels[utt] = [(begin, end, label)]
        else:
            utt_subseg_labels[utt].append((begin, end, label))
    
    merged_segment_to_labels = []
    for utt, subseg_to_labels in utt_subseg_labels.items():
        if len(subseg_to_labels) == 0:
            continue
        
        (begin, end, label) = subseg_to_labels[0]
        e = end
        for (b, e, la) in subseg_to_labels[1:]:
            if b <=  end and la == label:
                end = e
            elif b > end:
                merged_segment_to_labels.append((utt, begin, end, label))
                begin, end, label = b, e, la
            elif b <= end and la != label:
                pivot = (b + end) / 2.0
                merged_segment_to_labels.append((utt, begin, pivot, label))
                begin, end, label = pivot, e, la
            else:
                raise ValueError
        merged_segment_to_labels.append((utt, begin, e, label))
    rttm_line_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n"
    for (utt, begin, end, label) in merged_segment_to_labels:
        #print(rttm_line_spec.format(utt, label, begin, end))
        print(rttm_line_spec.format(utt, 1, begin, end, label))
    return merged_segment_to_labels
 

#获得每一段语音的声纹向量
def VocalPrint_embeddings(audio_path,segments):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)

    utt_to_segments = read_segments(vad_result)

    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[i] = compute_fbank(speech_seg, subseg_cmn=True)
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for i, fbank in fbank_feat.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=speech_segments_id[i], window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    embeddings = extract_embedding(subseg_fbanks, batch_size=1, subseg_cmn=True)
    
    return embeddings


#获得每段语音的声纹向量（不用滑动窗口，但目前代码还有问题，结果不如滑动窗口好）
def VocalPrint_embeddings_nowindows(audio_path,segments):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)

    utt_to_segments = read_segments(vad_result)

    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)

    embeddings=[]
    for i, speech_seg in enumerate(speech_segments):
        fbanks_array=compute_fbank(speech_seg, subseg_cmn=True)
        fbank_feature=fbanks_array.reshape(1,np.shape(fbanks_array)[0],np.shape(fbanks_array)[1])
        #fbanks_array = np.stack(fbanks_array)
        embeddings.append(get_embedding(torch.from_numpy(fbank_feature)).detach().cpu().numpy())
    embeddings=np.vstack(embeddings)

    labels_list = cluster(embeddings,num_spks=None)



    return embeddings


#获得初步划分的每个人的语音的声纹向量
def VocalPrint_embeddings_id(audio_path,segments,voice2id,person_num):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    utt_to_segments = read_segments(vad_result)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        #print(seg_id,np.shape(fbank))
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    #subseg_fbanks为窗口扩充后的每个语音段的fbanks编码（n,3000,80) n为语音段数量
    n=len(voice2id)
    persons_voiceprint={}

    #将划分为同一个人的语音段fbanks拼接
    for i in range(0,n):
        if(voice2id[i]!=-1):
            if voice2id[i] in  persons_voiceprint:
                persons_voiceprint[voice2id[i]]=np.concatenate((persons_voiceprint[voice2id[i]],subseg_fbanks[i]),axis=0)
            else: 
                persons_voiceprint[voice2id[i]]=subseg_fbanks[i]

    #embeddings = np.zeros((person_num, 192))
    embeddings = np.random.uniform(-2, 2, (person_num, 192))
    for key,value in persons_voiceprint.items():

        embedding = get_embedding(torch.from_numpy(value.reshape(1,np.shape(value)[0],np.shape(value)[1]))).detach().cpu().numpy()
        embedding = np.squeeze(embedding)
        embeddings[key] =embedding

    return embeddings
    

#获得初步划分的每个人的语音的声纹向量
def VocalPrint_embeddings_id_wear(audio_path,segments,voice2id,person_num):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    utt_to_segments = read_segments(vad_result)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        #print(seg_id,np.shape(fbank))
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    #subseg_fbanks为窗口扩充后的每个语音段的fbanks编码（n,3000,80) n为语音段数量
    n=len(voice2id)
    persons_voiceprint={}

    #将划分为同一个人的语音段fbanks拼接
    for i in range(0,n):
        if(voice2id[i]!=-1):
            if voice2id[i] in  persons_voiceprint:
                persons_voiceprint[voice2id[i]]=np.concatenate((persons_voiceprint[voice2id[i]],subseg_fbanks[i]),axis=0)
            else: 
                persons_voiceprint[voice2id[i]]=subseg_fbanks[i]

    #embeddings = np.zeros((person_num, 192))
    embeddings = np.random.uniform(-2, 2, (person_num, 192))
    for key,value in persons_voiceprint.items():
        frank_feature = torch.from_numpy(value.reshape(1,np.shape(value)[0],np.shape(value)[1]))
        # 计算张量占用的内存大小（以字节为单位）
        memory_size = frank_feature.element_size() * frank_feature.numel()
        print(f'Tensor memory size: {memory_size / (1024 ** 2):.2f} MB')
        #continue
        embedding = get_embedding(torch.from_numpy(value.reshape(1,np.shape(value)[0],np.shape(value)[1]))).detach().cpu().numpy()
        torch.cuda.empty_cache()
        embedding = np.squeeze(embedding)
        embeddings[key] =embedding

    return embeddings



#获得初步划分的每个人的语音的声纹向量,对同一个人的不同段音频分别求声纹特征再取平均值
def VocalPrint_embeddings_id_wear_mean(audio_path,segments,voice2id,person_num):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    utt_to_segments = read_segments(vad_result)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        #print(seg_id,np.shape(fbank))
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    #subseg_fbanks为窗口扩充后的每个语音段的fbanks编码（n,3000,80) n为语音段数量
    n=len(voice2id)
    persons_voiceprint={}
    segments_number = {}

    #将划分为同一个人的语音段的声纹向量相加，并记录每个人被划分的音频数量
    for i in range(0,n):
        value = subseg_fbanks[i]
        frank_feature = torch.from_numpy(value.reshape(1,np.shape(value)[0],np.shape(value)[1]))        
        embedding = get_embedding(frank_feature).detach().cpu().numpy()
        if(voice2id[i]!=-1):
            if voice2id[i] in  persons_voiceprint:
                persons_voiceprint[voice2id[i]]=persons_voiceprint[voice2id[i]]+embedding
                segments_number[voice2id[i]] = segments_number[voice2id[i]]+1
            else: 
                persons_voiceprint[voice2id[i]]=embedding
                segments_number[voice2id[i]] = 1

    embeddings = np.random.uniform(-2, 2, (person_num, 192))
    for key,value in persons_voiceprint.items():
        embedding = value / segments_number[key]
        embedding = np.squeeze(embedding)
        embeddings[key] =embedding

    return embeddings

#根据视觉初步划分的每个人的id，以及子段聚类再次划分后的语音段，再次进行聚类划分
def VocalPrint_Detect_spk_num_visual(audio_path,segments,voice2id,person_num):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    utt_to_segments = read_segments(vad_result)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        #print(seg_id,np.shape(fbank))
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    #subseg_fbanks为窗口扩充后的每个语音段的fbanks编码（n,3000,80) n为语音段数量
    n=len(voice2id)
    persons_voiceprint={}   #记录同一个人的声纹向量
    segments_number = {}    #记录同一个人的声纹向量语音段数
    set_id ={}              #记录同一个人的语音段的id下标

    #先初步得到每个语音段的声纹向量
    embeddings = extract_embedding(subseg_fbanks, batch_size=1, subseg_cmn=True)

    #将划分为同一个人的语音段的声纹向量相加并求平均值
    for i in range(0,n):
        if(voice2id[i]!=-1):   
            if voice2id[i] in  persons_voiceprint:
                persons_voiceprint[voice2id[i]]=persons_voiceprint[voice2id[i]]+embeddings[i]
                segments_number[voice2id[i]] = segments_number[voice2id[i]]+1
                set_id[voice2id[i]].append(i)
            else: 
                persons_voiceprint[voice2id[i]]=embeddings[i]
                segments_number[voice2id[i]] = 1
                set_id[voice2id[i]]=[i]
    for key,value in persons_voiceprint.items():
        embedding = value / segments_number[key]
        embedding = np.squeeze(embedding)
        for index in set_id[key]:
            embeddings[index] = embedding

    subseg_i = {}
    emb_dict = OrderedDict()

    for i, subseg_id in enumerate(subsegs):
        subseg_i[subseg_id] = embeddings[i]
    
    for sub_seg_id, emb in subseg_i.items():
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {}
            emb_dict[utt]['sub_seg'] = []
            emb_dict[utt]['embs'] = []
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)
    
    subsegs_list = []
    embeddings_list = []
    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))

    #print(embeddings_list[0])
    
    labels_list = cluster(embeddings_list[0],num_spks=person_num)


    return labels_list,embeddings

def VocalPrint_embeddings_mean(audio_path,embeddings,voice2id,person_num):
    n=len(voice2id)
    persons_voiceprint={}
    segments_number = {}

    #将划分为同一个人的语音段的声纹向量相加，并记录每个人被划分的音频数量
    for i in range(0,n):
        if(voice2id[i]!=-1):
            if voice2id[i] in  persons_voiceprint:
                persons_voiceprint[voice2id[i]]=persons_voiceprint[voice2id[i]]+embeddings[i]
                segments_number[voice2id[i]] = segments_number[voice2id[i]]+1
            else: 
                persons_voiceprint[voice2id[i]]=embeddings[i]
                segments_number[voice2id[i]] = 1

    id_embeddings = np.random.uniform(-2, 2, (person_num, 192))
    for key,value in persons_voiceprint.items():
        embedding = value / segments_number[key]
        embedding = np.squeeze(embedding)
        id_embeddings[key] =embedding
    
    return id_embeddings

#声纹检测并根据全局追踪的人数并聚类划分到人物id
def VocalPrint_Detect_spk_num(audio_path,segments,n):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    #print(vad_result)
    utt_to_segments = read_segments(vad_result)
    
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)

    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        # print(len(speech_seg))
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)

     
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    
    # print(len(subseg_fbanks))
    embeddings = extract_embedding(subseg_fbanks, batch_size=1, subseg_cmn=True)

    # subseg_i = {}
    # emb_dict = OrderedDict()

    # for i, subseg_id in enumerate(subsegs):
    #     subseg_i[subseg_id] = embeddings[i]
    
    # for sub_seg_id, emb in subseg_i.items():
    #     utt = sub_seg_id.split('-')[0]
    #     if utt not in emb_dict:
    #         emb_dict[utt] = {}
    #         emb_dict[utt]['sub_seg'] = []
    #         emb_dict[utt]['embs'] = []
    #     emb_dict[utt]['sub_seg'].append(sub_seg_id)
    #     emb_dict[utt]['embs'].append(emb)
    
    # subsegs_list = []
    # embeddings_list = []
    # for utt, utt_emb_dict in emb_dict.items():
    #     subsegs_list.append(utt_emb_dict['sub_seg'])
    #     embeddings_list.append(np.stack(utt_emb_dict['embs']))

    # print(embeddings_list[0])
    # print(embeddings_list[0].shape)
    # print(embeddings)
    # print(embeddings.shape)
    labels_list = cluster(embeddings,num_spks=n)
    # labels_list = cluster(embeddings_list[0],num_spks=n)
    
    return labels_list,embeddings
    
    # with open("output.txt", 'w') as f:
    #     for (subsegs, labels) in zip(subsegs_list, [labels_list]):
    #         [print(subseg, label, file=f) for (subseg, label) in zip(subsegs, labels)]

    # # make rttm
    # utt_subseg_labels = OrderedDict()
    # for line in open("output.txt", 'r'):
    #     subseg, label = line.strip().split()
    #     utt, begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
    #     begin = (int(begin_ms) + int(begin_frames) * frame_shift) / 1000.0
    #     end = (int(begin_ms) + int(end_frames) * frame_shift) / 1000.0
    #     if utt not in utt_subseg_labels:
    #         utt_subseg_labels[utt] = [(begin, end, label)]
    #     else:
    #         utt_subseg_labels[utt].append((begin, end, label))
    
    # merged_segment_to_labels = []
    # for utt, subseg_to_labels in utt_subseg_labels.items():
    #     if len(subseg_to_labels) == 0:
    #         continue
        
    #     (begin, end, label) = subseg_to_labels[0]
    #     e = end
    #     for (b, e, la) in subseg_to_labels[1:]:
    #         if b <=  end and la == label:
    #             end = e
    #         elif b > end:
    #             merged_segment_to_labels.append((utt, begin, end, label))
    #             begin, end, label = b, e, la
    #         elif b <= end and la != label:
    #             pivot = (b + end) / 2.0
    #             merged_segment_to_labels.append((utt, begin, pivot, label))
    #             begin, end, label = pivot, e, la
    #         else:
    #             raise ValueError
    #     merged_segment_to_labels.append((utt, begin, e, label))
    # rttm_line_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n"
    # for (utt, begin, end, label) in merged_segment_to_labels:
    #     #print(rttm_line_spec.format(utt, label, begin, end))
    #     print(rttm_line_spec.format(utt, 1, begin, end, label))
    # return merged_segment_to_labels


#声纹检测并根据全局追踪的人数并聚类划分到人物id
def VocalPrint_Detect_spk_num_probability(audio_path,segments,spks_num_probabilitys,rate):
    # first step
    # Voice Activity Detection
    vad_result = get_vad_result(segments,audio_path)
    #print(vad_result)
    utt_to_segments = read_segments(vad_result)
    
    speech_segments_id, speech_segments = get_speech_segments(utt_to_segments=utt_to_segments, wav_path=audio_path)
    #print(speech_segments)
    fbank_feat = {}
    for i, speech_seg in enumerate(speech_segments):
        fbank_feat[speech_segments_id[i]] = compute_fbank(speech_seg, subseg_cmn=True)

     
    subsegs, subseg_fbanks = [], []
    frame_shift=10
    window_secs = 300# 1.5
    period_secs = 0# 0.75
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    for seg_id, fbank in fbank_feat.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank=fbank, seg_id=seg_id, window_fs=window_fs, period_fs=period_fs, frame_shift=frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    

    embeddings = extract_embedding(subseg_fbanks, batch_size=1, subseg_cmn=True)

    subseg_i = {}
    emb_dict = OrderedDict()

    for i, subseg_id in enumerate(subsegs):
        subseg_i[subseg_id] = embeddings[i]
    
    for sub_seg_id, emb in subseg_i.items():
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {}
            emb_dict[utt]['sub_seg'] = []
            emb_dict[utt]['embs'] = []
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)
    
    subsegs_list = []
    embeddings_list = []
    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))

    #print(embeddings_list[0])
    
    labels_list,num_spks = cluster_probability(embeddings_list[0],num_spks_probabilitys=spks_num_probabilitys,rate=rate)
    
    return labels_list,embeddings,num_spks