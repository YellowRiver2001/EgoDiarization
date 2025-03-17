from moviepy.editor import *
import numpy as np
from mytools import speech_enhancement
from mytools.fsmn_VAD import Fsmn_VAD
from spectral_cluster import cluster,cluster_probability
from VocalPrint import *
import itertools
import os

#将音频段划分为子音频段
def generate_subsegments(segments, subseg_length=1500, subseg_step=750):
	subsegments = []
	for start_time, end_time in segments:
		current_time = start_time
		while current_time + subseg_length <= end_time:
			subsegments.append([current_time, current_time + subseg_length])
			current_time += subseg_step

		# Check if there is a remaining segment smaller than subseg_length
		if current_time < end_time:
			subsegments.append([current_time, end_time])

	return subsegments

def read_2d_array_from_file(file_path):
		array = []
		with open(file_path, 'r') as file:
			for line in file:
				row = line.strip().split()  # 假设每个元素之间用空格分隔
				row = [x for x in row]  # 转换每个元素为整数
				array.append(row)
		return np.array(array).astype(np.float32)

#只得出未检测语音片段
def expand_time_segments(input_segments, total_range):
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
			full_segments.append([current_start, start - 1])
			# full_segments.append((start,end))
		# 更新当前总时间范围的起始时间
		current_start = max(end + 1, current_start)
	
	# 添加剩余的时间范围
	if current_start <= current_end:
		full_segments.append([current_start, current_end])
		
	return full_segments

#找到没有交集的区间
def find_non_intersecting_segments(segments1, segments2):
	non_intersecting_segments = []

	for seg2 in segments2:
		intersects = False
		for seg1 in segments1:
			if has_intersection(seg1, seg2):
				intersects = True
				break
		if not intersects:
			non_intersecting_segments.append(seg2)

	return non_intersecting_segments

#合并子段同类
def merge_subsegments(segments,labels):
	new_seg_list = []
	for i, seg in enumerate(segments):
		seg_st, seg_ed = seg
		seg_st = float(seg_st)
		seg_ed = float(seg_ed)
		cluster_id = labels[i]
		if i == 0:
			new_seg_list.append([ seg_st, seg_ed, cluster_id])
		elif cluster_id == new_seg_list[-1][2]:
			if seg_st > new_seg_list[-1][1]:
				new_seg_list.append([seg_st, seg_ed, cluster_id])
			else:
				new_seg_list[-1][1] = seg_ed
		else:
			if seg_st < new_seg_list[-1][1]:
				p = (new_seg_list[-1][1]+seg_st) / 2
				new_seg_list[-1][1] = p
				seg_st = p
			new_seg_list.append([seg_st, seg_ed, cluster_id])
	segments = []
	labels = []
	for item in new_seg_list:
		segments.append([int(item[0]),int(item[1])])
		labels.append(item[2])
	return segments,labels

#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
def has_intersection(seg1, seg2):
	return not (seg1[1] <= seg2[0] or seg2[1] <= seg1[0])
	#求两个矩阵的余弦相似度矩阵

def cosine_similarity(M1, M2):
	M1_normalized = M1 / np.linalg.norm(M1, axis=1, keepdims=True)
	M2_normalized = M2 / np.linalg.norm(M2, axis=1, keepdims=True)
	return 0.8 * (1.0 + np.dot(M1_normalized, M2_normalized.T))

def save_matrix_to_txt(matrix, filename):
	np.savetxt(filename, matrix, delimiter='\t')

'''
def cal_speaker_vector use the asd output: array and pyannote output: segments
example: if the array like [[0, 0, 0.1, 0.2, 0][0, 0, 0.1, 0.2, 0]]there are 2 speakers
and the segments like [[0, 1000], [2000, 3000]] 
the speaker_vectors have n rows, which is the number of segments rows, 
and 2 columns, which is the number of speakers
so we get like this nparray
    [0.55, 0],  # segment 1 has 2 speakers, the first speaker has 0.55 score, the second speaker has 0.55 score
    [0, 0.9]    # segment 2 has 2 speakers, the first speaker has 0 score, the second speaker has 0.9 score
'''
def cal_speaker_vector(array,segments):
    speaker_vectors = []
    for segment in segments:
        l=int(int(segment[0])*25/1000)
        r=int(int(segment[1])*25/1000)
        speaker_vector = []
        for row in array:
            speaker_vector.append(np.mean(row[l:r+1]))
        speaker_vectors.append(speaker_vector)
    return np.array(speaker_vectors)



#计算每个类中平均值
def embeddings_mean(segments_embeddings,labels):
	m = max(labels)+1
	# print("m:",m)
	class_embeddings = [[] for _ in range(m)]
	i = 0
	for embedding in segments_embeddings:
			class_embeddings[labels[i]].append(embedding)
			i=i+1
	# print(class_embeddings)
	id_embeddings = []
	for class_embedding in class_embeddings:
			id_embeddings.append(np.mean(np.array(class_embedding),axis=0))
	return np.array(id_embeddings)

def calculate_probabilities(probabilities):
    n = len(probabilities)
    result = [0] * (n + 1)  # 初始化结果列表，长度为n+1
    
    # 遍历所有可能的说话人数k (1到n)
    for k in range(1, n + 1):
        # 获取所有k个人的组合
        for combination in itertools.combinations(range(n), k):
            prob = 1.0
            for i in range(n):
                if i in combination:
                    prob *= probabilities[i]
                else:
                    prob *= (1 - probabilities[i])
            result[k] += prob
            
    return result

def read_txt(file_path):
		array = []
		with open(file_path, 'r') as file:
			for line in file:
				row = line.strip().split()  # 假设每个元素之间用空格分隔
				row = [float(x) for x in row]  # 转换每个元素为整数
				array.append(row)
		return array

def find_max_index(lst):
    max_value = max(lst)
    max_index = max(i for i, v in enumerate(lst) if v == max_value)
    return max_index

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def score_to_probability(score):
    # 将分数映射到概率范围
    if score<=0:
        return 0
    else:
        probability = score+0.7
    return probability
#计算每个类中最小相似度
def cal_class_min_sim(similarity_matrix,labels):
	m = len(similarity_matrix[0])
	class_sims = [[] for _ in range(m)]
	i = 0
	for prince in similarity_matrix:
			class_sims[labels[i]].append(prince[labels[i]])
			i=i+1
	# print(class_sims)

	min_sim = []
	for class_sim in class_sims:
			min_sim.append(min(class_sim))
	return min_sim

#计算每个类中平均值
def cal_class_mean_sim(similarity_matrix,labels):
	m = len(similarity_matrix[0])
	class_sims = [[] for _ in range(m)]
	i = 0
	for prince in similarity_matrix:
			class_sims[labels[i]].append(prince[labels[i]])
			i=i+1
	# print(class_sims)
	min_sim = []
	for class_sim in class_sims:
			min_sim.append(np.mean(class_sim))
	return min_sim


def find_missing_intervals(intervals, total_range):
    # 将区间列表按照开始值排序
    intervals.sort()

    # 初始化结果列表和当前开始位置
    missing_intervals = []
    current_start = total_range[0]

    for interval in intervals:
        start, end = interval

        if current_start < start:
            missing_intervals.append([current_start, start])

        current_start = max(current_start, end)

    if current_start < total_range[1]:
        missing_intervals.append([current_start, total_range[1]])

    return missing_intervals

def interval_intersection(list1, list2):
    intersections = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        start1, end1 = list1[i]
        start2, end2 = list2[j]

        start = max(start1, start2)
        end = min(end1, end2)
        if start <= end:
            intersections.append([start, end])
        if end1 < end2:
            i += 1
        else:
            j += 1

    return intersections

def add_no_segments(segments1, segments2,total_range):
    missing_intervals = find_missing_intervals(segments1, total_range)
    intersection = interval_intersection(missing_intervals, segments2)
    return intersection


#实验一，啥也不加   
#实验二，只增强
#实验三，只聚类辅助
#实验四，只视听向量
#实验五，增强+聚类辅助   
#实验六，增强+视听向量
#实验七，聚类数辅助+视听向量
#实验八，啥都加
#实验九，用ground_truth的聚类数进行聚类辅助

#rate,clean_segments,add_no_segments.

#没有微调的消融实验
def voice2id_pyannote2_final(file_id, audio_path_ori, segments_path_ori,num_test,a,ans,model):
    # 视频文件路径
    audio_path = audio_path_ori # 指定的音频文件路径
    segments_path = os.path.join(segments_path_ori, "test_powerset_pretrained_segments.txt")   #只用segmentation模型的结果
    save_path = segments_path_ori
    # print("save_path", save_path)
    #segments = [[2866, 3754], [4061, 5511], [5221, 6313], [8566, 9795], [14573, 16075]]
    # print("audio_path:",audio_path)
    # print("segments_path:",segments_path)

    segments = []
    with open(segments_path,'r') as f:
        for line in f:
            # 去除行末的换行符并按分隔符分割
            row = line.strip().split(' ')
            segments.append([int(row[0]),int(row[1])])
    import copy
    #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
    #去掉了小于200的segments,也就是小于200ms的segments 0.2s
    my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
    for segment in my_segments:
        if segment[1]-segment[0]<400:
            segments.remove([segment[0],segment[1]])
    # 使用列表推导式确保过滤小段
    # segments = [seg for seg in segments if seg[1] - seg[0] >= 500]  # 增加阈值到500ms更安全
    # print("segments:",segments)
    if num_test == 8:
        person_num=None
        #聚类数辅助
        # 读取说话分数矩阵
        #此文件格式为每一行代表一个说话人的说话分数,例如一个视频中检测到3人说话
        #则文件有3行,列为帧数,其中概率为说话人概率
        file_path = save_path+'/speaker_global_EGO4d.txt'  # 文件路径  

        array = read_2d_array_from_file(file_path)
        person_scores=[]
        for row in array:
            count = 0 
            for item in row:
                if float(item)!=0:
                    count = count+1
            if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
                # person_num = person_num+1
                person_scores.append(score_to_probability(max(row)))
        # 将一行中最大的概率值提取出来,并将其转换为概率值(如果全局概率都小于0,返回0)
        # person_scores的形状为
        speakers_probabilities = calculate_probabilities(person_scores)
        # get the speakers_probabilities like [0, 0.1, 0.7] when there are 2 speakers
        # no spekaer with 0, 1 spekaer with 0.1, 2 speakers with 0.7 
        num_spks_probabilitys = speakers_probabilities
        person_num = find_max_index(speakers_probabilities) +1 # asd does not get the camera speaker
        # print("visual:",person_num)
        #计算说话分数向量
        speaker_vector = cal_speaker_vector(array,segments)

        #segments_embeddings1 is (N, 192) which N is the counts of segments
        #speaker_vector is (N, speakers) 
        segments_embeddings = VocalPrint_embeddings(audio_path,segments)  #计算每个声音片段的声纹向量与说话分数向量并将其合并,然后进行谱聚类
        segments_embeddings = np.concatenate((segments_embeddings, speaker_vector), axis=1)  #合并


        total_range = [0,300000]
        #声音增强后处理

        # split_path = save_path+'split_wav_powerset/'   #用于存放被分段的音频文件
        # clean_path = save_path+'clean_wav_powerset/'   #用于存放被音频增强（去噪）的音频文件
        split_path = save_path+'/split_wav_powerset_segments/'   #用于存放被分段的音频文件    powerset第一阶段segmataiton的segments
        clean_path = save_path+'/clean_wav_powerset_segments/'   #用于存放被音频增强（去噪）的音频文件    
        # split_path = save_path+'split_wav_powerset_Ego4d_segments/'   #用于存放被分段的音频文件    powerset第一阶段segmataiton的segments
        # clean_path = save_path+'clean_wav_powerset_Ego4d_segments/'   #用于存放被音频增强（去噪）的音频文件   
        # split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
        # clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

        # if a==0:
        # #已经跑过一遍，这边暂时先注释
        speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
        clean_audio_path = clean_path+'/clean_audio.wav'   #分割去噪后的音频文件路径
        #对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
        # if a==0:
        clean_segments = Fsmn_VAD(clean_audio_path,save_path+'/clean_segments_pretrained.txt',model)
        # clean_segments = np.loadtxt(save_path+'clean_segments_pretrained.txt')
        # clean_segments = np.loadtxt(save_path+'clean_nosplit_segments.txt')   #这里其实时powerset结果按照split方法增强后的结果，这里名字当时弄错了
        #找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
        # no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())
        no_segments = find_non_intersecting_segments(segments,clean_segments)


        #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
        my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
        for segment in my_no_segments:
            if segment[1]-segment[0]<200:
                no_segments.remove([segment[0],segment[1]])
        no_speaker_vector = cal_speaker_vector(array,no_segments)
        
        # print(segments)
        labels,person_num = cluster_probability(segments_embeddings,num_spks_probabilitys=num_spks_probabilitys,rate = a)

        #对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
        id_embeddings = embeddings_mean(segments_embeddings,labels)

        if len(no_segments)>0:
            sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
            sub_embeddings = np.concatenate((sub_embeddings, no_speaker_vector), axis=1)  #合并
            #将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
            sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
            save_matrix_to_txt(sub_similarity_matrix,save_path+"/voiceprint_id_similarity_no_pyannote_study"+str(num_test)+".txt")
            similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
            #计算每个类中的与其平均声纹向量的相似度的平均值
            class_mean_sim = cal_class_mean_sim(similarity_matrix,labels)

            no_final_id=[]
            no_final_segments = []
            num = 0
            for embedding in sub_similarity_matrix:
                my_max=0
                index=0
                for i in range(len(embedding)):
                    if embedding[i]>my_max:
                        my_max = embedding[i]
                        index=i
                #if my_max>=0.65:
                if my_max>=class_mean_sim[index]:  #如果与当前平均声纹相似度大于等于该类中相似度平均值的，则加入。
                #if my_max>0.675:
                    no_final_segments.append(no_segments[num])
                    no_final_id.append(index)
                num = num+1		

            id_final = labels.tolist() + no_final_id
            segments_final = segments + no_final_segments
        else:
            id_final = labels.tolist()
            segments_final = segments


        np.savetxt(save_path+'/final_segments_powset_final_study'+str(num_test)+'_segments.txt',segments_final,fmt='%d')
        np.savetxt(save_path+'/id_segments_powset_final_study'+str(num_test)+'.txt',id_final,fmt='%d')
        return id_final,segments_final



