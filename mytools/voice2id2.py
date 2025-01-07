import sys, time, os, tqdm, torch, argparse, glob, cv2, pickle
import soundfile
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from mytools import speech_enhancement
from mytools.fsmn_VAD import Fsmn_VAD
import os
import cv2 as cv
import sys

from VocalPrint import *

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

#读取说话分数矩阵
def read_2d_array_from_file(file_path):
		array = []
		with open(file_path, 'r') as file:
			for line in file:
				row = line.strip().split()  # 假设每个元素之间用空格分隔
				row = [x for x in row]  # 转换每个元素为整数
				array.append(row)
		return array

def read_set_from_file(file_path):
		array = set()
		with open(file_path, 'r') as file:
			for line in file:
				row = line.strip().split()
				print(row)
				array.add(row[-1])
		return array

#找到区间的重叠部分
def find_overlaps(intervals):
    if not intervals:
        return []
    
    # 排序
    intervals.sort(key=lambda x: x[0])
    
    overlaps = []
    current_start, current_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start <= current_end:
            # 存在重叠
            overlaps.append([max(start, current_start), min(end, current_end)])
            current_end = max(current_end, end)
        else:
            # 没有重叠
            current_start, current_end = start, end
    
    return overlaps
#去掉重叠部分
def subtract_overlaps(intervals, overlaps):
    result = []
    
    for start, end in intervals:
        temp_intervals = [[start, end]]
        
        for overlap_start, overlap_end in overlaps:
            new_intervals = []
            for int_start, int_end in temp_intervals:
                if int_end <= overlap_start or int_start >= overlap_end:
                    # 无重叠部分
                    new_intervals.append([int_start, int_end])
                else:
                    # 有重叠部分
                    if int_start < overlap_start:
                        new_intervals.append([int_start, overlap_start])
                    if int_end > overlap_end:
                        new_intervals.append([overlap_end, int_end])
            
            temp_intervals = new_intervals
        
        result.extend(temp_intervals)
    
    return result
# 得到去重叠化的声音片段区间
def remove_overlapping_parts(intervals):
    overlaps = find_overlaps(intervals)
    non_overlapping_intervals = subtract_overlaps(intervals, overlaps)
    return non_overlapping_intervals

def find_contained_intervals(A, B):
    """
    查找列表 A 中的每个区间是否包含列表 B 中的某个区间，并记录 B 中符合条件区间的下标。

    :param A: 包含区间的列表 A，每个区间是一个元组 (start, end)。
    :param B: 包含区间的列表 B，每个区间是一个元组 (start, end)。
    :return: 一个列表，其中每个元素是 A 中每个区间对应的 B 中区间的下标。
    """
    results = []
    
    for a_start, a_end in A:
        found_index = -1
        
        for index, (b_start, b_end) in enumerate(B):
            # 检查 A 中的区间是否包含 B 中的区间
            if a_start <= b_start and a_end >= b_end:
                found_index = index
                break  # 找到符合条件的区间后，跳出内层循环
        
        # 将找到的下标添加到结果列表中
        results.append(found_index)
    
    return results



def find_overlap_intervals(A,indices):
	results = []
	for i, (a_start, a_end) in enumerate(A):
		if indices[i]==-1:
			indices[i] = indices[i-1]   #先暂且让其和上一个段的id相同，避免出现除了自己以外没有包含其区间的情况
			for j, (a2_start, a2_end) in enumerate(A):
				if indices[j]!=-1:
					# 检查 A 中的区间是否包含 B 中的区间
					if a_start >= a2_start and a_end <= a2_end:
						indices[i] = indices[j]
						break
	return indices

def find_nocontained_intervals(A, B):
	"""
	查找列表 A 中的不包含列表 B 中的任何区间的区间
	:param A: 包含区间的列表 A，每个区间是一个元组 (start, end)。
	:param B: 包含区间的列表 B，每个区间是一个元组 (start, end)。
	:return: 一个A 中的不包含列表 B 中的任何区间的区间列表
	"""
	results = []
	overlap_indices = []
	for index, (a_start, a_end) in enumerate(A):        
		flag = 1
		for b_start, b_end in B:
			# 检查 A 中的区间是否包含 B 中的区间
			if a_start <= b_start and a_end >= b_end:
				flag=0
				break  # 找到符合条件的区间后，跳出内层循环
		if flag:
			results.append([a_start, a_end])
			overlap_indices.append(index)
	return results,overlap_indices

def save_matrix_to_txt(matrix, filename):
	np.savetxt(filename, matrix, delimiter='\t')


#根据a作为b中下标得到新数组
def get_elements_from_indices(a, b):
    return [b[index] for index in a if 0 <= index < len(b)]

#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
def has_intersection(seg1, seg2):
	return not (seg1[1] <= seg2[0] or seg2[1] <= seg1[0])

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

#求两个矩阵的余弦相似度矩阵
def cosine_similarity(M1, M2):
	M1_normalized = M1 / np.linalg.norm(M1, axis=1, keepdims=True)
	M2_normalized = M2 / np.linalg.norm(M2, axis=1, keepdims=True)
	return 0.5 * (1.0 + np.dot(M1_normalized, M2_normalized.T))

#实验一，只声纹平均    √	
#实验二，聚类数辅助  +声纹平均 √
#实验三，去噪+平均声纹  √
#实验四，啥也不加，直接聚类 √
#实验五，聚类数辅助    √
#实验六，去噪+声纹平均+聚类数辅助   √

#将声音片段进行去重叠
def voice2id_powerset_msdwild(save_path,num_test,ans,model):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'powerset_segments.txt'
	segments_final=[]
	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	# #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	# my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	# for segment in my_segments:
	# 	if segment[1]-segment[0]<200:
	# 		segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		# file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		# file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径
		file_path = save_path+'powerset_result.txt'
		data = read_set_from_file(file_path)
		print(data)
		person_num = len(data)
		# array = read_2d_array_from_file(file_path)
		# #print(np.shape(array))
		# person_num = 0   #由于一般msdwild没有头戴者
		# for row in array:
		# 	count = 0 
		# 	for item in row:
		# 		if float(item)!=0:
		# 			count = count+1
		# 	if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
		# 		person_num = person_num+1
		# print("predict_person_num:",person_num)

 

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [3,6]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		#clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments)

		# #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		# my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		# for segment in my_no_segments:
		# 	if segment[1]-segment[0]<200:
		# 		no_segments.remove([segment[0],segment[1]])

	origin_segments = segments	#保留原来的声音片段区间序列
	segments = remove_overlapping_parts(origin_segments)   #将声音片段区间序列进行去重叠化
	# print("origin_segments:",origin_segments)
	# print("segments",segments)
    #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	# my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	# for segment in my_segments:
	# 	if segment[1]-segment[0]<200:
	# 		segments.remove([segment[0],segment[1]])

	labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	if num_test in [4,5]:
		indices = find_contained_intervals(origin_segments,segments)
		indices = find_overlap_intervals(origin_segments,indices)
		id_final = get_elements_from_indices(indices,labels)
		overlap_segments,overlap_indices = find_nocontained_intervals(origin_segments,segments) #找到完全被包含重叠区域
		if len(overlap_segments)>0:
			overlap_embeddings = VocalPrint_embeddings(audio_path,overlap_segments) 
			similarity_matrix=cosine_similarity(overlap_embeddings,id_embeddings)
			save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_overlap_study"+str(name)+".txt")


			for i in range(len(similarity_matrix)):
				prince = similarity_matrix[i][:].tolist()  # 创建子列表的副本
				max_value = max(prince)
				max_index = prince.index(max_value)
				if max_index == id_final[overlap_indices[i]]:
					prince.remove(max_value)
					second_max = max(prince)
					max_index = prince.index(second_max)
					id_final[overlap_indices[i]] = max_index
				else:
					id_final[overlap_indices[i]] = max_index
		segments_final = origin_segments 


	if num_test in [1,3]:
		person_num = max(labels)
	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量

	sub_embeddings = []
	if num_test in [3,6]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			print(no_segments)
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			if len(sub_embeddings):
				#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
				sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
				save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_pyannote_study"+str(name)+".txt")
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
					if my_max>0.7:
					#if my_max>0.675:
						no_final_segments.append(no_segments[num])
						no_final_id.append(index)
					num = num+1

	if num_test in [1,2,3,6]:
		similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
		save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_pyannote_study"+str(name)+".txt")

		voice2id_final=[]

		for prince in similarity_matrix:
			my_max=0
			index=0
			for i in range(len(prince)):
				if prince[i]>my_max:
					my_max = prince[i]
					index=i
			voice2id_final.append(index)
		
		if num_test in [3,6]:
			if len(no_segments)>0:
				indices = find_contained_intervals(origin_segments,segments)
				print("indices:",indices)
				indices = find_overlap_intervals(origin_segments,indices)
				print("indices:",indices)
				id_final = get_elements_from_indices(indices,voice2id_final)
				print("id_final:",id_final)
				overlap_segments,overlap_indices = find_nocontained_intervals(origin_segments,segments) #找到完全被包含重叠区域
				overlap_embeddings = VocalPrint_embeddings(audio_path,overlap_segments) 
				similarity_matrix=cosine_similarity(overlap_embeddings,id_embeddings)
				save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_overlap_study"+str(name)+".txt")
				voice2id_final=[]
				print("id_final:",len(id_final))
				print("overlap_indices",overlap_indices)
				print("overlap_segments:",overlap_segments)
				print("origin_segments",origin_segments)
				for i in range(len(similarity_matrix)):
					prince = similarity_matrix[i][:].tolist()  # 创建子列表的副本
					max_value = max(prince)
					max_index = prince.index(max_value)
					print(overlap_indices[i])
					if max_index == id_final[overlap_indices[i]]:
						prince.remove(max_value)
						second_max = max(prince)
						max_index = prince.index(second_max)
						id_final[overlap_indices[i]] = max_index
					else:
						id_final[overlap_indices[i]] = max_index
				if len(sub_embeddings):
					id_final = id_final+no_final_id
					segments_final = origin_segments+no_final_segments
				else:
					segments_final = origin_segments
			else:
				indices = find_contained_intervals(origin_segments,segments)
				print("indices:",indices)
				indices = find_overlap_intervals(origin_segments,indices)
				print("indices:",indices)
				id_final = get_elements_from_indices(indices,voice2id_final)
				print("id_final:",id_final)
				overlap_segments,overlap_indices = find_nocontained_intervals(origin_segments,segments) #找到完全被包含重叠区域
				print("id_final:",len(id_final))
				print("overlap_indices",overlap_indices)
				print("overlap_segments:",overlap_segments)
				print("origin_segments",origin_segments)
				if len(overlap_segments)>0:
					overlap_embeddings = VocalPrint_embeddings(audio_path,overlap_segments) 
					similarity_matrix=cosine_similarity(overlap_embeddings,id_embeddings)
					save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_overlap_study"+str(name)+".txt")
					voice2id_final=[]

					for i in range(len(similarity_matrix)):
						prince = similarity_matrix[i][:].tolist()  # 创建子列表的副本
						max_value = max(prince)
						max_index = prince.index(max_value)
						if max_index == id_final[overlap_indices[i]]:
							prince.remove(max_value)
							second_max = max(prince)
							max_index = prince.index(second_max)
							id_final[overlap_indices[i]] = max_index
						else:
							id_final[overlap_indices[i]] = max_index
				segments_final = origin_segments

		else:
			indices = find_contained_intervals(origin_segments,segments)
			indices = find_overlap_intervals(origin_segments,indices)
			id_final = get_elements_from_indices(indices,voice2id_final)
			overlap_segments,overlap_indices = find_nocontained_intervals(origin_segments,segments) #找到完全被包含重叠区域
			if len(overlap_segments)>0:
				overlap_embeddings = VocalPrint_embeddings(audio_path,overlap_segments) 
				similarity_matrix=cosine_similarity(overlap_embeddings,id_embeddings)
				save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_overlap_study"+str(name)+".txt")
				voice2id_final=[]
				for i in range(len(similarity_matrix)):
					prince = similarity_matrix[i][:]  # 创建子列表的副本
					max_value = max(prince)
					max_index = prince.index(max_value)
					if max_index == id_final[overlap_indices[i]]:
						prince.remove(max_value)
						second_max = max(prince)
						max_index = prince.index(second_max)
						id_final[overlap_indices[i]] = max_index
					else:
						id_final[overlap_indices[i]] = max_index

			segments_final = origin_segments 




	np.savetxt(save_path+'final_segments_pyannote_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final



def voice2id_powerset_Easycom(save_path,num_test,ans,model):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'pyannote_powerset_pretrained.txt'
	segments_final=[]
	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	# 将区间列表转换为元组列表
	intervals_as_tuples = [tuple(interval) for interval in segments]

	# 使用集合去重
	unique_intervals_as_tuples = set(intervals_as_tuples)

	# 将集合中的元组转换回列表，并排序
	segments = sorted([list(interval) for interval in unique_intervals_as_tuples])
	# import copy
	# #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	# my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	# for segment in my_segments:
	# 	if segment[1]-segment[0]<200:
	# 		segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		#file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

		array = read_2d_array_from_file(file_path)
		#print(np.shape(array))
		person_num = 1
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				person_num = person_num+1
		print("person_num:",person_num)

 

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [3,6]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		# clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		# no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())
		no_segments = find_non_intersecting_segments(segments,clean_segments)

		# #去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		# my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		# for segment in my_no_segments:
		# 	if segment[1]-segment[0]<200:
		# 		no_segments.remove([segment[0],segment[1]])

		
	labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	if num_test in [4,5]:
		id_final = labels
		segments_final = segments

	if num_test in [1,3]:
		person_num = max(labels)+1
	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	if num_test in [1,2,3,6]:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	if num_test in [3,6]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			print(no_segments)
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			print(sub_embeddings)
			# print(id_embeddings)
			#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
			sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
			save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_powerset_study"+str(name)+".txt")

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
				if my_max>0.7:
				#if my_max>0.675:
					no_final_segments.append(no_segments[num])
					no_final_id.append(index)
				num = num+1

	if num_test in [1,2,3,6]:
		similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
		save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_powerset_study"+str(name)+".txt")

		voice2id_final=[]

		for prince in similarity_matrix:
			my_max=0
			index=0
			for i in range(len(prince)):
				if prince[i]>my_max:
					my_max = prince[i]
					index=i
			voice2id_final.append(index)
		
		if num_test in [3,6]:
			if len(no_segments)>0:
				id_final = voice2id_final + no_final_id
				segments_final = segments + no_final_segments
			else:
				id_final = voice2id_final 
				segments_final = segments
		else:
			id_final = voice2id_final
			segments_final = segments 


	np.savetxt(save_path+'final_segments_powerset_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_powerset_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final

