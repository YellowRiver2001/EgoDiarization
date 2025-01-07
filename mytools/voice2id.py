import sys, time, os, tqdm, torch, argparse, glob, subprocess, cv2, pickle, numpy, pdb, math, python_speech_features
import soundfile
from moviepy.editor import *
import ast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# from mytools import speech_enhancement
# from mytools.fsmn_VAD import Fsmn_VAD
import os
import cv2 as cv
import sys

from VocalPrint import *
# from self_attention_train1 import *
#from self_attention_train_train import *



'''
实验1：先vad,得到语音段，再根据语音段，对未检测语音段按每1.5s进行划分，分别进行去噪，再合并，然后再进行vad，
然后用前面的语音检测的段在去噪后的视频上进行子段声纹特征提取，聚类，合并同类段，然后再声纹提取，聚类，
然后计算每个类的声纹向量平均值，然后对上述第二次vad的语音段与第一次vad语音段每交集的语音段（即去噪后又多检测出的语音段）
然后对这些语音段进行子段划分，声纹提取，聚类，合并同类段，再声纹提取，然后，和每个说话人代表声纹向量进行
相似度比对，大于一定阈值的进行划分。


实验2：先vad,得到语音段，再根据语音段，对未检测语音段按每1.5s进行划分，分别进行去噪，再合并，然后再进行vad，
然后用前面的语音检测的段在原音频上进行子段声纹特征提取，聚类，合并同类段，然后再声纹提取，聚类，
然后计算每个类的声纹向量平均值，然后对上述第二次vad的语音段与第一次vad语音段每交集的语音段（即去噪后又多检测出的语音段）
然后对这些语音段在原音频上进行子段划分，声纹提取，聚类，合并同类段，再声纹提取，然后，和每个说话人代表声纹向量进行
相似度比对，大于一定阈值的进行划分。

实验3：不对未检测的语音进行划分，（因为划分后会出现很多噪音还是去不掉，vad的时候造成较高的误检率）,就对整段的未检测语音进行去噪.

实验4：用检测出的漏检语音段，加入之前检测的语音段列表一起，然后作为segments，之后和sub_vad2方法一样。
实验5： 去掉最后筛选的阈值
实验5：不用视觉的人数，试试。

实验六：调整全局说话人的策略，从50-100
实验七：用 segments_pyannote3.1的segments结果作为vad
'''
def whisper2id_with_segments_spk_num_2_voiceprince_clean(choice,save_path,num_test,ans,model):
	#获取视频名

	#save_path = os.path.join(args.videoFolder, args.videoName,'pyavi/')
	#语音识别whisperX,语音分段
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	if choice=='3D':
		segments_path = save_path+'segment_3D-speaker.txt'
		#segments_path = save_path+'segments_pyannote3.1.txt'
		#segments_path = save_path+'subsegment_3D-speaker.txt'
	if choice=='whisperx':
		segments_path = save_path+'segments.txt'
	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	def read_2d_array_from_file(file_path):
			array = []
			with open(file_path, 'r') as file:
				for line in file:
					row = line.strip().split()  # 假设每个元素之间用空格分隔
					row = [x for x in row]  # 转换每个元素为整数
					array.append(row)
			return array

	# 读取说话分数矩阵
	file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
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
	print(person_num)
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
	total_range = [0,300000]
	#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
	split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
	clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件
	#已经跑过一遍，这边暂时先注释
	#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split')
	if num_test==3:
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'no_split',ans)
		clean_audio_path = clean_path+'clean_audio_nosplit.wav'   #去噪后的音频文件路径

	if num_test==1 or num_test==2:
		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #去噪后的音频文件路径


	#对去噪后的音频进行重新语音活动检测
	#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_segments.txt',model)
	clean_segments = np.loadtxt(save_path+'clean_segments.txt')

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
	#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
	no_segments = find_non_intersecting_segments(segments,clean_segments)


	#对检测到的语音划分子段
	sub_segments = generate_subsegments(segments)



	#person_num = None
	if num_test==1:
		labels,_ = VocalPrint_Detect_spk_num(clean_audio_path,sub_segments,person_num)  #得到每个语音段对应的id
		#labels = VocalPrint_Detect_spk_num(audio_path,segments,None)  #得到每个语音段对应的id
	if num_test==2 or num_test==3:
		labels,_ = VocalPrint_Detect_spk_num(audio_path,sub_segments,person_num)  #得到每个语音段对应的id


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
	#将所有同类子段合并，形成新的语音段
	segments,labels  =  merge_subsegments(sub_segments,labels)

	print(segments)

	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments2 = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments2:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])
	
	if num_test==1:
		#对用子段划分好的语音段，再次聚类。
		labels ,segments_embeddings= VocalPrint_Detect_spk_num(clean_audio_path,segments,person_num)  #得到每个语音段对应的id
	if num_test==2 or num_test==3:
		labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id



	print("my_labels:")
	print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	if num_test==1:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(clean_audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
	if num_test==2 or num_test==3:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	#对剩余重新检测到的音频片段划分子段，然后依次计算声纹向量，和每个人的声纹向量进行比对
	subsegments = generate_subsegments(no_segments)
	my_no_segments = copy.deepcopy(subsegments)  #这里需要深度拷贝
	for segment in my_no_segments:
		if segment[1]-segment[0]<100:
			subsegments.remove([segment[0],segment[1]])

	if num_test==1:
		no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(clean_audio_path,subsegments,None)  #得到每个语音段对应的id
	if num_test==2 or num_test==3:
		no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,subsegments,None)  #得到每个语音段对应的id

	no_segments,no_labels  =  merge_subsegments(subsegments,no_labels)

	#获得每个子段的声纹向量（同类子段合并后的）
	if num_test==1:
		sub_embeddings = VocalPrint_embeddings(clean_audio_path,no_segments) 
	if num_test==2 or num_test==3:
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

	#求两个矩阵的余弦相似度矩阵
	def cosine_similarity(M1, M2):
		M1_normalized = M1 / np.linalg.norm(M1, axis=1, keepdims=True)
		M2_normalized = M2 / np.linalg.norm(M2, axis=1, keepdims=True)
		return 0.5 * (1.0 + np.dot(M1_normalized, M2_normalized.T))

	def save_matrix_to_txt(matrix, filename):
		np.savetxt(filename, matrix, delimiter='\t')


	#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
	sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
	save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_segments_clean6.txt")

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
		if my_max>=0.65:
			no_final_segments.append(no_segments[num])
			no_final_id.append(index)
		num = num+1

	similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
	
	# save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_custer2.txt")



	voice2id_final=[]

	for prince in similarity_matrix:
		my_max=0
		index=0
		for i in range(len(prince)):
			if prince[i]>my_max:
				my_max = prince[i]
				index=i
		voice2id_final.append(index)
	

	id_final = voice2id_final + no_final_id
	segments_final = segments + no_final_segments

	np.savetxt(save_path+'final_segments_clean6.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_clean6.txt',id_final,fmt='%d')

	print("id_segments_clean6:",id_final)


	return id_final,segments_final


'''
实验4：用检测出的漏检语音段，加入之前检测的语音段列表一起，然后作为segments，之后和sub_vad2方法一样。
实验5：去掉最后筛选的阈值
'''

def whisper2id_with_segments_spk_num_2_voiceprince_clean4(choice,save_path,num_test,ans,model):

	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	if choice=='3D':
		segments_path = save_path+'segment_3D-speaker.txt'
		#segments_path = save_path+'subsegment_3D-speaker.txt'
	if choice=='whisperx':
		segments_path = save_path+'segments.txt'
	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	def read_2d_array_from_file(file_path):
			array = []
			with open(file_path, 'r') as file:
				for line in file:
					row = line.strip().split()  # 假设每个元素之间用空格分隔
					row = [x for x in row]  # 转换每个元素为整数
					array.append(row)
			return array

	# 读取说话分数矩阵
	file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
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
	print(person_num)
	person_num = None
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
	total_range = [0,300000]
	#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
	split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
	clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

	#已经跑过一遍，这边暂时先注释
	#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
	clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

	#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
	#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
	clean_segments = np.loadtxt(save_path+'clean_segments.txt')

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
	#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
	no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
	for segment in my_no_segments:
		if segment[1]-segment[0]<100:
			no_segments.remove([segment[0],segment[1]])

	#将没检测到的语音段扩充到原检测的语音段列表中
	segments = segments+no_segments

	segments.sort()
	#print(segments)
	#对检测到的语音划分子段
	sub_segments = generate_subsegments(segments)
	
	if num_test==1:
		labels,_ = VocalPrint_Detect_spk_num(clean_audio_path,sub_segments,person_num)  #得到每个语音段对应的id
		#labels = VocalPrint_Detect_spk_num(audio_path,segments,None)  #得到每个语音段对应的id
	if num_test==2 or num_test==3:
		labels,_ = VocalPrint_Detect_spk_num(audio_path,sub_segments,person_num)  #得到每个语音段对应的id


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
	#将所有同类子段合并，形成新的语音段
	segments,labels  =  merge_subsegments(sub_segments,labels)

	print(segments)
	if num_test==1:
		#对用子段划分好的语音段，再次聚类。
		labels ,segments_embeddings= VocalPrint_Detect_spk_num(clean_audio_path,segments,person_num)  #得到每个语音段对应的id
	if num_test==2 or num_test==3:
		labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	person_num = max(labels)+1

	print("my_labels:")
	print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	if num_test==1:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(clean_audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
	if num_test==2 or num_test==3:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	

	#求两个矩阵的余弦相似度矩阵
	def cosine_similarity(M1, M2):
		M1_normalized = M1 / np.linalg.norm(M1, axis=1, keepdims=True)
		M2_normalized = M2 / np.linalg.norm(M2, axis=1, keepdims=True)
		return 0.5 * (1.0 + np.dot(M1_normalized, M2_normalized.T))

	def save_matrix_to_txt(matrix, filename):
		np.savetxt(filename, matrix, delimiter='\t')


	#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
	similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
	save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_clean5.txt")



	id_final=[]
	segments_final=[]
	num=0
	for prince in similarity_matrix:
		my_max=0
		index=0
		for i in range(len(prince)):
			if prince[i]>my_max:
				my_max = prince[i]
				index=i
		if num_test == 2 or num_test == 1:
			if my_max>0.65:
				id_final.append(index)
				segments_final.append(segments[num])
		if num_test == 3:
			id_final.append(index)
			segments_final.append(segments[num])
		num=num+1
	


	np.savetxt(save_path+'final_clean5_segments.txt',segments_final,fmt='%d')
	np.savetxt(save_path+'id_clean5.txt',id_final,fmt='%d')
	print("final_id:",id_final)


	return id_final,segments_final

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
	return 0.5 * (1.0 + np.dot(M1_normalized, M2_normalized.T))

def save_matrix_to_txt(matrix, filename):
	np.savetxt(filename, matrix, delimiter='\t')
	
#实验1：pyannote的segment作为vad结果其余和实验6一样        
#实验2：pyannote的segment作为vad结果，做去噪，不做子段划分和双聚类 
#实验3：pyannote的segment作为vad结果，不做去噪，只做声纹平均
#实验4：用ground_truth作为说话人数的参考，其余和实验2一样

def voice2id_pyannote(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'segments_pyannote3.1.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	# 读取说话分数矩阵
	file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
	array = read_2d_array_from_file(file_path)
	#print(np.shape(array))
	person_num = 1
	for row in array:
		count = 0 
		for item in row:
			if float(item)!=0:
				count = count+1
		if count>50:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
			person_num = person_num+1
	print(person_num)

	if num_test==4:
		person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
		print('ground_truth:',person_num)
		num_test = 2

	total_range = [0,300000]

	if num_test==2 :
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')





		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				no_segments.remove([segment[0],segment[1]])

	if num_test == 1:
		#对检测到的语音划分子段
		sub_segments = generate_subsegments(segments)
		labels,_ = VocalPrint_Detect_spk_num(audio_path,sub_segments,person_num)  #得到每个语音段对应的id
		#将所有同类子段合并，形成新的语音段
		segments,labels  =  merge_subsegments(sub_segments,labels)

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_segments2 = copy.deepcopy(segments)  #这里需要深度拷贝
		for segment in my_segments2:
			if segment[1]-segment[0]<100:
				segments.remove([segment[0],segment[1]])

		#对剩余重新检测到的音频片段划分子段，然后依次计算声纹向量，和每个人的声纹向量进行比对
		subsegments = generate_subsegments(no_segments)
		my_no_segments = copy.deepcopy(subsegments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				subsegments.remove([segment[0],segment[1]])
		
		no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,subsegments,None)  #得到每个语音段对应的id
		no_segments,no_labels  =  merge_subsegments(subsegments,no_labels)

		
	labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	if num_test==2:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

		#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
		sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
		save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_segments_pyannote"+str(name)+".txt")

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
			#if my_max>0.7:
			if my_max>0.675:
				no_final_segments.append(no_segments[num])
				no_final_id.append(index)
			num = num+1


	similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
	save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_pyannote"+str(name)+".txt")

	voice2id_final=[]

	for prince in similarity_matrix:
		my_max=0
		index=0
		for i in range(len(prince)):
			if prince[i]>my_max:
				my_max = prince[i]
				index=i
		voice2id_final.append(index)
	
	if num_test==2:
		id_final = voice2id_final + no_final_id
		segments_final = segments + no_final_segments
	if num_test==3: 
		id_final = voice2id_final
		segments_final = segments 

	np.savetxt(save_path+'final_segments_pyannote'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final


#实验一，只声纹平均    √	
#实验二，聚类数辅助  +声纹平均 √
#实验三，去噪+平均声纹  √
#实验四，啥也不加，直接聚类 √
#实验五，聚类数辅助    √
#实验六，去噪+声纹平均+聚类数辅助   √
#实验七，去噪+聚类数辅助



# def voice2id_pyannote2(save_path,file_no,num_test):
def voice2id_pyannote2(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#audio_path = '/home/rx/audio-visual/active-speaker-detection/vad/audios/'+str(file_no)+'.wav'

	#segments_path = save_path+'segments_pyannote3.1.txt'
	#segments_path = save_path+'pyannote_Powerset.txt'
	segments_path = save_path+'segments_Powerset_Ego4d.txt'
	#segments_path = save_path+'pyannote_Powerset_pretrained.txt'
	#segments_path = save_path+'powerset_segments.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<200:
			segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6,7]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		#file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

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
		print(person_num)

 

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [3,6,7]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<200:
				no_segments.remove([segment[0],segment[1]])

		
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
	if num_test in [1,2,3,6,7]:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值

	if num_test in [3,6,7]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

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
				id_final = voice2id_final + no_final_id
				segments_final = segments + no_final_segments
			else:
				id_final = voice2id_final 
				segments_final = segments
		else:
			id_final = voice2id_final
			segments_final = segments 

	if num_test==7:
		if len(no_segments)>0:
			id_final = labels.tolist() + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = labels.tolist()
			segments_final = segments



	np.savetxt(save_path+'final_segments_powerset_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_powerset_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final





#实验一，啥也不加
#实验二，去噪+双聚类+聚类数辅助+声纹平均
#实验三，去掉去噪
#实验四，去掉双聚类
#实验五，去掉聚类数辅助 
#实验六，去掉声纹平均 附带会去掉去噪



def voice2id_3D_Speaker_study(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#segments_path = save_path+'segments_pyannote3.1.txt'
	segments_path = save_path+'segment_3D-speaker.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	person_num=None

	#聚类数辅助
	if num_test in [2,3,4,6]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		array = read_2d_array_from_file(file_path)
		#print(np.shape(array))
		person_num = 1
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>50:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				person_num = person_num+1
		print(person_num)

 

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [2,4,5]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')





		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				no_segments.remove([segment[0],segment[1]])

	#对检测到的语音划分子段
	sub_segments = generate_subsegments(segments)
	labels,_ = VocalPrint_Detect_spk_num(audio_path,sub_segments,person_num)  #得到每个语音段对应的id
	#将所有同类子段合并，形成新的语音段
	segments,labels  =  merge_subsegments(sub_segments,labels)
	if num_test ==4:
		_,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id


	if num_test in [2,3,5,6]:

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_segments2 = copy.deepcopy(segments)  #这里需要深度拷贝
		for segment in my_segments2:
			if segment[1]-segment[0]<100:
				segments.remove([segment[0],segment[1]])
		labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	if num_test in [2,4,5]:
		#对剩余重新检测到的音频片段划分子段，然后依次计算声纹向量，和每个人的声纹向量进行比对
		subsegments = generate_subsegments(no_segments)
		my_no_segments = copy.deepcopy(subsegments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				subsegments.remove([segment[0],segment[1]])
		
		no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,subsegments,None)  #得到每个语音段对应的id
		no_segments,no_labels  =  merge_subsegments(subsegments,no_labels)

	#不做声纹平均的实验
	if num_test in [1,6]:
		id_final = labels
		segments_final = segments
	#不做聚类数辅助的实验
	if num_test in [1,5]:
		person_num = max(labels)+1
	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	#做声纹平均的实验
	if num_test in [2,3,4,5]:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
	#加了去噪的实验
	if num_test in [2,4,5]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

		#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
		sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
		save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_segments_3D-Speaker_study"+str(name)+".txt")

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

	#加了声纹平均的实验
	if num_test in [2,3,4,5]:
		similarity_matrix=cosine_similarity(segments_embeddings,id_embeddings)
		save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_3D-Speaker_study"+str(name)+".txt")

		voice2id_final=[]

		for prince in similarity_matrix:
			my_max=0
			index=0
			for i in range(len(prince)):
				if prince[i]>my_max:
					my_max = prince[i]
					index=i
			voice2id_final.append(index)
		
		if num_test in [2,4,5,6]:
			id_final = voice2id_final + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = voice2id_final
			segments_final = segments 




	np.savetxt(save_path+'final_segments_3D-Speaker_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_3D-Speaker_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final


#实验一，只声纹平均    
#实验二，聚类数辅助  +声纹平均 
#实验三，去噪+平均声纹  
#实验四，啥也不加，直接聚类 
#实验五，聚类数辅助    
#实验六，去噪+声纹平均+聚类数辅助   √
#实验七，去噪+聚类数辅助

def voice2id_pyannote_attention_study(save_path,num_test):


	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'segments_pyannote3.1.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		array = read_2d_array_from_file(file_path)
		#print(np.shape(array))
		person_num = 1
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>50:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				person_num = person_num+1
		print(person_num)


	total_range = [0,300000]

	if num_test in [3,6]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')





		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				no_segments.remove([segment[0],segment[1]])

		
	labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	segments_embeddings_labels = {}
	segments_embeddings_labels['labels'] =labels
	segments_embeddings_labels['segments_embeddings'] =segments_embeddings

	# save_pkl_path = save_path+'segments_embeddings_labels.pkl'
	# #将数据写入 .pkl 文件
	# with open(save_pkl_path, 'wb') as file:
	# 	pickle.dump(segments_embeddings_labels, file)


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
		#id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
		seq_length = 200
		id_embeddings = single_inference_data(save_path,segments,segments_embeddings,labels,seq_length)

	if num_test in [3,6]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
		#save_pkl_path = save_path+'no_segments_embeddings_labels.pkl'
		# #将数据写入 .pkl 文件
		# with open(save_pkl_path, 'wb') as file:
		# 	pickle.dump(sub_embeddings, file)
		#将数据从pkl文件中读取
		# with open(save_pkl_path, 'rb') as file:
		# 	sub_embeddings = pickle.load(file)

		#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
		sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
		save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_attention_train_study"+str(name)+".txt")

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
		save_matrix_to_txt(similarity_matrix,save_path+"voiceprint_id_similarity_attention_train_study"+str(name)+".txt")

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
			id_final = voice2id_final + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = voice2id_final
			segments_final = segments 




	np.savetxt(save_path+'final_segments_attention_train_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_attention_train_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final

import itertools

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


def voice2id_pyannote2_person_num_study(save_path,num_test):
	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'segments_pyannote3.1.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	num_spks_probabilitys=None
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def score_to_probability(score):
		# 将分数映射到概率范围
		if score<=0:
			return 0
		else:
			probability = score+0.7
		return probability

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		array = read_txt(file_path)
		# persons_num = len(array)   #检测到的总人数
		#print(np.shape(array))
		person_scores=[]
		person_score = []
		for row in array:
			count = 0 
			for item in row:
				if item!=0:
					count = count+1
			if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				#person_num = person_num+1
				person_scores.append(score_to_probability(max(row)))
			#person_score.append(max(row))
		speakers_probabilities = calculate_probabilities(person_scores)
		num_spks_probabilitys = speakers_probabilities
		speakers_num = find_max_index(speakers_probabilities) + 1
		person_num = speakers_num
		print("visual:",person_num)
		# print(num_spks_probabilitys)

 

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
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')





		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				no_segments.remove([segment[0],segment[1]])

		
	labels ,segments_embeddings,person_num= VocalPrint_Detect_spk_num_probability(audio_path,segments,num_spks_probabilitys,0.8)  #得到每个语音段对应的id
	print("viusal+cluster:",person_num)

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
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

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
			id_final = voice2id_final + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = voice2id_final
			segments_final = segments 




	np.savetxt(save_path+'final_segments_pyannote_speaker_num_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote_speaker_num_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final


def voice2id_pyannote2_person_num_attention_study(save_path,num_test):
	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'segments_pyannote3.1.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<100:
			segments.remove([segment[0],segment[1]])

	num_spks_probabilitys=None
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def score_to_probability(score):
		# 将分数映射到概率范围
		if score<=0:
			return 0
		else:
			probability = score+0.7
		return probability

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		array = read_txt(file_path)
		persons_num = len(array)   #检测到的总人数
		#print(np.shape(array))
		person_scores=[]
		person_score = []
		for row in array:
			count = 0 
			for item in row:
				if item!=0:
					count = count+1
			if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				#person_num = person_num+1
				person_scores.append(score_to_probability(max(row)))
			#person_score.append(max(row))
		speakers_probabilities = calculate_probabilities(person_scores)
		num_spks_probabilitys = speakers_probabilities
		speakers_num = find_max_index(speakers_probabilities) +1
		person_num = speakers_num
		print("visual:",person_num)
		# print(num_spks_probabilitys)

 

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
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')





		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<100:
				no_segments.remove([segment[0],segment[1]])

		
	labels ,segments_embeddings,person_num= VocalPrint_Detect_spk_num_probability(audio_path,segments,num_spks_probabilitys,0.8)  #得到每个语音段对应的id
	print("viusal+cluster:",person_num)

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
		#id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
		seq_length = 200
		id_embeddings = single_inference_data(save_path,segments,segments_embeddings,labels,seq_length)
	if num_test in [3,6]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

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
			id_final = voice2id_final + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = voice2id_final
			segments_final = segments 




	np.savetxt(save_path+'final_segments_pyannote_speaker_num_attention_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote_speaker_num_attention_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final



def voice2id_pyannote2_new_attention(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#audio_path = '/home/rx/audio-visual/active-speaker-detection/vad/audios/'+str(file_no)+'.wav'

	#segments_path = save_path+'segments_pyannote3.1.txt'
	#segments_path = save_path+'pyannote_Powerset.txt'
	segments_path = save_path+'segments_Powerset_Ego4d.txt'
	#segments_path = save_path+'pyannote_Powerset_pretrained.txt'
	#segments_path = save_path+'powerset_segments.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<200:
			segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		#file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

		array = read_2d_array_from_file(file_path)
		#print(np.shape(array))
		person_num = 1
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>50:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				person_num = person_num+1
		print(person_num)

 

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
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<200:
				no_segments.remove([segment[0],segment[1]])

		
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
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			# 在第一维上拼接
			final_embeddings = np.vstack((np.array(id_embeddings), np.array(segments_embeddings), np.array(sub_embeddings)))
			segments_final = segments + no_segments
			id_final = single_inference_data(save_path,segments_final,final_embeddings,person_num)
		else:
			final_embeddings = np.vstack((np.array(id_embeddings), np.array(segments_embeddings)))
			segments_final =segments
			id_final = single_inference_data(save_path,segments_final,final_embeddings,person_num)



	np.savetxt(save_path+'final_segments_pyannote_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final



def voice2id_powerset_msdwild(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	segments_path = save_path+'powerset_segments.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<200:
			segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6]:
		# 读取说话分数矩阵
		# file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

		array = read_2d_array_from_file(file_path)
		#print(np.shape(array))
		person_num = 0   #由于一般msdwild没有头戴者
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>50:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				person_num = person_num+1
		print("predict_person_num:",person_num)

 

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
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<200:
				no_segments.remove([segment[0],segment[1]])

		
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
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 

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
				id_final = voice2id_final + no_final_id
				segments_final = segments + no_final_segments
			else:
				id_final = voice2id_final 
				segments_final = segments
		else:
			id_final = voice2id_final
			segments_final = segments 




	np.savetxt(save_path+'final_segments_pyannote_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_pyannote_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final


#计算每个类中最小相似度
def cal_class_min_sim(similarity_matrix,labels):
	m = len(similarity_matrix[0])
	class_sims = [[] for _ in range(m)]
	i = 0
	for prince in similarity_matrix:
			class_sims[labels[i]].append(prince[labels[i]])
			i=i+1
	print(class_sims)

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
	print(class_sims)
	min_sim = []
	for class_sim in class_sims:
			min_sim.append(np.mean(class_sim))
	return min_sim

#实验一，只声纹平均    
#实验二，聚类数辅助  +声纹平均 
#实验三，去噪+平均声纹  
#实验四，啥也不加，直接聚类 
#实验五，聚类数辅助    
#实验六，去噪+声纹平均+聚类数辅助   √
#实验七，去噪+聚类数辅助

#在原本方法基础上，改进声音增强策略部分
def voice2id_pyannote2_new_enhance(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#audio_path = '/home/rx/audio-visual/active-speaker-detection/vad/audios/'+str(file_no)+'.wav'

	#segments_path = save_path+'segments_pyannote3.1.txt'
	#segments_path = save_path+'pyannote_Powerset.txt'
	segments_path = save_path+'segments_Powerset_Ego4d.txt'
	#segments_path = save_path+'pyannote_Powerset_pretrained.txt'
	#segments_path = save_path+'powerset_segments.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	# my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	# for segment in my_segments:
	# 	if segment[1]-segment[0]<200:
	# 		segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6,7]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		#file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

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
		print(person_num)

 

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [3,6,7]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav_powerset/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav_powerset/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		clean_segments = np.loadtxt(save_path+'clean_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
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
	if num_test in [1,2,3,6,7]:
		id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值



	if num_test in [1,2,3,6,7]:
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
	if num_test in [3,6,7]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
			sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
			save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_pyannote_study"+str(name)+".txt")

			#计算每个类中的与其平均声纹向量最小的相似度
			# class_min_sim = cal_class_min_sim(similarity_matrix,labels)
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
				if my_max>=class_mean_sim[index]:  #如果与当前平均声纹相似度大于等于该类中相似度最小的，则加入。
					print("class_mean_sim:",class_mean_sim[index])
				#if my_max>0.675:
					no_final_segments.append(no_segments[num])
					no_final_id.append(index)
				num = num+1		
	if num_test in [3,6]:	
		id_final = voice2id_final + no_final_id
		segments_final = segments + no_final_segments
	if num_test in [1,2]:
		id_final = voice2id_final 
		segments_final = segments

	if num_test==7:
		if len(no_segments)>0:
			id_final = labels.tolist() + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = labels.tolist()
			segments_final = segments


	np.savetxt(save_path+'final_segments_powset_new_enhance_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_powset_new_enhance_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final


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

from spectral_cluster import cluster,cluster_probability

#计算每个类中平均值
def embeddings_mean(segments_embeddings,labels):
	m = max(labels)+1
	print("m:",m)
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

#在原本方法基础上，改进声音增强策略部分,并改进将视觉信息添加进行聚类方法
# def voice2id_pyannote2_new_enhance_visual(save_path,num_test,ans,model):
def voice2id_pyannote2_new_enhance_visual(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#audio_path = '/home/rx/audio-visual/active-speaker-detection/vad/audios/'+str(file_no)+'.wav'
	#segments_path = save_path+'segments_pyannote3.1.txt'
	#segments_path = save_path+'pyannote_Powerset.txt'
	segments_path = save_path+'segments_Powerset_Ego4d.txt'
	#segments_path = save_path+'pyannote_Powerset_pretrained.txt'


	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<200:
			segments.remove([segment[0],segment[1]])

	person_num=None

	if num_test in [2,5,6,7]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		#file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径

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
		print(person_num)

		speaker_vector = cal_speaker_vector(array,segments)

	# if num_test==4:
	# 	person_num = int(np.loadtxt(save_path+'ground_truth_person_num.txt'))
	# 	print('ground_truth:',person_num)
	# 	num_test = 2

	total_range = [0,300000]

	if num_test in [3,6,7]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav_powerset/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav_powerset/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		#clean_segments = np.loadtxt(save_path+'clean_segments.txt')
		clean_segments = np.loadtxt(save_path+'clean_nosplit_segments.txt')  #注：这里是搞错了，其实就是powerset版本的split


		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<200:
				no_segments.remove([segment[0],segment[1]])

		no_speaker_vector = cal_speaker_vector(array,no_segments)

	#计算每个声音片段的声纹向量与说话分数向量并将其合并,然后进行谱聚类
	segments_embeddings = VocalPrint_embeddings(audio_path,segments) 
	segments_embeddings = np.concatenate((segments_embeddings, speaker_vector), axis=1)  #合并
	print("segments_embeddings:",segments_embeddings.shape)
	labels = cluster(segments_embeddings,num_spks=person_num)
	print(labels)
	

	# labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	if num_test in [4,5]:
		id_final = labels
		segments_final = segments

	if num_test in [1,3]:
		person_num = max(labels)+1
	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	if num_test in [1,2,3,6,7]:
		# id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
		id_embeddings = embeddings_mean(segments_embeddings,labels)
		print("id_embeddings:",id_embeddings.shape)
	

	if num_test in [1,2,3,6,7]:
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
	if num_test in [3,6,7]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			sub_embeddings = np.concatenate((sub_embeddings, no_speaker_vector), axis=1)  #合并
			#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
			sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
			save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_pyannote_study"+str(name)+".txt")

			#计算每个类中的与其平均声纹向量最小的相似度
			class_min_sim = cal_class_min_sim(similarity_matrix,labels)

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
				if my_max>=class_min_sim[index]:  #如果与当前平均声纹相似度大于等于该类中相似度最小的，则加入。
				#if my_max>0.675:
					no_final_segments.append(no_segments[num])
					no_final_id.append(index)
				num = num+1		
	if num_test in [3,6]:	
		id_final = voice2id_final + no_final_id
		segments_final = segments + no_final_segments
	if num_test in [1,2]:
		id_final = voice2id_final 
		segments_final = segments

	if num_test==7:
		if len(no_segments)>0:
			id_final = labels.tolist() + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = labels.tolist()
			segments_final = segments


	np.savetxt(save_path+'final_segments_powset_new_enhance_visual3_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_powset_new_enhance_visual3_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final



#实验一，啥也不加   
#实验二，只增强
#实验三，只聚类辅助
#实验四，只视听向量
#实验五，增强+聚类辅助   
#实验六，增强+视听向量
#实验七，聚类数辅助+视听向量

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def score_to_probability(score):
	# 将分数映射到概率范围
	if score<=0:
		return 0
	else:
		probability = score+0.7
	return probability

#在原本方法基础上，改进声音增强策略部分,并改进将视觉信息添加进行聚类方法,再加上概率视觉辅助聚类数。
# def voice2id_pyannote2_new_enhance_visual_probability(save_path,num_test,ans,model):
def voice2id_pyannote2_new_enhance_visual_probability(save_path,num_test):

	name=num_test
	# 视频文件路径
	audio_path = save_path+'audio.wav' # 指定的音频文件路径
	#audio_path = '/home/rx/audio-visual/active-speaker-detection/vad/audios/'+str(file_no)+'.wav'
	#segments_path = save_path+'segments_pyannote3.1.txt'
	#segments_path = save_path+'pyannote_Powerset.txt'
	# segments_path = save_path+'segments_Powerset_Ego4d.txt'
	segments_path = save_path+'pyannote_Powerset_pretrained.txt'

	segments = []
	with open(segments_path,'r') as f:
		for line in f:
			# 去除行末的换行符并按分隔符分割
			row = line.strip().split(' ')
			segments.append([int(row[0]),int(row[1])])
	import copy
	#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
	my_segments = copy.deepcopy(segments)  #这里需要深度拷贝
	for segment in my_segments:
		if segment[1]-segment[0]<200:
			segments.remove([segment[0],segment[1]])

	person_num=None


	#聚类数辅助
	if num_test in [3,5,7]:
		# 读取说话分数矩阵
		file_path = save_path+'speaker_global_EGO4d.txt'  # 文件路径
		#file_path = save_path+'speaker_global_Ego4d.txt'  # 文件路径
		array = read_2d_array_from_file(file_path)
		persons_num = len(array)   #检测到的总人数
		#print(np.shape(array))
		person_scores=[]
		person_score = []
		person_num = 1
		for row in array:
			count = 0 
			for item in row:
				if float(item)!=0:
					count = count+1
			if count>100:   #这里计算当前id总共被追踪到的帧数，小于100的默认是全局追踪遗漏（一般是一晃而过的）的，不计入总人数
				# person_num = person_num+1
				person_scores.append(score_to_probability(max(row)))

		# print(person_num)
		speakers_probabilities = calculate_probabilities(person_scores)
		num_spks_probabilitys = speakers_probabilities
		speakers_num = find_max_index(speakers_probabilities) +1
		person_num = speakers_num
		print("visual:",person_num)


		speaker_vector = cal_speaker_vector(array,segments)

	total_range = [0,300000]

	#增强
	if num_test in [3,6,7]:
		#no_segments = expand_time_segments(segments, total_range)  #选出没有被检测到人声的语音段
		split_path = save_path+'split_wav_powerset/'   #用于存放被分段的音频文件
		clean_path = save_path+'clean_wav_powerset/'   #用于存放被音频增强（去噪）的音频文件

		#已经跑过一遍，这边暂时先注释
		#speech_enhancement.Enhance(audio_path,segments,split_path,clean_path,total_range,'split',ans)
		clean_audio_path = clean_path+'clean_audio.wav'   #分割去噪后的音频文件路径

		#对去噪后的音频进行重新语音活动检测，这里检测过了，就不用检测了
		#clean_segments = Fsmn_VAD(clean_audio_path,save_path+'clean_nosplit_segments.txt',model)
		# clean_segments = np.loadtxt(save_path+'clean_segments.txt')
		clean_segments = np.loadtxt(save_path+'clean_nosplit_segments.txt')

		#找到clean_segments中与segments没有交集的区间，即通过去噪重新检测到的语音段
		no_segments = find_non_intersecting_segments(segments,clean_segments.tolist())

		#去掉过短的segments：否则在计算每个segment的时候会因为过短而出现报错
		my_no_segments = copy.deepcopy(no_segments)  #这里需要深度拷贝
		for segment in my_no_segments:
			if segment[1]-segment[0]<200:
				no_segments.remove([segment[0],segment[1]])
		no_speaker_vector = cal_speaker_vector(array,no_segments)
	
	#计算每个声音片段的声纹向量与说话分数向量并将其合并,然后进行谱聚类
	segments_embeddings = VocalPrint_embeddings(audio_path,segments) 
	segments_embeddings = np.concatenate((segments_embeddings, speaker_vector), axis=1)  #合并
	print("segments_embeddings:",segments_embeddings.shape)
	if num_test in [2,5,6,7]:
	# labels = cluster(segments_embeddings,num_spks=person_num)
		labels,person_num = cluster_probability(segments_embeddings,num_spks_probabilitys=num_spks_probabilitys,rate=0.8)
	else:
		labels = cluster(segments_embeddings,num_spks=person_num)

	print(labels)
	

	# labels ,segments_embeddings= VocalPrint_Detect_spk_num(audio_path,segments,person_num)  #得到每个语音段对应的id

	if num_test in [4,5]:
		id_final = labels
		segments_final = segments

	if num_test in [1,3]:
		person_num = max(labels)+1
	# print("my_labels:")
	# print(labels)
	#对划分好同一id的语音段合并进行声纹特征提取，得到每个人的声纹特征
	#id_embeddings = VocalPrint_embeddings_id_wear(audio_path,segments,labels,person_num)    #这里是将同一个人的音频合并提取声纹向量
	if num_test in [1,2,3,6,7]:
		# id_embeddings = VocalPrint_embeddings_id_wear_mean(audio_path,segments,labels,person_num)   #这里是将同一个人的音频分别提取声纹向量，再求平均值
		id_embeddings = embeddings_mean(segments_embeddings,labels)
		print("id_embeddings:",id_embeddings.shape)
	

	if num_test in [1,2,3,6,7]:
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
	if num_test in [3,6,7]:
		# no_labels ,no_segments_embeddings= VocalPrint_Detect_spk_num(audio_path,no_segments,None)  #得到每个语音段对应的id
		if len(no_segments)>0:
			sub_embeddings = VocalPrint_embeddings(audio_path,no_segments) 
			sub_embeddings = np.concatenate((sub_embeddings, no_speaker_vector), axis=1)  #合并
			#将这些子段声纹向量和说话人代表声纹向量进行相似度比对，大于0.65的将其归为说话语音段
			sub_similarity_matrix=cosine_similarity(sub_embeddings,id_embeddings)
			save_matrix_to_txt(sub_similarity_matrix,save_path+"voiceprint_id_similarity_no_pyannote_study"+str(name)+".txt")

			#计算每个类中的与其平均声纹向量最小的相似度
			class_min_sim = cal_class_min_sim(similarity_matrix,labels)

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
				if my_max>=class_min_sim[index]:  #如果与当前平均声纹相似度大于等于该类中相似度最小的，则加入。
				#if my_max>0.675:
					no_final_segments.append(no_segments[num])
					no_final_id.append(index)
				num = num+1		
	if num_test in [3,6]:	
		id_final = voice2id_final + no_final_id
		segments_final = segments + no_final_segments
	if num_test in [1,2]:
		id_final = voice2id_final 
		segments_final = segments

	if num_test==7:
		if len(no_segments)>0:
			id_final = labels.tolist() + no_final_id
			segments_final = segments + no_final_segments
		else:
			id_final = labels.tolist()
			segments_final = segments


	np.savetxt(save_path+'final_segments_powset_pretrained_new_enhance_visual_probs_study'+str(name)+'_segments.txt',segments_final,fmt='%d')
	#np.savetxt(save_path+'segments.txt',segments,)
	np.savetxt(save_path+'id_segments_powset_pretrained_new_enhance_visual_probs_study'+str(name)+'.txt',id_final,fmt='%d')

	#print("id_segments_pyannote3.1:",id_final)


	return id_final,segments_final
