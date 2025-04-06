from collections import OrderedDict
import kaldiio
import numpy as np
import scipy.linalg
from sklearn.cluster._kmeans import k_means
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import linkage,fcluster
from cluster_test  import *






def cluster(embeddings, p=.01, num_spks=None, min_num_spks=1, max_num_spks=20):
    # Define utility functions
    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + np.dot(M, M.T))

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)

        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M
 
    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        #print(eig_values)
        #print(np.diff(eig_values[:max_num_spks + 1]))
        num_spks = num_spks if num_spks is not None \
            else np.argmax(np.diff(eig_values[:max_num_spks + 1])) + 1
        num_spks = max(num_spks, min_num_spks)
        # print(num_spks)
        return eig_vectors[:, :num_spks]
        #return eig_vectors

    def kmeans(data):
        k = data.shape[1]
        # centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        centers, labels, _ = k_means(data, k, random_state=None, n_init=10)
        return labels

    #不需要给定类别数的聚类算法
    def perform_dbscan(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return labels
    
    def my_OPTICS(X):
        clustering = OPTICS(min_samples=0.05).fit(X)

        # 提取聚类标签和核心距离
        labels = clustering.labels_
        return labels

    def hierarchical_clustering(similarity_matrix, threshold, method='ward'):
        # 计算层次聚类
        Z = linkage(similarity_matrix, method=method)
        
        # 根据阈值获取聚类结果
        labels = fcluster(Z, threshold, criterion='distance')
    
        return labels
    
    # def similarity_clustering(similarity_matrix, threshold=0.7,max_thresshold=0.8):
    #     num_samples = similarity_matrix.shape[0]
    #     labels = np.arange(num_samples)  # 初始化每个样本的类别标签

    #     for i in range(num_samples):
    #         max=0
    #         index=-1
    #         for j in range(num_samples):
    #             if i!=j and similarity_matrix[i, j] > threshold and similarity_matrix[i, j]>max:
    #                 # 找到当前行最大的相似度
    #                 max=similarity_matrix[i, j]
    #                 index = j  
    #             if i<j and similarity_matrix[i, j] >= max_thresshold:
    #                 labels[j]=labels[i]
    #         if max!=0:
    #             if labels[i]>labels[index]:
    #                 labels[i] = labels[index]
    #             else :
    #                 labels[index] = labels[i]
       
    #     return labels

    def save_matrix_to_txt(matrix, filename):
        np.savetxt(filename, matrix, delimiter='\t')

    # # Fallback for trivial cases
    # if len(embeddings) <= 2:
    #     return [0] * len(embeddings)
    if len(embeddings) <= 1:
        return np.array([0] * len(embeddings))

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))
    save_matrix_to_txt(similarity_matrix,"./similarity.txt")
    #labels=hierarchical_clustering(similarity_matrix,0.70)
    #labels=similarity_clustering(similarity_matrix, threshold=0.7)
    # Prune matrix with p interval
    pruned_similarity_matrix = prune(similarity_matrix, p)
    # Compute Laplacian
    laplacian_matrix = laplacian(pruned_similarity_matrix)
    # Compute spectral embeddings
    spectral_embeddings = spectral(laplacian_matrix, num_spks,min_num_spks, max_num_spks)
    # #print(spectral_embeddings)
    # #print(len(spectral_embeddings[0]))
    # # Assign class labels
    # #labels = perform_dbscan(spectral_embeddings,0.125,1)
    labels = kmeans(spectral_embeddings)
    # #print(embeddings)
    # #print(len(embeddings))

    # # labels = my_OPTICS(similarity_matrix)
    # print(labels)

    return labels

def read_emb(scp):

    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
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

    return subsegs_list, embeddings_list



def cluster_probability(embeddings, p=.01, num_spks_probabilitys=None, min_num_spks=1, max_num_spks=20,rate=0.8):
    # Define utility functions
    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + np.dot(M, M.T))

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)

        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M

    def softmax(scores):
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities
    def find_max_index(lst):
        max_value = max(lst)
        max_index = max(i for i, v in enumerate(lst) if v == max_value)
        return max_index
    def spectral(M, num_spks_probabilitys, min_num_spks, max_num_spks,rate):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        diff = softmax(np.diff(eig_values[:max_num_spks]))

        # print("diff:",diff)
        
        if num_spks_probabilitys is  None:
            num_spks = np.argmax(np.diff(eig_values[:max_num_spks + 1])) + 1
        else:
            if isinstance(num_spks_probabilitys, float):
                diff[0] = rate*num_spks_probabilitys + (1-rate)*diff[0]
            else:
                if len(num_spks_probabilitys)<=len(diff):
                    for i,num_spks_probability in enumerate(num_spks_probabilitys):
                        diff[i] = rate*num_spks_probability + (1-rate)*diff[i]
                else:
                    for i,sub_diff in enumerate(diff):
                        diff[i] = rate*num_spks_probabilitys[i] + (1-rate)*sub_diff
            
            num_spks = find_max_index(diff)+1
        # print(diff)

        num_spks = max(num_spks, min_num_spks)
        # print(num_spks)
        return eig_vectors[:, :num_spks],num_spks
        #return eig_vectors

    def kmeans(data):
        k = data.shape[1]
        # centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        centers, labels, _ = k_means(data, k, random_state=None, n_init=10)
        return labels


    

    def save_matrix_to_txt(matrix, filename):
        np.savetxt(filename, matrix, delimiter='\t')

    # Fallback for trivial cases
    if len(embeddings) <= 1:
        return np.array([0] * len(embeddings)) ,1

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))
    # save_matrix_to_txt(similarity_matrix,"./similarity.txt")
    # Prune matrix with p interval
    pruned_similarity_matrix = prune(similarity_matrix, p)
    # Compute Laplacian
    laplacian_matrix = laplacian(pruned_similarity_matrix)
    # Compute spectral embeddings
    spectral_embeddings,num_spks = spectral(laplacian_matrix, num_spks_probabilitys,min_num_spks, max_num_spks,rate)
    # # Assign class labels
    # #labels = perform_dbscan(spectral_embeddings,0.125,1)
    labels = kmeans(spectral_embeddings)

    return labels,num_spks



def cluster2(embeddings, p=.01, num_spks=None, min_num_spks=1, max_num_spks=20):
    # Define utility functions
    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + np.dot(M, M.T))

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)

        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M
 
    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        #print(eig_values)
        #print(np.diff(eig_values[:max_num_spks + 1]))
        num_spks = num_spks if num_spks is not None \
            else np.argmax(np.diff(eig_values[:max_num_spks + 1])) + 1
        num_spks = max(num_spks, min_num_spks)
        # print(num_spks)
        return eig_vectors[:, :num_spks]
        #return eig_vectors

    def kmeans(data):
        k = data.shape[1]
        # centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        centers, labels, _ = k_means(data, k, random_state=None, n_init=10)
        return labels
    # # Fallback for trivial cases
    # if len(embeddings) <= 2:
    #     return [0] * len(embeddings)
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))
    # Prune matrix with p interval
    pruned_similarity_matrix = prune(similarity_matrix, p)
    # Compute Laplacian
    laplacian_matrix = laplacian(pruned_similarity_matrix)
    # Compute spectral embeddings
    spectral_embeddings = spectral(laplacian_matrix, num_spks,min_num_spks, max_num_spks)
    labels = kmeans(spectral_embeddings)
    return labels