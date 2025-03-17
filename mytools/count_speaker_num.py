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



# # 示例用法
# probabilities = [0.3, 0.5, 0.2]  # 3 个人的说话概率
# prob_result = calculate_probabilities(probabilities)
# for i, prob in enumerate(prob_result):
#     if i == 0:
#         continue  # 忽略0个人说话的情况
#     print(f"{i} 个人说话的概率: {prob:.4f}")

# import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # 转换为概率
# probabilities = sigmoid(0.4)
# print(probabilities)
