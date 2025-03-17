# import torch
# import torch.nn as nn

# from model.Classifier import BGRU
# from model.Encoder import visual_encoder, audio_encoder

# class ASD_Model2(nn.Module):
#     def __init__(self):
#         super(ASD_Model2, self).__init__()
#         self.GRU = BGRU(128)

   

#     def forward_audio_visual_backend(self, x1, x2):  
#         x = x1 + x2 
#         x = self.GRU(x)   
#         x = torch.reshape(x, (-1, 128))
#         return x    


#     def forward(self, audioFeature, visualFeature):
#         audioEmbed = self.forward_audio_frontend(audioFeature)
#         visualEmbed = self.forward_visual_frontend(visualFeature)
#         outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  
#         outsV = self.forward_visual_backend(visualEmbed)
#         return outsAV, outsV

import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden size by 2 due to bidirectional GRU
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state for bidirectional GRU
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, :x.size(1), :])  # Only take the first n rows' output
        print(out.size())
        return out.view(-1, x.size(2))  # Reshape output to have n rows and k columns

# 假设输入矩阵的维度为 (n+v, k)
n = 10
v = 5
k = 20
input_size = k
hidden_size = 10
num_layers = 2
output_size = n * k
model = BiGRUModel(input_size, hidden_size, num_layers, output_size)
input_data = torch.randn(n+v, k)  # 生成随机输入数据
input_data = input_data.unsqueeze(0)  # 添加 batch 维度，维度变为 (1, n+v, k)
output = model(input_data)  # 进行前向传播得到输出
print(output.size())  # 输出: torch.Size([1, 200])
# output = output.view(n, k)  # 重新调整输出的形状为 (n, k)
# print(output.size())  # 输出: torch.Size([10, 20])