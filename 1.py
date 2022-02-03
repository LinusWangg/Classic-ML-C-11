import torch
import torch.nn as nn
 
rnn = nn.LSTM(input_size=4, hidden_size=6, num_layers=1)
input = torch.randn(3, 2, 4)  # batch=2, seq_len=3, input_size=4
output, (hn, cn) = rnn(input)  # 如果h0和c0未给出，则默认为0
 
print(output.shape, hn.shape, cn.shape)
print(output[-1, :, :])
print(hn[-1, :, :])