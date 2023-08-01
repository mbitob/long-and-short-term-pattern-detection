import torch
import torch.nn as nn
import math
import copy

class AttentionHead(torch.nn.Module):
    def __init__(self, in_head_dim, out_head_dim, kernel_size=3, padding = 'same'):
        super(AttentionHead, self).__init__()
        self.in_head_dim = in_head_dim
        self.out_head_dim = out_head_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.querry_func = nn.Conv1d(in_channels=in_head_dim, out_channels = out_head_dim, kernel_size = kernel_size, padding=padding)
        self.key_func = nn.Conv1d(in_channels=in_head_dim, out_channels = out_head_dim, kernel_size = kernel_size, padding=padding)
        self.value_func = nn.Conv1d(in_channels=in_head_dim, out_channels = out_head_dim, kernel_size = kernel_size, padding=padding)
        self.attention_matrix = None
        self.attention_embedding = None

        #self.attention = None

    def forward(self, x):
        # x --> (N, L) batch, sequence length 
        querry = torch.transpose(self.querry_func(x), 1,2)  #(N, L, out_head_dim)
        key = torch.transpose(self.key_func(x), 1,2)       #(N, L, out_head_dim)
        value = torch.transpose(self.value_func(x), 1,2)    #(N, L, out_head_dim)

        #print('querry_shape:', querry.shape)
        #print('key_shape:', key.shape)
        #print('value_shape:', value.shape)

        scaled_querry_key = torch.matmul(querry, torch.transpose(key, 1,2)) / math.sqrt(self.out_head_dim) 

        softmax_scaled_querry_key = nn.functional.softmax(scaled_querry_key, dim = 2)
        self.attention_matrix = softmax_scaled_querry_key # (N, L, L)

        #print('attention_matrix', self.attention_matrix.shape)

        self.attention_embedding = torch.matmul(softmax_scaled_querry_key, value) 
        return self.attention_embedding # (N, L, Eq=out_channels) batch, sequence_length, embedding_dim of one head


class MultiAttentionHead(torch.nn.Module):
    def __init__(self, input_dim=1,  hidden_dim_per_head=2, convolution_kernels = (3,9,25)):
        super(MultiAttentionHead, self).__init__()
        self.num_heads = len(convolution_kernels) # Number of kernels determines 
        self.embedding_dim = hidden_dim_per_head * self.num_heads
        self.convolution_kernels = convolution_kernels
        self.input_dim = input_dim
        self.heads = [AttentionHead(input_dim, hidden_dim_per_head, kernel_size = size )  for size in convolution_kernels]

        self.LinTransform = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        multi_head_attention = self.heads[0](x)
        for head in self.heads[1::]:  
            #print(multi_head_attention.shape)
            #out = head(x)
            #print(out.shape)
            multi_head_attention  = torch.cat((multi_head_attention , head(x)), 2)
        multi_head_attention = self.LinTransform(multi_head_attention)
        multi_head_attention = self.Tanh(multi_head_attention)

        return torch.transpose(multi_head_attention, 1,2)