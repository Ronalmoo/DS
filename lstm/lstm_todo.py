import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        # to do
        pass
        # Create cell state
        # 새로운 input과 이전 스텝의 hidden state의 선형 연산 구현
        # 이전 스텝의 hidden state에서 다음 hidden state로 가는 선형 연산
        # Xavier initialization
        # 실패 이유 hidden 이 각각의 게이트에 대해 연산이 이뤄지므로 게이트 갯수(4)만큼 곱해야 함
        
    def reset parameters(self):
        # to do
        pass
    
    def forward(self, x, hidden):
        # to do
        pass
        # hidden (torch.zeros(), torch.zeros()를 각각 h, c에 할당)
        # input_gate, forget_gate, cell_gate, output_gate 구현
        # forget_gate = sigmoid(W_f * [h_t-1, x_t] + b_f)
        # input_gate = sigmoid(W_i * [h_t-1, x_t] + b_i)
        # cell_gate = tanh(W_c * [h_t-1, x_t] + b_c)
        # 현재 시점의 히든과 셀을 hy, cy로 하자. 이전 시점 변수명이 어렵..
        # cy = f_t * c_t-1 + i_t * c_tilde_t
        # o_t = sigmoid(W_o * [h_t-1, x_t] + b_o)
        # hy = o_t * tanh(cy)
        