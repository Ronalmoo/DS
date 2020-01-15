import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch import Tensor
import torch.nn.functional as F
import math


class LSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initializer
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        box = self.x2h(x) + self.h2h(hx)
        box = box.squeeze()
        input_gate, forget_gate, cell_gate, output_gate = box.chunk(4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        cy = torch.mul(cx, forget_gate) + torch.mul(input_gate, cell_gate)
        hy = torch.mul(output_gate, torch.tanh(cy))

        return (hy, cy)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super().__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        outs = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
        

if __name__ == "__main__":
    input_dim = 28
    hidden_dim = 128
    layer_dim = 1
    output_dim = 2
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    print(model.forward)