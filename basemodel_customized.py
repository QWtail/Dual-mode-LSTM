'''
Basic models
Author: Wei QIU
Date: 20/07/2021
'''
import math
import torch
import numpy as np
from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter

class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size,dropout):
        super(LSTMCell, self).__init__(input_size, hidden_size,bias=True,num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_ = nn.Dropout(self.dropout)

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, ht):

        hx, cx = ht
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate_, forgetgate_, cellgate_, outgate_ = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate_)
        forgetgate = torch.sigmoid(forgetgate_)
        cellgate = torch.tanh(cellgate_)
        outgate = torch.sigmoid(outgate_ )
        
        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * torch.tanh(cy)

        cy = self.dropout_(cy)
        hy = self.dropout_(hy)
        
        return [outgate,hy, cy],[ingate,forgetgate]


class LSTMCell_residualgated(RNNCellBase):
    def __init__(self, input_size, hidden_size,dropout):
        super(LSTMCell_residualgated, self).__init__(input_size, hidden_size,bias=True,num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_ = nn.Dropout(self.dropout)

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.weight_ha = Parameter(torch.Tensor(  hidden_size, input_size))
        self.bias_ha = Parameter(torch.Tensor( hidden_size))
        self.weight_hc = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hc = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, resid, ht):

        hx, cx = ht
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)
        residualgate_ = F.linear(resid, self.weight_ha, self.bias_ha) + F.linear(cx, self.weight_hc, self.bias_hc)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate_ )
        residualgate = torch.tanh(residualgate_)
        
        cy = forgetgate * cx + residualgate * ingate * cellgate
        hy = outgate * torch.tanh(cy)

        cy = self.dropout_(cy)
        hy = self.dropout_(hy)
        
        return [outgate,hy, cy],[ingate,forgetgate]
