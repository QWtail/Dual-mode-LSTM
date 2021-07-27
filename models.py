'''
Main Models
Author: Wei QIU
Date: 20/07/2021
'''

from utils import *
from basemodel_customized import *

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.n_layers = self.args.n_layers
        self.out_features = self.args.rnn_size
        self.inputLayer = nn.Linear(self.args.input_size, self.out_features)
        self.cell = nn.ModuleDict()
        for i in range(self.n_layers):
            self.cell['u_%d'%i] = LSTMCell(self.out_features, self.out_features,self.dropout)
        self.outputLayer = nn.Linear(self.out_features, self.args.output_size)
        self.input_Ac = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.xavier_uniform_(self.inputLayer.weight, std=self.args.std_in)

        for i in range(self.n_layers):
            nn.init.xavier_uniform_(self.cell['u_%d'%i].weight_ih)
            nn.init.xavier_uniform_(self.cell['u_%d'%i].weight_hh, gain=0.001)

            nn.init.constant_(self.cell['u_%d'%i].bias_ih, 0.0)
            nn.init.constant_(self.cell['u_%d'%i].bias_hh, 0.0)
            n = self.cell['u_0'].bias_ih.size(0)
            nn.init.constant_(self.cell['u_%d'%i].bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.xavier_uniform_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs):

        raw_input = inputs

        num_stud = raw_input.shape[1]
        para_ = dict()
        for i in range(self.n_layers):
            para_['hidden_states_%d'%i] = torch.zeros(self.args.input_size, self.out_features, requires_grad=True)
            para_['cell_states_%d'%i] = torch.zeros(self.args.input_size, self.out_features, requires_grad=True)
        
        for sem_ in range(self.args.seq_length):
            input_embedded = self.input_Ac(self.inputLayer(raw_input[sem_]))
            for i in range(self.n_layers):
                if i == 0:
                    lstm_state, gates = self.cell['u_%d'%i].forward(input_embedded, (para_['hidden_states_%d'%i], para_['cell_states_%d'%i]))
                else:
                    lstm_state = self.cell['u_%d'%i].forward(para_['hidden_states_%d' % (i-1)],(para_['hidden_states_%d' % i], para_['cell_states_%d' % i]))
                _, para_['hidden_states_%d'%i], para_['cell_states_%d'%i] = lstm_state

            _, para_['hidden_states_%d'%(self.n_layers-1)], para_['cell_states_%d'%(self.n_layers-1)] = lstm_state

        predictions_ = self.input_Ac(self.outputLayer(para_['hidden_states_%d'%(self.n_layers-1)]))

        return predictions_


class LSTMresidualgated(nn.Module):
    def __init__(self, args):
        super(LSTMresidualgated, self).__init__()
        self.args = args
        
        self.n_layers = self.args.n_layers
        self.out_features = self.args.rnn_size
        self.inputLayer = nn.Linear(self.args.input_size, self.out_features)
        self.cell = nn.ModuleDict()
        for i in range(self.n_layers):
            self.cell['u_%d'%i] = LSTMCell_residualgated(self.args.input_size, self.out_features,self.dropout)
        self.outputLayer = nn.Linear(self.out_features, self.args.output_size)
        self.input_Ac = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for i in range(self.n_layers):
            nn.init.xavier_uniform_(self.cell['u_%d'%i].weight_ih)
            nn.init.xavier_uniform_(self.cell['u_%d'%i].weight_hh, gain=0.001)

            nn.init.constant_(self.cell['u_%d'%i].bias_ih, 0.0)
            nn.init.constant_(self.cell['u_%d'%i].bias_hh, 0.0)
            n = self.cell['u_0'].bias_ih.size(0)
            nn.init.constant_(self.cell['u_%d'%i].bias_ih[n // 4:n // 2], 1.0)

        nn.init.xavier_uniform_(self.if_cell.alpha)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.xavier_uniform_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs):

        raw_input = inputs
        
        num_stud = raw_input.shape[1]
        para_ = dict()
        for i in range(self.n_layers):
            para_['hidden_states_%d'%i] = torch.zeros(self.args.input_size, self.out_features, requires_grad=True)
            para_['cell_states_%d'%i] = torch.zeros(self.args.input_size, self.out_features, requires_grad=True)
        
        for sem_ in range(self.args.seq_length):
            input_embedded = self.input_Ac(self.inputLayer(raw_input[sem_]))
            if sem_>0:
                resid = raw_input[sem_] - raw_input[sem_-1]
            else:
                resid = torch.zeros(raw_input[sem_].size())
            for i in range(self.n_layers):
                if i == 0:
                    lstm_state, gates = self.cell['u_%d'%i].forward(input_embedded,resid, (para_['hidden_states_%d'%i], para_['cell_states_%d'%i]))
                    ingate, forgetgate = gates
                    it[str(sem_)] = ingate
                    ft[str(sem_)] = forgetgate
                else:
                    lstm_state = self.cell['u_%d'%i].forward(para_['hidden_states_%d' % (i-1)],(para_['hidden_states_%d' % i], para_['cell_states_%d' % i]))
                _, para_['hidden_states_%d'%i], para_['cell_states_%d'%i] = lstm_state

            _, para_['hidden_states_%d'%(self.n_layers-1)], para_['cell_states_%d'%(self.n_layers-1)] = lstm_state

        predictions_ = self.input_Ac(self.outputLayer(para_['hidden_states_%d'%(self.n_layers-1)]))

        predictions_nei = 0
        return predictions_
