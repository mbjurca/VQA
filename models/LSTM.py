import torch
from torch import nn
import math


class LSTMLayer(nn.Module):

    def __init__(self, input_size, hidden_size, device):

        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        # f_t: forget gate weights
        self.w_f = nn.Linear(input_size+hidden_size, hidden_size)

        # i_t: input gate weights
        self.w_i = nn.Linear(input_size+hidden_size, hidden_size)
        
        # c_t: new memory
        self.w_c = nn.Linear(input_size+hidden_size, hidden_size)

        # o_t: output gate weights
        self.w_o = nn.Linear(input_size+hidden_size, hidden_size)

        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)

        h_t_list = []
        c_t_list = []

        for t in range(sequence_length):
            # get the input at the current timestep
            x_t = x[:, t, :]

            # run the LSTM cell
            common_input = torch.cat([x_t,h_t], dim=1)

            # forget gate
            f_t = self.w_f(common_input)
            f_t = self.sigmoid(f_t)

            # input gate 
            i_t = self.w_i(common_input)
            i_t = self.sigmoid(i_t)

            new_c = self.w_c(common_input)
            new_c = self.tanh(new_c)

            # update cell state
            c_t = f_t * c_t + i_t * new_c

            # output gate
            o_t = self.w_o(common_input)
            o_t = self.sigmoid(o_t)

            # update the hidden state
            h_t = o_t * self.tanh(c_t)

            # save the hidden states and cell states
            h_t_list.append(h_t.unsqueeze(dim=1))
            c_t_list.append(c_t.unsqueeze(dim=1))

        h_t_list = torch.cat(h_t_list, dim=1) # batch_size, sequence_length, hidden_size
        c_t_list = torch.cat(c_t_list, dim=1) # batch_size, sequence_length, hidden_size

        return h_t_list, c_t_list


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTM, self).__init__()

        self.lstm_layer = LSTMLayer(input_size=input_size, hidden_size=hidden_size, device = device)

    def forward(self, x):
        h_t, _ = self.lstm_layer(x)

        return h_t


class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(DeepLSTM, self).__init__()

        # LSTM layers
        self.lstm_layer1 = LSTMLayer(input_size, hidden_size, device=device)
        self.lstm_layer2 = LSTMLayer(hidden_size, hidden_size, device=device)

        # fully-connected layer
        self.fc = nn.Linear(2 * 2 * hidden_size, output_size)

        # tanh activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        # LSTM layer 1
        h_t_1, c_t_1 = self.lstm_layer1(x)

        # LSTM layer 2
        h_t_2, c_t_2 = self.lstm_layer2(h_t_1)

        # concatenate cell states and hidden states from each LSTM layer
        # when using SLM
        #concatenated_embedding = torch.cat((h_t_1, c_t_1, h_t_2, c_t_2), dim=2)
        # without SLM        
        concatenated_embedding = torch.cat([h_t_1[:,-1,:], c_t_1[:,-1,:], h_t_2[:,-1,:], c_t_2[:,-1,:]], dim=1)

        # fully-connected layer + tanh non-linearity
        output = self.tanh(self.fc(concatenated_embedding))

        return output
    
