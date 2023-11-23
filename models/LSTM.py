import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        # forget gate weights
        self.w_f = nn.Linear(input_size+hidden_size, hidden_size)

        # input gate weights
        self.w_i = nn.Linear(input_size+hidden_size, hidden_size)
        self.w_c = nn.Linear(input_size+hidden_size, hidden_size)

        # output gate weights
        self.w_o = nn.Linear(input_size+hidden_size, hidden_size)

        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t_prev = torch.zeros(batch_size, self.hidden_size).to(self.device)

        hidden_state_list = []

        for idx in range(sequence_length):

            x_t = x[:, idx, :]

            common_input = torch.cat((h_t, x_t), dim=1)

            # forget gate
            f_t = self.w_f(common_input)
            f_t = self.sigmoid(f_t)

            # input gate 
            i_t = self.w_i(common_input)
            i_t = self.sigmoid(i_t)

            c_t = self.w_c(common_input)
            c_t = self.tanh(c_t)

            # update cell state
            c_t_prev = c_t_prev * f_t + i_t * c_t

            # output gate
            o_t = self.w_o(common_input)
            o_t = self.sigmoid(o_t)

            # update the hidden state
            h_t = self.tanh(c_t_prev) * o_t

            hidden_state_list.append(h_t.unsqueeze(dim=1))

        output = torch.cat(hidden_state_list, dim=1) # batch_size, seqence_length, hidden_size

        return output

        




