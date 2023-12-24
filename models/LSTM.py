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
        
        # new memory
        self.w_c = nn.Linear(input_size+hidden_size, hidden_size)

        # output gate weights
        self.w_o = nn.Linear(input_size+hidden_size, hidden_size)

        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)

        hidden_state_list = []
        c_t_list = []

        for t in range(sequence_length):

            x_t = x[:, t, :]

            # run the LSTM cell
            common_input = torch.cat((h_t, x_t), dim=1)

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

            hidden_state_list.append(h_t.unsqueeze(dim=1))
            c_t_list.append(c_t.unsqueeze(dim=1))

        hidden_state_list = torch.cat(hidden_state_list, dim=1) # batch_size, seqence_length, hidden_size
        c_t_list = torch.cat(c_t_list, dim=1) # batch_size, seqence_length, hidden_size

        return hidden_state_list, c_t_list

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(CustomLSTM, self).__init__()

        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, device = device)

    def forward(self, x):
        h_t, c_t = self.lstm(x)
        return h_t, c_t

class LSTMDeep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(LSTMDeep, self).__init__()

        # LSTM layers
        self.lstm_layer1 = CustomLSTM(input_size, hidden_size, device=device)
        self.lstm_layer2 = CustomLSTM(hidden_size, hidden_size, device=device)

        # Fully-connected layer
        self.fc = nn.Linear(2 * 2 * hidden_size, output_size)

        # Tanh activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        # LSTM layer 1
        h1, c1 = self.lstm_layer1(x)

        # LSTM layer 2
        h2, c2 = self.lstm_layer2(h1)

        # Concatenate last cell state and last hidden state from each LSTM layer
        concatenated_embedding = torch.cat([h1[:,-1,:], c1[:,-1,:], h2[:,-1,:], c2[:,-1,:]], dim=1)

        # Fully-connected layer + tanh non-linearity
        output = self.tanh(self.fc(concatenated_embedding))

        #print(output.shape)

        return output        