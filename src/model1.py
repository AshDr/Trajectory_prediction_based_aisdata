import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

DEVICE = 'cpu'


class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size, embed_size, num_layers, dropout=0):
        super().__init__()
        self.in_features = in_features
        self.hidden_siz = hidden_size
        self.embedding = nn.Embedding(in_features, embed_size)
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers)

    def forword(self, enc_input):
        seq_len, batch_size, embedding_size = enc_input.size()
        h_0 = torch.rand(1, batch_size, self.hidden_siz)
        c_0 = torch.rand(1, batch_size, self.hidden_siz)
        output, state = self.rnn(enc_input, (h_0, c_0))
        return output, state


class Decoder(nn.Module):
    def __init__(self, in_features, hidden_size, dropout=0):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.loss = nn.CrossEntropyLoss
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=dropout, num_layers=1)

    def forward(self, enc_output, dec_input):
        output, _ = self.rnn(dec_input, enc_output)
        return output


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, in_features, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features
        self.hidden_size = hidden_size
        # 还要有lostfun

    def forward(self, enc_input, dec_input, dec_output):
        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        dec_input = dec_input.permute(1, 0, 2)
        _,enc_output = self.encoder(enc_input)
        output,_ = self.decoder(enc_output,dec_input)
        # 然后对output做评估
        #...
        loss = 0
        for i in range(len(output)):
            #loss += self.loss(x,y)
        return output,loss
    