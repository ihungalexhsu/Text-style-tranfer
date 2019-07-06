import torch
import numpy as np
import torch.nn  as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class RCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding, rnn_hidden_dim,
                 linear_dim, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(pre_embedding, requires_grad=True)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=rnn_hidden_dim,
                            num_layers=3,
                            dropout=0.2,
                            bidirectional=True,
                            batch_first=True)
        self.dropout=nn.Dropout(0.2)
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(embedding_dim+2*rnn_hidden_dim, linear_dim)
        self.tanh = nn.Tanh()
        # Fully-Connected Layer
        self.fc = nn.Linear(linear_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, ilens):
        embedded = self.embedding(x) # batch x seq_len
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.lstm.flatten_parameters()
        xpack, _ = self.lstm(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True, total_length=total_length)
        input_features = torch.cat([xpad, embedded],2) # batch x seq_len x (emb+2*hidden)
        
        linear_output = self.tanh(self.W(input_features))
        # reshape for max_pool
        linear_output = linear_output.permute(0,2,1)
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        max_out_features = self.dropout(max_out_features)
        final_output = self.fc(max_out_features)
        prediction = final_output.topk(1, dim=1)[1]
        return self.log_softmax(final_output), prediction
