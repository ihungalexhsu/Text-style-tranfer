import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import cc
from torch.distributions.categorical import Categorical
import random

class E2E(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, enc_layers, 
                 enc_dropout, bidirectional, dec_hidden_dim, dec_dropout, bos, 
                 eos, pad, n_styles, style_emb_dim, att_dim, pre_embedding=None, update_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(pre_embedding, requires_grad=update_embedding)
        
        self.encoder = Encoder(embedding_dim, enc_hidden_dim, enc_layers,
                               enc_dropout, bidirectional)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_hidden_dim,
                               dec_dropout, bos, eos, pad, 
                               enc_hidden_dim*2 if bidirectional else enc_hidden_dim,
                               n_styles, style_emb_dim, att_dim)

    def forward(self, xs, ilens, styles, dec_input=None,
                max_dec_timesteps=15, sample=False):
        emb = self.embedding(xs)
        enc_outputs, enc_lens = self.encoder(emb, ilens)
        _, recon_log_probs, predictions, attns =\
            self.decoder(enc_outputs, enc_lens, styles, dec_input,
                         max_dec_timesteps, sample)
        
        return recon_log_probs, predictions, attns

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, 
                 dropout_rate, bidirectional=True):
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout_rate, 
                              batch_first=True)
        self.dropout_rate = dropout_rate
                          
    def forward(self, embedded, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            embedded = embedded[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.encoder.flatten_parameters()
        xpack, _ = self.encoder(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True,
                                          total_length=total_length)
        if need_sort:
            xpad = xpad[unsort_idx]
            ilens = ilens[unsort_idx]
        return xpad, cc(ilens)

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, 
                 dropout_rate, bos, eos, pad, enc_out_dim,
                 n_styles, style_emb_dim, att_dim):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.style_embedding = nn.Embedding(n_styles, style_emb_dim)
        nn.init.orthogonal_(self.style_embedding.weight)
        self.hidden_dim = hidden_dim
        self.enc_out_dim = enc_out_dim
        self.attention = dotAttn(hidden_dim, enc_out_dim, att_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.GRU = nn.GRU(embedding_dim+style_emb_dim, hidden_dim, num_layers=2, 
                          batch_first=True, dropout=0.1)
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(enc_out_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward_step(self, emb, style_emb, dec_h, enc_output, enc_len):
        cell_inp = torch.cat([style_emb, emb], dim=-1).unsqueeze(1)
        cell_inp = self.dropout(cell_inp)
        dec_out, dec_h = self.GRU(cell_inp, dec_h)
        dec_out = self.dropout(dec_out)
        context, attn = self.attention(dec_out, enc_output, enc_output, enc_len)
        out_layer_inp = torch.cat([context, dec_out], dim=-1)
        logit = self.output_layer(out_layer_inp)
        return logit, dec_h, context, attn

    def forward(self, enc_output, enc_len, styles, dec_input=None, 
                max_dec_timesteps=15, sample=False):
        
        batch_size = enc_output.size(0)
        style_emb = self.style_embedding(styles) # batch x style_emb
        if dec_input is not None:
            # training stage
            # dec_input would be a tuple (dec_in, dec_out)
            pad_dec_in = dec_input[0]
            pad_dec_out = dec_input[1]
            seq_len = pad_dec_in.size(1)
            dec_input_emb = self.embedding(pad_dec_in) # batch x seq_len x emb_dim
            style_emb_seq = style_emb.unsqueeze(1).expand(-1, seq_len, -1)
            cell_inp = torch.cat([style_emb_seq, dec_input_emb], dim=-1)
            cell_inp = self.dropout(cell_inp)
            dec_out, _ = self.GRU(cell_inp)
            dec_out = self.dropout(dec_out)
            context, attns = self.attention(dec_out, enc_output, enc_output, enc_len)
            out_layer_inp = torch.cat([context, dec_out], dim=-1)
            logits = self.output_layer(out_layer_inp)
            prediction = torch.argmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=2)
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=pad_dec_out.unsqueeze(2)).squeeze(2)
        else:
            olength = max_dec_timesteps
            dec_h = None
            logits, prediction, attns = [], [], []
            for t in range(olength):
                if t==0:
                    bos = cc(torch.Tensor([self.bos for _ in range(batch_size)]).type(torch.LongTensor))
                    emb = self.embedding(bos)
                else:
                    emb = self.embedding(prediction[-1])
                    
                logit, dec_h, context, attn = \
                    self.forward_step(emb, style_emb, dec_h, enc_output, enc_len)
                attns.append(attn.squeeze(1))
                logits.append(logit.squeeze(1))
                if not sample:
                    prediction.append(torch.argmax(logit.squeeze(1), dim=-1))
                else:
                    sampled_indices = Categorical(logits=logit.squeeze(1)).sample() 
                    prediction.append(sampled_indices)
            logits = torch.stack(logits, dim=1) # batch x length x output_dim
            log_probs = F.log_softmax(logits, dim=2)
            prediction = torch.stack(prediction, dim=1) # batch x length
            attns = torch.stack(attns, dim=1) # batch x length x enc_len
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=prediction.unsqueeze(2)).squeeze(2)
        return logits, dec_output_log_probs, prediction, attns

class dotAttn(nn.Module):
    def __init__(self, query_dim, key_dim, att_dim):
        '''
        basic setting:
        query_dim is decoder hidden dim
        key_dim is encoder output dim
        att_dim is projected dim
        '''
        super().__init__()
        self.mlp_query = nn.Linear(query_dim, att_dim, bias=True)
        self.mlp_key = nn.Linear(key_dim, att_dim, bias=False)
    
    def forward(self, query, keys, value, key_len=None, scaling=1.0):
        '''
        :param query:
            hidden state of the decoder, in shape (batch, seq_len, dec_dim)
        :param keys:
            encoder output, in shape (batch, enc_maxlen, enc_dim)
        :param key_len:
            encoder output lens, in shape (batch)
        :param value:
            usually encoder output, in shape (batch, enc_maxlen, enc_dim)
        '''
        query = self.mlp_query(query) # [BxLDxQ] -> [BxLDxA]
        keys = self.mlp_key(keys) # [BxLxK] -> [BxLxA]
        energy = torch.bmm(query, keys.transpose(1,2)) # [BxLDxL]
        if key_len is not None:
            mask = []
            for b in range(key_len.size(0)):
                mask.append([0]*key_len[b].item()+[1]*(keys.size(1)-key_len[b].item()))
            mask = cc(torch.ByteTensor(mask).unsqueeze(1)) # [BxL] -> [Bx1xL]
            energy = energy.masked_fill_(mask, -1e10)
        energy = F.softmax(energy * scaling, dim=2) # [BxLDxL]
        context = torch.bmm(energy, value) # [BxLDxV]
        return context, energy
