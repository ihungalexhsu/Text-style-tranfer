import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from utils import cc
from utils import pad_list
from utils import _seq_mask
from utils import get_enc_context
from torch.distributions.categorical import Categorical
import random
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

class MaskGen(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        self.BiRNN = nn.GRU(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=bidirectional, batch_first=True)
        self.Output_layer = nn.GRU(hidden_dim*2 if bidirectional else hidden_dim, 
                                   1, num_layers=1, bidirectional=False, 
                                   batch_first=True)
    
    def forward(self, x, ilens):
        total_length = x.size(1)
        x = self.embedding(x)
        xpack = pack_padded_sequence(x, ilens, batch_first=True)
        self.BiRNN.flatten_parameters()
        xpack, _ = self.BiRNN(xpack)
        self.Output_layer.flatten_parameters()
        xpack, _ = self.Output_layer(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True,
                                          total_length=total_length)
        rand = cc(torch.randn(xpad.size())/100)
        xpad = torch.sigmoid((xpad+rand)*10000)
        return xpad, cc(ilens)
'''
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad, hidden_dim, n_layers, 
                 dropout_rate, bidirectional=True, pre_embedding=None, 
                 update_embedding=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        if pre_embedding is not None:
            self.embedding = self.embedding.from_pretrained(pre_embedding, 
                                                            freeze=not(update_embedding))
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout_rate, 
                              batch_first=True)
        self.dropout_rate = dropout_rate
                          
    def forward(self, x, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            x = x[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        embedded = self.embedding(x)
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
'''
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
                 n_styles, style_emb_dim, att_dim,
                 pre_embedding=None, update_embedding=True):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=not(update_embedding))
        else:
            self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.style_embedding = nn.Embedding(n_styles, style_emb_dim)
        nn.init.orthogonal_(self.style_embedding.weight)
        self.hidden_dim = hidden_dim
        self.enc_out_dim = enc_out_dim
        self.attention = dotAttn(hidden_dim, enc_out_dim, att_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.GRUCell = nn.GRUCell(embedding_dim+style_emb_dim+enc_out_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(enc_out_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def zero_state(self, ori_tensor, dim=None):
        '''
        a util function that new a zero tensor at the same shape of (batch, dim)
        '''
        if not dim:
            return ori_tensor.new_zeros(ori_tensor.size(0), self.hidden_dim)
        else:
            return ori_tensor.new_zeros(ori_tensor.size(0), dim)

    def forward_step(self, emb, style_emb, dec_h, context, 
                     enc_output, enc_len, coverage):
        cell_inp = torch.cat([emb, style_emb, context], dim=-1)
        cell_inp = self.dropout(cell_inp)
        dec_h = self.GRUCell(cell_inp, dec_h)
        dec_h = self.dropout(dec_h)
        context, attn = self.attention(dec_h, enc_output, enc_output, 
                                       coverage, enc_len)
        out_layer_inp = torch.cat([context, dec_h], dim=-1)
        out_layer_inp = self.dropout(out_layer_inp)
        logit = self.output_layer(out_layer_inp)
        return logit, dec_h, context, attn

    def forward(self, enc_output, enc_len, styles, dec_input=None, 
                tf_rate=1.0, max_dec_timesteps=15, sample=False):
        batch_size = enc_output.size(0)
        # prepare data input/output
        if dec_input is not None:
            # dec_input would be a tuple (dec_in, dec_out)
            pad_dec_in = dec_input[0]
            pad_dec_out = dec_input[1]
            # get length info
            batch_size, olength = pad_dec_out.size(0), pad_dec_out.size(1)
            # map idx to embedding
            dec_input_emb = self.embedding(pad_dec_in)
        else:
            olength = max_dec_timesteps
        
        # prepare style embedding
        style_emb = self.style_embedding(styles)

        # initialization
        dec_h = self.zero_state(enc_output)
        context = self.zero_state(enc_output, dim=self.enc_out_dim)
        logits, prediction, attns = [], [], []
        
        for t in range(olength):
            if dec_input is not None:
                # teacher forcing
                tf = True if np.random.random_sample() <= tf_rate else False
                if tf or t==0:
                    emb = dec_input_emb[:,t,:]
                else:
                    self.embedding(prediction[-1])
            else:
                if t == 0:
                    bos = cc(torch.Tensor([self.bos for _ in range(batch_size)]).type(torch.LongTensor))
                    emb = self.embedding(bos)
                else:
                    emb = self.embedding(prediction[-1])
            if len(attns)==0:
                coverage = enc_output.new_zeros((enc_output.size(0),enc_output.size(1)))
            else:
                coverage = sum(attns)
            logit, dec_h, context, attn = \
                self.forward_step(emb, style_emb, dec_h, context,
                                  enc_output, enc_len, coverage)
            attns.append(attn.squeeze(1))
            logits.append(logit)
            if not sample:
                prediction.append(torch.argmax(logit, dim=-1))
            else:
                sampled_indices = Categorical(logits=logit).sample() 
                prediction.append(sampled_indices)

        logits = torch.stack(logits, dim=1) # batch x length x output_dim
        log_probs = F.log_softmax(logits, dim=2)
        prediction = torch.stack(prediction, dim=1) # batch x length
        attns = torch.stack(attns, dim=1) # batch x length x enc_len

        # get the log probs of the true label(batch x length)
        if dec_input is not None:
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=pad_dec_out.unsqueeze(2)).squeeze(2)
        else:
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
        self.mlp_coverage = nn.Linear(1, att_dim, bias=False)
        self.mlp_out = nn.Linear(att_dim, 1, bias=False)
    
    def forward(self, query, keys, value, cover, key_len=None, scaling=1.0):
        '''
        :param query:
            previous hidden state of the decoder, in shape (batch, dec_dim)
        :param keys:
            encoder output, in shape (batch, enc_maxlen, enc_dim)
        :param key_len:
            encoder output lens, in shape (batch)
        :param value:
            usually encoder output, in shape (batch, enc_maxlen, enc_dim)
        '''
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        query = self.mlp_query(query) # [Bx1xQ] -> [Bx1xA]
        #keys = self.mlp_key(keys).transpose(1,2) # [BxLxK] -> [BxLxA] -> [BxAxL]
        keys = self.mlp_key(keys) # [BxLxK] -> [BxLxA]
        #energy = torch.bmm(query, keys) # [Bx1xL]
        cover = self.mlp_coverage(cover.unsqueeze(2))
        energy = self.mlp_out(torch.tanh(query+keys+cover)).transpose(1,2) # [BxLxA]
        if key_len is not None:
            mask = []
            for b in range(key_len.size(0)):
                #mask.append([0]*key_len[b].item()+[1]*(keys.size(2)-key_len[b].item()))
                mask.append([0]*key_len[b].item()+[1]*(keys.size(1)-key_len[b].item()))
            mask = cc(torch.ByteTensor(mask).unsqueeze(1)) # [BxL] -> [Bx1xL]
            energy = energy.masked_fill_(mask, -1e10)
        energy = F.softmax(energy * scaling, dim=2) # [Bx1xL]
        context = torch.bmm(energy, value) # [Bx1xV]
        return context.squeeze(1), energy

class MultiHeadDotAttn(nn.Module):
    def __init__(self, query_dim, key_dim, att_dim, heads):
        '''
        basic setting:
        query_dim is decoder hidden dim
        key_dim is encoder output dim
        att_dim is projected dim
        '''
        super().__init__()
        self.heads = heads
        self.mlp_query = nn.ModuleList([nn.Linear(query_dim, att_dim, bias=False) for _ in range(heads)])
        self.mlp_key = nn.ModuleList([nn.Linear(key_dim, att_dim, bias=False) for _ in range(heads)])
    
    def forward(self, query, keys, value, key_len=None, scaling=1.0):
        '''
        :param query:
            previous hidden state of the decoder, in shape (batch, dec_dim)
        :param keys:
            encoder output, in shape (batch, enc_maxlen, enc_dim)
        :param key_len:
            encoder output lens, in shape (batch)
        :param value:
            usually encoder output, in shape (batch, enc_maxlen, enc_dim)
        '''
        cs, es = [], []
        for h in range(self.heads):
            q = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
            q = self.mlp_query[h](q) # [Bx1xQ] -> [Bx1xA]
            k = self.mlp_key[h](keys).transpose(1,2) # [BxLxK] -> [BxLxA] -> [BxAxL]
            energy = torch.bmm(q, k) # [Bx1xL]
            if key_len is not None:
                mask = []
                for b in range(key_len.size(0)):
                    mask.append([0]*key_len[b].item()+[1]*(keys.size(1)-key_len[b].item()))
                mask = cc(torch.ByteTensor(mask).unsqueeze(1)) # [BxL] -> [Bx1xL]
                energy = energy.masked_fill_(mask, -1e10)
            energy = F.softmax(energy * scaling, dim=2) # [Bx1xL]
            context = torch.bmm(energy, value) # [Bx1xV]
            es.append(energy.squeeze(1))
            cs.append(context.squeeze(1))
        contexts = torch.cat(cs, dim=1) #[B x (heads*V)]
        energys = torch.stack(es, dim=1) #[B x heads x L]
        return contexts, energys

'''
class LSTMAttClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, attention_dim, pre_embedding=None,
                 update_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        if pre_embedding is not None:
            self.embedding = self.embedding.from_pretrained(pre_embedding, 
                                                            freeze=not(update_embedding))
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention = MultiHeadDotAttn(rnn_hidden_dim*2, rnn_hidden_dim*2,
                                          attention_dim, 3)
        self.dnn = nn.Sequential(
            nn.Linear(rnn_hidden_dim*6, dnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dnn_hidden_dim, 2),
        )
                          
    def forward(self, x, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            x = x[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        embedded = self.embedding(x)
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.rnn.flatten_parameters()
        xpack, _ = self.rnn(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True, total_length=total_length)
        rnnout = self.dropout(xpad)
        query = get_enc_context(rnnout, cc(ilens))
        query = query.new_ones(query.size())
        context, att_energy = self.attention(query, rnnout, rnnout, 
                                             key_len=cc(ilens), scaling=1.0)
        #att_energy = None
        #context = query
        if need_sort:
            rnnout=rnnout[unsort_idx]
            ilens=ilens[unsort_idx]
            context=context[unsort_idx]
            att_energy=att_energy[unsort_idx]
        context = self.dropout(context)
        logits = self.dnn(context)
        log_probs = F.log_softmax(logits, dim=1)
        prediction = log_probs.topk(1, dim=1)[1]
        return logits, log_probs, prediction, att_energy
'''

class LSTMAttClassifier(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, attention_dim):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention = MultiHeadDotAttn(rnn_hidden_dim*2, rnn_hidden_dim*2,
                                          attention_dim, 3)
        self.dnn = nn.Sequential(
            nn.Linear(rnn_hidden_dim*6, dnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dnn_hidden_dim, 2),
        )
                          
    def forward(self, embedded, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            embedded = embedded[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.rnn.flatten_parameters()
        xpack, _ = self.rnn(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True, total_length=total_length)
        rnnout = self.dropout(xpad)
        query = get_enc_context(rnnout, cc(ilens))
        query = query.new_ones(query.size())
        context, att_energy = self.attention(query, rnnout, rnnout, 
                                             key_len=cc(ilens), scaling=1.0)
        #att_energy = None
        #context = query
        if need_sort:
            rnnout=rnnout[unsort_idx]
            ilens=ilens[unsort_idx]
            context=context[unsort_idx]
            att_energy=att_energy[unsort_idx]
        context = self.dropout(context)
        logits = self.dnn(context)
        log_probs = F.log_softmax(logits, dim=1)
        prediction = log_probs.topk(1, dim=1)[1]
        return logits, log_probs, prediction, att_energy

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad=0, pre_embedding=None, 
                 update_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        if pre_embedding is not None:
            self.embedding = self.embedding.from_pretrained(pre_embedding, 
                                                            freeze=not(update_embedding))
    def forward(self, x):
        return self.embedding(x)
