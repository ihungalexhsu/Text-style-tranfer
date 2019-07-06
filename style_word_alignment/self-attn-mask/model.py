import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import cc
from utils import pad_list
from utils import get_enc_context
import random
import os

class dotAttn_2(nn.Module):
    def __init__(self, query_dim, key_dim, att_dim):
        '''
        basic setting:
        query_dim is decoder hidden dim
        key_dim is encoder output dim
        att_dim is projected dim
        '''
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(query_dim, key_dim))
    
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
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        energy = torch.matmul(torch.matmul(query, self.matrix), keys.transpose(1,2)) # [Bx1xL]
        if key_len is not None:
            mask = []
            for b in range(key_len.size(0)):
                mask.append([0]*key_len[b].item()+[1]*(keys.size(1)-key_len[b].item()))
            mask = cc(torch.ByteTensor(mask).unsqueeze(1)) # [BxL] -> [Bx1xL]
            energy = energy.masked_fill_(mask, -1e10)
        energy = F.softmax(energy * scaling, dim=2) # [Bx1xL]
        context = torch.bmm(energy, value) # [Bx1xV]
        return context.squeeze(1), energy

class dotAttn(nn.Module):
    def __init__(self, query_dim, key_dim, att_dim):
        '''
        basic setting:
        query_dim is decoder hidden dim
        key_dim is encoder output dim
        att_dim is projected dim
        '''
        super().__init__()
        self.mlp_query = nn.Linear(query_dim, att_dim, bias=False)
        self.mlp_key = nn.Linear(key_dim, att_dim, bias=False)
    
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
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        query = self.mlp_query(query) # [Bx1xQ] -> [Bx1xA]
        keys = self.mlp_key(keys).transpose(1,2) # [BxLxK] -> [BxLxA] -> [BxAxL]
        energy = torch.bmm(query, keys) # [Bx1xL]
        if key_len is not None:
            mask = []
            for b in range(key_len.size(0)):
                mask.append([0]*key_len[b].item()+[1]*(keys.size(2)-key_len[b].item()))
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
        #self.mlp_key = nn.ModuleList([nn.Linear(key_dim, query_dim, bias=False) for _ in range(heads)])
    
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
        contexts = torch.cat(cs, dim=1)
        energys = torch.stack(es, dim=1)
        return contexts, energys

class LSTMAttClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, attention_dim,
                 pad_idx=0, pre_embedding=None, update_embedding=True):
        super().__init__()
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=not(update_embedding))
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        #self.attention = dotAttn_2(rnn_hidden_dim*2, rnn_hidden_dim*2, attention_dim) 
        self.attention = MultiHeadDotAttn(rnn_hidden_dim*2, rnn_hidden_dim*2,
                                          attention_dim, 2)
        self.dnn = nn.Sequential(
            nn.Linear(rnn_hidden_dim*4, dnn_hidden_dim),
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
        embedded = self.dropout(self.embedding(x))
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.rnn.flatten_parameters()
        xpack, _ = self.rnn(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True, total_length=total_length)
        rnnout = self.dropout(xpad)
        query = get_enc_context(rnnout, cc(ilens))
        #query = query.new_ones(query.size())
        context, att_energy = self.attention(query, rnnout, rnnout, 
                                             key_len=cc(ilens), scaling=1.0)
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

class StructureSelfAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, attention_dim, attention_hop,
                 pad_idx=0, pre_embedding=None, update_embedding=True):
        super().__init__()
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=not(update_embedding))
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ws1 = nn.Linear(rnn_hidden_dim*2, attention_dim, bias=False)
        self.ws2 = nn.Linear(attention_dim, attention_hop, bias=False)
        self.tanh = nn.Tanh()
        self.dnn = nn.Sequential(
            nn.Linear(2*rnn_hidden_dim*attention_hop, dnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dnn_hidden_dim, 2),
        )
        self.attention_hop = attention_hop
        self.attention_dim = attention_dim
                          
    def forward(self, x, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            x = x[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        embedded = self.dropout(self.embedding(x))
        total_length = embedded.size(1)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        self.rnn.flatten_parameters()
        xpack, _ = self.rnn(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True, total_length=total_length)
        ilens = cc(ilens)
        rnnout = self.dropout(xpad) #(batch, len, hid_dim*2)
        attn_energy = self.ws2(self.tanh(self.ws1(rnnout))) #(batch, len, hop)
        #attn_energy = self.ws2(self.ws1(rnnout)) #(batch, len, hop)
        #all_ones = cc(torch.ones((x.size(0), self.attention_hop, self.attention_dim)))
        #attn_energy = torch.bmm(self.ws1(rnnout), self.ws2(all_ones).transpose(1,2))
        mask = []
        for b in range(ilens.size(0)):
            mask.append([0]*ilens[b].item()+[1]*(x.size(1)-ilens[b].item()))
        mask = cc(torch.ByteTensor(mask).unsqueeze(2)) #(batch, len) -> (batch, len, 1)
        attn_energy = attn_energy.masked_fill_(mask, -1e10)
        attn_energy = F.softmax(attn_energy.transpose(1,2), dim=2) #(batch, hop, len)
        context = torch.bmm(attn_energy, rnnout) #(batch, hop, len)*(batch, len, hid_dim*2)
        if need_sort:
            rnnout=rnnout[unsort_idx]
            ilens=ilens[unsort_idx]
            context=context[unsort_idx] #(batch, hop, hid_dim*2)
            attn_energy=attn_energy[unsort_idx]
        context = self.dropout(context.view(x.size(0), -1))
        logits = self.dnn(context)
        log_probs = F.log_softmax(logits, dim=1)
        prediction = log_probs.topk(1, dim=1)[1]
        return logits, log_probs, prediction, attn_energy

