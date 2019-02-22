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
import os

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate, 
                 pad_idx=0, bidirectional=True, pre_embedding=None, update_embedding=True):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(pre_embedding)
        self.embedding.weight.requires_grad = update_embedding
        layers = []
        hdim = hidden_dim*2 if bidirectional else hidden_dim
        for i in range(n_layers):
            input_dim = embedding_dim if i==0 else hdim
            layers.append(nn.GRU(input_dim, hidden_dim, num_layers=1,
                                 bidirectional=bidirectional, batch_first=True))
        self.enc = nn.ModuleList(layers)
        self.dropout_rate=dropout_rate
                          
    def forward(self, x, ilens, need_sort=False):
        if need_sort:
            sort_idx = np.argsort((-ilens).cpu().numpy())
            x = x[sort_idx]
            ilens = ilens[sort_idx]
            unsort_idx = np.argsort(sort_idx)
        embedded = self.embedding(x)
        total_length = embedded.size(1)
        for layer in self.enc:
            xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
            layer.flatten_parameters()
            xpack, _ = layer(xpack)
            xpad, ilens = pad_packed_sequence(xpack, batch_first=True,
                                              total_length=total_length)
            embedded = F.dropout(xpad, self.dropout_rate, training=self.training)
        embedded = torch.tanh(embedded)
        if need_sort:
            embedded = embedded[unsort_idx]
            ilens = ilens[unsort_idx]
        return embedded, cc(ilens)

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, 
                 dropout_rate, bos, eos, pad, enc_out_dim,
                 n_styles, style_emb_dim=1, use_enc_init=True,
                 use_attention=False, attention=None, att_odim=100,
                 use_style_embedding=True, ls_weight=0, labeldist=None, 
                 give_context_directly=False):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        self.embedding = torch.nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_style_embedding = use_style_embedding
        if use_style_embedding:    
            self.style_embedding = torch.nn.Embedding(n_styles, style_emb_dim)
            torch.nn.init.orthogonal_(self.style_embedding.weight)
        self.attention = attention
        self.att_odim = att_odim
        self.dropout_rate = dropout_rate
        self.use_enc_init = use_enc_init
        if use_attention:
            if use_style_embedding:
                self.GRUCell = nn.GRUCell(embedding_dim+style_emb_dim+att_odim, hidden_dim)
            else:
                self.GRUCell = nn.GRUCell(embedding_dim+att_odim, hidden_dim)
            self.output_layer = torch.nn.Linear(hidden_dim+att_odim, output_dim)
        else:
            # in this version, input contains input_words+style+enc_context
            if use_style_embedding:
                self.GRUCell = nn.GRUCell(embedding_dim+style_emb_dim+enc_out_dim, hidden_dim)
            else:
                self.GRUCell = nn.GRUCell(embedding_dim+enc_out_dim, hidden_dim)
            # in this version, input contains input_words only
            #self.GRUCell = nn.GRUCell(embedding_dim)
            self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        if use_enc_init:
            # projecting layer to map the encoder ouput dim to dec_hidden_dim
            if use_style_embedding:
                self.project = torch.nn.Linear(enc_out_dim+style_emb_dim, hidden_dim)
            else:
                self.project = torch.nn.Linear(enc_out_dim, hidden_dim)        
        self.GRUCell2 = nn.GRUCell(hidden_dim, hidden_dim)
        # label smoothing hyperparameters
        self.ls_weight = ls_weight
        self.labeldist = labeldist
        self.give_context_directly = give_context_directly
        if labeldist is not None:
            self.vlabeldist = cc(torch.from_numpy(np.array(labeldist, dtype=np.float32)))

    def zero_state(self, ori_tensor, dim=None):
        '''
        a util function that new a zero tensor at the same shape of (batch, dim)
        '''
        if not dim:
            return ori_tensor.new_zeros(ori_tensor.size(0), self.hidden_dim)
        else:
            return ori_tensor.new_zeros(ori_tensor.size(0), dim)

    def forward_step(self, emb, style_emb, dec_h, context, attn, enc_output, enc_len):
        if self.use_style_embedding:
            cell_inp = torch.cat([emb, style_emb, context], dim=-1)
        else:
            cell_inp = torch.cat([emb, context], dim=-1)
        cell_inp = F.dropout(cell_inp, self.dropout_rate, training=self.training)
        dec_h0 = self.GRUCell(cell_inp, dec_h[0])
        dec_h1 = self.GRUCell2(dec_h0, dec_h[1])
        if self.use_attention:
            context, attn = self.attention(enc_output, enc_len, dec_h1, attn)
            output = torch.cat([dec_h1, context], dim=-1)
        else:
            output = dec_h1

        output = F.dropout(output, self.dropout_rate, training=self.training)
        logit = self.output_layer(output)
        dec_h = (dec_h0, dec_h1)
        return logit, dec_h, context, attn

    def forward(self, enc_output, enc_len, styles, dec_input=None, tf_rate=1.0, 
                max_dec_timesteps=500, sample=False):
        '''
        if give_context_directly, the input of enc_output would be diretly the last
        hidden state of encoder, rather than a sequence of encoder output.
        it means that it cannot support attention.
        also, the shape of enc_output would be like (batch, enc_out_dim)
        '''
        batch_size = enc_output.size(0)
        if dec_input is not None:
            # dec_input would be a tuple (dec_in, dec_out)
            pad_dec_input_in = dec_input[0]
            pad_dec_input_out = dec_input[1]
            # get length info
            batch_size, olength = pad_dec_input_out.size(0), pad_dec_input_out.size(1)
            # map idx to embedding
            dec_input_embedded = self.embedding(pad_dec_input_in)
        if self.use_style_embedding:
            style_embedded = self.style_embedding(styles) #batch x style_emb_dim
        else:
            style_embedded = None

        # initialization
        if self.use_enc_init:
            if self.use_style_embedding:
                if self.give_context_directly:
                    dec_h = self.project(torch.cat([enc_output,style_embedded]), dim=-1)
                else:
                    dec_h = self.project(torch.cat([get_enc_context(enc_output,enc_len),style_embedded], dim=-1))
            else:
                if self.give_context_directly:
                    dec_h = self.project(enc_output)
                else:
                    dec_h = self.project(get_enc_context(enc_output,enc_len))

        else:
            dec_h = self.zero_state(enc_output)
        dec_h2 = self.zero_state(enc_output)
        dec_h = (dec_h , dec_h2)
        if self.use_attention:
            if self.give_context_directly:
                raise ValueError('cannot use attention and give context directly together')
            else:
                context = self.zero_state(enc_output, dim=self.att_odim)
        else:
            if self.give_context_directly:
                context = enc_output
            else:
                context = get_enc_context(enc_output, enc_len)

        attn = None
        logits, prediction, attns = [], [], []
        # reset the attention module
        if self.use_attention:
            try:
                self.attention.module.reset()
            except:
                self.attention.reset()

        # loop for each timestep
        olength = max_dec_timesteps if not dec_input else olength
        for t in range(olength):
            if dec_input is not None:
                # teacher forcing
                tf = True if np.random.random_sample() <= tf_rate else False
                if tf or t==0:
                    emb = dec_input_embedded[:,t,:]
                else:
                    self.embedding(prediction[-1])
            else:
                if t == 0:
                    bos = cc(torch.Tensor([self.bos for _ in range(batch_size)]).type(torch.LongTensor))
                    emb = self.embedding(bos)
                else:
                    emb = self.embedding(prediction[-1])

            logit, dec_h, context, attn = \
                self.forward_step(emb, style_embedded, dec_h, context, attn, 
                                  enc_output, enc_len)

            attns.append(attn)
            logits.append(logit)
            if not sample:
                prediction.append(torch.argmax(logit, dim=-1))
            else:
                sampled_indices = Categorical(logits=logit).sample() 
                prediction.append(sampled_indices)

        logits = torch.stack(logits, dim=1) # batch x length x output_dim
        log_probs = F.log_softmax(logits, dim=2)
        prediction = torch.stack(prediction, dim=1) # batch x length
        if self.use_attention:
            attns = torch.stack(attns, dim=1) # batch x length x enc_len

        # get the log probs of the true label(batch x length)
        if dec_input:
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=pad_dec_input_out.unsqueeze(2)).squeeze(2)
        else:
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=prediction.unsqueeze(2)).squeeze(2)

        # label smoothing : q'(y|x) = (1-e)*q(y|x) + e*u(y)
        if self.ls_weight > 0:
            loss_reg = torch.sum(log_probs * self.vlabeldist, dim=2) # u(y)
            dec_output_log_probs=(1-self.ls_weight)*dec_output_log_probs+self.ls_weight*loss_reg

        return logits, dec_output_log_probs, prediction, attns

class Domain_discri(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, pad_idx=0, pre_embedding=None, update_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(pre_embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.dropout_rate = dropout_rate
        self.dnn = nn.Sequential(
            nn.Linear(2*rnn_hidden_dim, dnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dnn_hidden_dim, dnn_hidden_dim),
            nn.LeakyReLU(0.2),
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
        rnnout = F.dropout(xpad, self.dropout_rate, training=self.training)
        if need_sort:
            rnnout=rnnout[unsort_idx]
            ilens=ilens[unsort_idx]
        logits = self.dnn(get_enc_context(rnnout,cc(ilens)))
        log_probs = F.log_softmax(logits, dim=1)
        prediction = log_probs.topk(1, dim=1)[1]
        return logits, log_probs, prediction

class DenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_vec):
        '''
        hidden_dim_vec is a vector store hidden dim
        e.g. [128,256,128]
        '''
        super().__init__()
        layers = []
        assert len(hidden_dim_vec)>0
        for i, h_d in enumerate(hidden_dim_vec):
            idim = input_dim if i==0 else hidden_dim_vec[i-1]
            layers.append(nn.Linear(idim, h_d))

        self.hidden_layer = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim_vec[-1], output_dim)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, input_var):
        for layer in self.hidden_layer:
            input_var = layer(input_var)
            input_var = self.activation(input_var)
        output = self.output_layer(input_var)
        return torch.tanh(output)

class Dense_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_vec):
        '''
        hidden_dim_vec is a vector store hidden dim
        e.g. [128,256,128]
        '''
        super().__init__()
        layers=[]
        assert len(hidden_dim_vec)>0
        for i, h_d in enumerate(hidden_dim_vec):
            idim = input_dim if i==0 else hidden_dim_vec[i-1]
            layers.append(nn.Linear(idim, h_d))

        self.hidden_layer = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim_vec[-1], output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input_var):
        for layer in self.hidden_layer:
            input_var = layer(input_var)
            input_var = self.activation(input_var)
        
        logits = self.output_layer(input_var) # batch x out_dim
        log_probs = F.log_softmax(logits, dim=1)
        prediction = log_probs.topk(1, dim=1)[1]
        return logits, log_probs, prediction

