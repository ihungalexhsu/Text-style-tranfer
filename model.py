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
from torch.distributions.categorical import Categorical
import random
import os

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate, 
                 pad_idx=0, bidirectional=True, pre_embedding=None, update_embedding=True):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(pre_embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.enc = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout_rate)

    def forward(self, x, ilens):
        embedded = self.embedding(x)
        xpack = pack_padded_sequence(embedded, ilens, batch_first=True)
        output, hidden = self.enc(xpack)
        output, ilens = pad_packed_sequence(output, batch_first=True)
        ilens = np.array(ilens, dtype=np.int64).tolist()
        return output, ilens

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, 
                 dropout_rate, bos, eos, pad,
                 use_attention=False, attention=None, att_odim=100, enc_out_dim,
                 ls_weight=0, labeldist=None):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        self.embedding = torch.nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.attention = attention
        self.att_odim = att_odim
        self.dropout_rate = dropout_rate
        
        if use_attention:
            self.GRUCell = nn.GRUCell(embedding_dim + att_odim, hidden_dim)
            self.output_layer = torch.nn.Linear(hidden_dim + att_odim, output_dim)
        else:
            self.GRUCell = nn.GRUCell(embedding_dim + enc_out_dim, hidden_dim)
            self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        # label smoothing hyperparameters
        self.ls_weight = ls_weight
        self.labeldist = labeldist
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

    def forward_step(self, emb, dec_h, context, attn, enc_output, enc_len):
        cell_inp = torch.cat([emb, context], dim=-1)
        cell_inp = F.dropout(cell_inp, self.dropout_rate, training=self.training)
        dec_h = self.GRUCell(cell_inp, dec_h)

        if self.use_attention:
            # run attention module
            context, attn = self.attention(enc_output, enc_len, dec_h, attn)
            output = torch.cat([dec_h, context], dim=-1)
        else:
            output = dec_h

        output = F.dropout(output, self.dropout_rate)
        logit = self.output_layer(output)
        return logit, dec_h, context, attn

    def forward(self, enc_output, enc_len, dec_input=None, tf_rate=1.0, max_dec_timesteps=500, sample=False):
        batch_size = enc_output.size(0)
        if dec_input is not None:
            # dec_input shape: (batch, len)
            # prepare input and output sequences
            bos = dec_input[0].data.new([self.bos])
            eos = dec_input[0].data.new([self.eos])
            dec_input_in = [torch.cat([bos, y], dim=0) for y in dec_input]
            dec_input_out = [torch.cat([y, eos], dim=0) for y in dec_input]
            pad_dec_input_in = pad_list(dec_input_in, pad_value=self.pad)
            pad_dec_input_out = pad_list(dec_input_out, pad_value=self.pad)
            # get length info
            batch_size, olength = pad_dec_input_out.size(0), pad_dec_input_out.size(1)
            # map idx to embedding
            dec_input_embedded = self.embedding(pad_dec_input_in)

        # initialization
        dec_h = self.zero_state(enc_output)
        if self.use_attention:
            context = self.zero_state(enc_output, dim=self.att_odim)
        else:
            context = enc_output[:,-1,:]

        attn = None
        logits, prediction, attns = [], [], []
        # reset the attention module
        if self.use_attention:
            if torch.cuda.is_available():
                try:
                    self.attention.module.reset()
                except:
                    self.attention.reset()
            else:
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
                self.forward_step(emb, dec_h, context, attn, enc_output, enc_len)

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
            dec_output_log_probs = (1 - self.ls_weight) * dec_output_log_probs + self.ls_weight * loss_reg

        return logits, dec_output_log_probs, prediction, attns

class E2E(torch.nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, enc_n_layers, subsample, dropout_rate, 
                 dec_hidden_dim, att_dim, conv_channels, conv_kernel_size, att_odim,
                 embedding_dim, output_dim, ls_weight, labeldist, 
                 pad=0, bos=1, eos=2):

        super(E2E, self).__init__()

        # encoder to encode acoustic features
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=enc_hidden_dim, 
                               n_layers=enc_n_layers, subsample=subsample, 
                               dropout_rate=dropout_rate)

        # attention module
        self.attention = AttLoc(encoder_dim=enc_hidden_dim, 
                                decoder_dim=dec_hidden_dim, 
                                att_dim=att_dim, 
                                conv_channels=conv_channels, 
                                conv_kernel_size=conv_kernel_size, 
                                att_odim=att_odim)

        # decoder 
        self.decoder = Decoder(output_dim=output_dim, 
                               hidden_dim=dec_hidden_dim, 
                               embedding_dim=embedding_dim,
                               attention=self.attention, 
                               dropout_rate=dropout_rate, 
                               att_odim=att_odim, 
                               ls_weight=ls_weight, 
                               labeldist=labeldist, 
                               bos=bos, 
                               eos=eos, 
                               pad=pad)

    def forward(self, data, ilens, true_label=None, tf_rate=1.0, 
                max_dec_timesteps=200, sample=False):
        
        enc_outputs, enc_lens = self.encoder(data, ilens)
        logits, log_probs, prediction, attns =\
            self.decoder(enc_outputs, enc_lens, true_label, tf_rate=tf_rate, 
                         max_dec_timesteps=max_dec_timesteps, sample=sample)
        return log_probs, prediction, attns

    def mask_and_cal_loss(self, log_probs, ys, mask=None):
        # mask is batch x max_len
        # add 1 to EOS
        if mask is None: 
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
        else:
            seq_len = [y.size(0) for y in ys]
        # divide by total length
        loss = -torch.sum(log_probs * mask) / sum(seq_len)
        return loss

class disentangle_clean(nn.Module):
    def __init__(self, clean_repre_dim, hidden_dim, nuisance_dim):
        super().__init__()
        self.LSTM = torch.nn.LSTM(clean_repre_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_layer = torch.nn.Linear(hidden_dim*2, nuisance_dim)
        
    def forward(self, clean_repre):
        output,_ = self.LSTM(clean_repre) # batch_size x seq_len x hidden_dim
        output = self.output_layer(output)
        return output

class disentangle_nuisance(nn.Module):
    def __init__(self, nuisance_dim, hidden_dim, clean_repre_dim):
        super().__init__()
        self.LSTM = torch.nn.LSTM(nuisance_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_layer = torch.nn.Linear(hidden_dim*2, clean_repre_dim)
    
    def forward(self, nuisance_data):
        output,_ = self.LSTM(nuisance_data)
        output = self.output_layer(output)
        return output

class addnoiselayer(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, clean_repre):
        output = F.dropout(clean_repre, self.dropout_p, training=self.training)
        return output

class reconstructRNN(nn.Module):
    def __init__(self, attention, att_odim, hidden_dim, output_dim):
        super().__init__()
        self.LSTM = nn.LSTMCell(output_dim+att_odim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim+att_odim, output_dim)
        self.hidden_dim = hidden_dim
        self.att_odim = att_odim
        self.attention = attention

    def forward_step(self, recons_inp, dec_h, dec_c, c, attn, 
                     enc_output, enc_len):
        cell_inp = torch.cat([recons_inp, c], dim=-1)
        dec_h, dec_c = self.LSTM(cell_inp, (dec_h, dec_c))

        # run attention module
        c, attn = self.attention(enc_output, enc_len, dec_h, attn)
        output = torch.cat([dec_h, c], dim=-1)
        logit = self.output_layer(output)
        return logit, dec_h, dec_c, c, attn

    def forward(self, enc_output, enc_len, recons_inputs):
        '''
        enc_output : batch_size x enc_length x enc_dim
        enc_len : list of length on the batch
        recons_inputs : the gold input : batch_size x feature_length x output_dim
        '''
        #initialization
        dec_h = enc_output.new_zeros(enc_output.size(0), self.hidden_dim)
        dec_c = enc_output.new_zeros(enc_output.size(0), self.hidden_dim)
        context = enc_output.new_zeros(enc_output.size(0), self.att_odim)
        attn = None
        logits, attns = [], []
        self.attention.reset()

        for t in range(recons_inputs.size(1)):
            recons_inp = recons_inputs[:, t, :]

            logit, dec_h, dec_c, context, attn = \
                self.forward_step(recons_inp, dec_h, dec_c, context,
                                  attn, enc_output, enc_len)
            attns.append(attn)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=1) # batch x feature_length x output_dim
        attns = torch.stack(attns, dim=1) # batch x feature_length x enc_length

        return logits, attns

class inverse_pBLSTM(nn.Module):
    def __init__(self, enc_hidden_dim, hidden_dim, hidden_project_dim, output_dim, 
                 n_layers, downsample):
        super().__init__()
        layers, project_layers = [], []
        for i in range(n_layers):
            idim = (enc_hidden_dim) if i == 0 else (hidden_project_dim)
            project_dim = hidden_dim if downsample[i] > 1 else hidden_dim*2
            project_to_dim = output_dim if i == (n_layers-1) else hidden_project_dim
            layers.append(nn.LSTM(idim, hidden_dim, num_layers=1, 
                                  bidirectional=True, batch_first=True))
            project_layers.append(nn.Linear(project_dim, project_to_dim))

        self.layers = nn.ModuleList(layers)
        self.project_layers = nn.ModuleList(project_layers)
        self.downsample = downsample
    
    def forward(self, enc_output, enc_len):
        for i, (layer, project_layer) in enumerate(zip(self.layers, self.project_layers)):
            xs_pack = pack_padded_sequence(enc_output, enc_len, batch_first=True)
            xs, (_,_) = layer(xs_pack)
            ys_pad, enc_len = pad_packed_sequence(xs, batch_first=True)
            enc_len = enc_len.numpy()

            downsub = self.downsample[i]
            if downsub > 1:
                ys_pad = ys_pad.contiguous().view(ys_pad.size(0), ys_pad.size(1)*2, ys_pad.size(2)//2)
                enc_len = [(length*2) for length in enc_len]
            projected = project_layer(ys_pad)
            enc_output = F.relu(projected)
        output_lens= np.array(enc_len, dtype=np.int64).tolist()
        return enc_output, output_lens

class AttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size, att_odim):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(encoder_dim, att_dim)
        self.mlp_dec = torch.nn.Linear(decoder_dim, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(conv_channels, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(in_channels=1, out_channels=conv_channels, 
                                        kernel_size=(1, 2 * conv_kernel_size + 1),
                                        stride=1,
                                        padding=(0, conv_kernel_size), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1, bias=False)
        self.mlp_o = torch.nn.Linear(encoder_dim, att_odim)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.att_odim = att_odim
        self.conv_channels = conv_channels
        
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_pad, enc_len, dec_h, att_prev, scaling=2.0):
        '''
        enc_pad:(batch, enc_length, enc_dim)
        enc_len:(batch) of int
        dec_h:(batch, 1, dec_dim)
        att_prev:(batch, enc_length)
        '''
        batch_size = enc_pad.size(0)
               
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_pad
            self.enc_length = self.enc_h.size(1)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h) # batch_size x enc_length x att_dim

        if dec_h is None:
            dec_h = enc_pad.new_zeros(batch_size, self.decoder_dim)
        else:
            dec_h = dec_h.view(batch_size, self.decoder_dim)

        # initialize attention weights to uniform
        if att_prev is None:
            att_prev = pad_list([self.enc_h.new(l).fill_(1.0 / l) for l in enc_len], 0)

        att_conv = self.loc_conv(att_prev.view(batch_size, 1, 1, self.enc_length))
        att_conv = att_conv.squeeze(2).transpose(1, 2) 
        # att_conv: batch_size x channel x 1 x frame -> batch_size x frame x channel
        att_conv = self.mlp_att(att_conv) # att_conv: batch_size x frame x channel -> batch_size x frame x att_dim

        dec_h_tiled = self.mlp_dec(dec_h).view(batch_size, 1, self.att_dim)
        
        att_state = torch.tanh(self.pre_compute_enc_h + dec_h_tiled + att_conv)
        e = self.gvec(att_state).squeeze(2) # batch_size x enc_length x 1 => batch_size x enc_length
        attn = F.softmax(scaling * e, dim=1)
        w_expanded = attn.unsqueeze(1) # w_expanded: batch_size x 1 x frame
        
        c = torch.bmm(w_expanded, self.enc_h).squeeze(1) 
        # batch x 1 x frame * batch x enc_length x enc_dim => batchx1xenc_dim
        c = self.mlp_o(c)
        return c, attn

class MultiHeadAttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size, heads, att_odim):
        super(MultiHeadAttLoc, self).__init__()
        self.heads = heads
        self.mlp_enc = torch.nn.ModuleList([torch.nn.Linear(encoder_dim, att_dim) for _ in range(self.heads)])
        self.mlp_dec = torch.nn.ModuleList([torch.nn.Linear(decoder_dim, att_dim, bias=False) \
                for _ in range(self.heads)])
        self.mlp_att = torch.nn.ModuleList([torch.nn.Linear(conv_channels, att_dim, bias=False) \
                for _ in range(self.heads)])
        self.loc_conv = torch.nn.ModuleList([torch.nn.Conv2d(
                1, conv_channels, (1, 2 * conv_kernel_size + 1), 
                padding=(0, conv_kernel_size), bias=False) for _ in range(self.heads)])
        self.gvec = torch.nn.ModuleList([torch.nn.Linear(att_dim, 1, bias=False) for _ in range(self.heads)])
        self.mlp_o = torch.nn.Linear(self.heads * encoder_dim, att_odim)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.conv_channels = conv_channels
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_pad, enc_len, dec_h, att_prev, scaling=2.0):
        batch_size =enc_pad.size(0)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_pad
            self.enc_length = self.enc_h.size(1)
            self.pre_compute_enc_h = [self.mlp_enc[h](self.enc_h) for h in range(self.heads)]

        if dec_h is None:
            dec_h = enc_pad.new_zeros(batch_size, self.decoder_dim)
        else:
            dec_h = dec_h.view(batch_size, self.decoder_dim)

        # initialize attention weights to uniform
        if att_prev is None:
            att_prev = []
            for h in range(self.heads):
                att_prev += [pad_list([self.enc_h.new(l).fill_(1.0 / l) for l in enc_len], 0)]
        
        cs, ws = [], []
        for h in range(self.heads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch_size, 1, 1, self.enc_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)
            dec_h_tiled = self.mlp_dec[h](dec_h).view(batch_size, 1, self.att_dim)
            att_state = torch.tanh(self.pre_compute_enc_h[h] + dec_h_tiled + att_conv)
            e = self.gvec[h](att_state).squeeze(2)
            attn = F.softmax(scaling * e, dim=1)
            ws.append(attn)
            w_expanded = attn.unsqueeze(1)
            c = torch.bmm(w_expanded, self.enc_h).squeeze(1)
            cs.append(c)
        c = self.mlp_o(torch.cat(cs, dim=1))
        return c, ws 

