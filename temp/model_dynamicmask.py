import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
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

class E2E(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, enc_layers, 
                 enc_dropout, bidirectional, maskgen_dim, dec_hidden_dim,
                 dec_dropout, bos, eos, pad, n_styles, style_emb_dim, att_dim,
                 classifier_dim, cls_dropout, cls_att_dim,
                 pre_embedding=None, update_embedding=True):
        super().__init__()
        '''
        if pre_embedding is not None:
            self.embedding = self.embedding.from_pretrained(pre_embedding, freeze=not(update_embedding))
        '''
        self.embedding_mask = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        self.embedding_style = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        self.embedding_content = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad)
        self.maskgen = MaskGen(embedding_dim, maskgen_dim, bidirectional)
        self.encoder = Encoder(embedding_dim, enc_hidden_dim, enc_layers,
                               enc_dropout, bidirectional)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_hidden_dim,
                               dec_dropout, bos, eos, pad, 
                               enc_hidden_dim*2 if bidirectional else enc_hidden_dim,
                               n_styles, style_emb_dim, att_dim, 
                               pre_embedding, update_embedding)
        self.classifier = LSTMAttClassifier(embedding_dim, classifier_dim, 
                                            cls_dropout, classifier_dim, cls_att_dim)
        #self.discriminator = LSTMAttClassifier(embedding_dim, classifier_dim, 
        #                                       cls_dropout, classifier_dim, cls_att_dim)
        #self.gradrever = GradientReversal(lambda_=1)

    def forward(self, xs, ilens, styles, dec_input=None, tf_rate=1.0,
                max_dec_timesteps=15, sample=False):
        emb_mask = self.embedding_mask(xs)
        mask, _ = self.maskgen(emb_mask, ilens)
        mask = mask[:,:,1]
        style_words = xs.cuda().float()*mask
        content_words = xs.cuda().float()-style_words
        emb_style = self.embedding_style(style_words.long())
        emb_content = self.embedding_content(content_words.long())
        _, cls_log_probs, cls_predicts, _ = self.classifier(emb_style, ilens)
        enc_outputs, enc_lens = self.encoder(emb_content, ilens)
        _, recon_log_probs, predictions, attns =\
            self.decoder(enc_outputs, enc_lens, styles, dec_input, tf_rate,
                         max_dec_timesteps, sample)
        #gradRever_content_words = self.gradrever(content_words)
        #_, dis_log_probs, dis_predicts, _ =\
        #    self.discriminator(gradRever_content_words, ilens)
        return mask,cls_log_probs,cls_predicts,recon_log_probs,predictions,attns

class MaskGen(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=True):
        super().__init__()
        self.BiRNN = nn.GRU(input_dim, hidden_dim, num_layers=1,
                            bidirectional=bidirectional, batch_first=True)
        self.Output_layer = nn.GRU(hidden_dim*2 if bidirectional else hidden_dim, 
                                   2, num_layers=1, bidirectional=False, 
                                   batch_first=True)
    
    def forward(self, x, ilens):
        total_length = x.size(1)
        xpack = pack_padded_sequence(x, ilens, batch_first=True)
        self.BiRNN.flatten_parameters()
        xpack, _ = self.BiRNN(xpack)
        self.Output_layer.flatten_parameters()
        xpack, _ = self.Output_layer(xpack)
        xpad, ilens = pad_packed_sequence(xpack, batch_first=True,
                                          total_length=total_length)
        xpad = F.gumbel_softmax(xpad, tau=0.5, hard=True, dim=-1)
        return xpad, cc(ilens)

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

class SentimentModule(nn.Module):
    def __init__(self, emoword_list, pre_embedding, dec_vocab_size):
        super().__init__()
        self.emoword_list = emoword_list # tensor of word_idx, length r
        self.dec_vocab_size = dec_vocab_size
        self.create_mapping()
        self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=True)
        self.emoword_emb = self.embedding(self.emoword_list) # r x emb_dim
        self.matrix = None
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def create_mapping(self):
        reverse_map = dict()
        for k,v in enumerate(self.emoword_list):
            reverse_map[v.cpu().int().item()]=k
        self.mapping = list()
        for i in range(self.dec_vocab_size):
            if i in reverse_map.keys():
                self.mapping.append(reverse_map[i])
            else:
                self.mapping.append(self.emoword_list.size(0))
        self.mapping = cc(torch.LongTensor(self.mapping))
        return

    def create_matrix(self, batch_input_words):
        # batch_input_words is a 2d tensor: batch x seq_len
        input_emb = self.embedding(batch_input_words) # batch x seq_len x emb_dim
        self.matrix = input_emb.new_zeros(input_emb.size(0), input_emb.size(1), self.emoword_emb.size(0)) # batch x seq_len x r
        for i,seq in enumerate(input_emb):
            self.matrix[i]=torch.from_numpy(cosine_similarity(seq.cpu().numpy(),self.emoword_emb.cpu().numpy()))
        self.matrix = cc(self.matrix)
        return
        
    def clean_matrix(self):
        self.matrix = None
        return 

    def forward(self, attn):
        # attn is a 2d tensor: batch x seq_len
        aux_prob = torch.bmm(attn.unsqueeze(1), self.matrix).squeeze(1)
        aux_prob = self.get_output_prob(aux_prob)
        aux_prob = self.LogSoftmax(aux_prob) # batch x r
        return aux_prob
    
    def get_output_prob(self, aux_prob):
        pad = aux_prob.new_zeros(aux_prob.size(0), 1)
        aux_prob = torch.cat([aux_prob,pad], dim=1)
        output_prob = torch.index_select(aux_prob, 1, self.mapping)
        return output_prob

class DecWithAux(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, 
                 dropout_rate, bos, eos, pad, enc_out_dim,
                 n_styles, style_emb_dim, att_dim,
                 pos_word_list, neg_word_list, Smodule_emb,
                 pre_embedding=None, update_embedding=True):
        super().__init__()
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
        self.pos_sentimodule = SentimentModule(pos_word_list, Smodule_emb, output_dim)
        self.neg_sentimodule = SentimentModule(neg_word_list, Smodule_emb, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.GRUCell = nn.GRUCell(embedding_dim+style_emb_dim+enc_out_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(enc_out_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.switch_layer = nn.Linear(enc_out_dim+hidden_dim+embedding_dim, 1)

    def zero_state(self, ori_tensor, dim=None):
        '''
        a util function that new a zero tensor at the same shape of (batch, dim)
        '''
        if not dim:
            return ori_tensor.new_zeros(ori_tensor.size(0), self.hidden_dim)
        else:
            return ori_tensor.new_zeros(ori_tensor.size(0), dim)

    def forward_step(self, emb, style_emb, dec_h, context, mask_enc_out, 
                     enc_output, enc_len, coverage):
        cell_inp = torch.cat([emb, style_emb, context], dim=-1)
        cell_inp = self.dropout(cell_inp)
        dec_h = self.GRUCell(cell_inp, dec_h)
        dec_h = self.dropout(dec_h)
        context, attn = self.attention(dec_h, enc_output, mask_enc_out, 
                                       coverage, enc_len)
        #context, attn = self.attention(dec_h, enc_output, enc_output, 
        #                               coverage, enc_len)
        out_layer_inp = torch.cat([context, dec_h], dim=-1)
        out_layer_inp = self.dropout(out_layer_inp)
        logit = self.output_layer(out_layer_inp)
        log_prob = F.log_softmax(logit, dim=1)
        return log_prob, dec_h, context, attn

    def forward(self, enc_output, enc_len, styles, enc_mask, senti, input_words,
                dec_input=None, tf_rate=1.0, max_dec_timesteps=15, sample=False):
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

        # prepare mask_enc_out
        mask_enc_out = enc_output.masked_fill_(enc_mask.unsqueeze(2).expand_as(enc_output), 0)

        # initialization
        dec_h = self.zero_state(enc_output)
        context = self.zero_state(enc_output, dim=self.enc_out_dim)
        if senti==1:
            self.pos_sentimodule.create_matrix(input_words)
        else:
            self.neg_sentimodule.create_matrix(input_words)
        final_probs, switchs, prediction, attns = [], [], [], []
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
            log_prob, dec_h, context, attn = \
                self.forward_step(emb, style_emb, dec_h, context, mask_enc_out,
                                  enc_output, enc_len, coverage)
            attns.append(attn.squeeze(1))
            #print('t', t)
            #print('attention',torch.argmax(attn.squeeze(1), dim=-1))
            switch = torch.sigmoid(self.switch_layer(torch.cat([emb, context, dec_h], dim=-1)))#batchx1
            switchs.append(switch)
            if senti==1:
                #aux_log_prob = self.pos_sentimodule.forward(attn.squeeze(1))
                aux_log_prob = self.pos_sentimodule.forward(enc_mask.float())
            else:
                #aux_log_prob = self.neg_sentimodule.forward(attn.squeeze(1))
                aux_log_prob = self.neg_sentimodule.forward(enc_mask.float())
            if t != enc_mask.size(1):
                if enc_mask[0,t]:
                    print('t', t)
                    print("attn", attn.squeeze(1)[0])
                    print("sentimodule pred", torch.argmax(aux_log_prob, dim=-1)[0])
                    print("ans", pad_dec_out[0,t])
                    print("switch",switch[0])
                    print()
            final_prob = switch*log_prob + (1-switch)*aux_log_prob 
            #final_prob = 0.5*log_prob + 0.5*aux_log_prob 
            final_probs.append(final_prob)
            if not sample:
                prediction.append(torch.argmax(final_prob, dim=-1))
            else:
                sampled_indices = Categorical(probs=torch.exp(final_prob)).sample() 
                prediction.append(sampled_indices)
        if senti==1:
            self.pos_sentimodule.clean_matrix()
        else:
            self.neg_sentimodule.clean_matrix()

        final_probs = torch.stack(final_probs, dim=1) # batch x length x output_dim
        prediction = torch.stack(prediction, dim=1) # batch x length
        attns = torch.stack(attns, dim=1) # batch x length x enc_len
        switchs = torch.stack(switchs, dim=1) # batch x length

        # get the log probs of the true label(batch x length)
        if dec_input is not None:
            dec_output_log_probs = torch.gather(final_probs, dim=2, index=pad_dec_out.unsqueeze(2)).squeeze(2)
        else:
            dec_output_log_probs = torch.gather(final_probs, dim=2, index=prediction.unsqueeze(2)).squeeze(2)

        return switchs, dec_output_log_probs, prediction, attns

class LSTMAttClassifier(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_dim, dropout_rate,
                 dnn_hidden_dim, attention_dim):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention = MultiHeadDotAttn(rnn_hidden_dim*2, rnn_hidden_dim*2,
                                          attention_dim, 2)
        self.dnn = nn.Sequential(
            nn.Linear(rnn_hidden_dim*2, dnn_hidden_dim),
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
        #query = query.new_ones(query.size())
        #context, att_energy = self.attention(query, rnnout, rnnout, 
        #                                     key_len=cc(ilens), scaling=1.0)
        att_energy = None
        context = query
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

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_=lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        lambda_=ctx.lambda_
        lambda_=grads.new_tensor(lambda_)
        dx=-lambda_*grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super().__init__()
        self.lambda_=lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

