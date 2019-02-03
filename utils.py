import torch 
import numpy as np
from tensorboardX import SummaryWriter
import editdistance
import torch.nn as nn
import torch.nn.init as init


def pad_list(xs, pad_value=0):
    '''
    xs is a list of tensor
    output would be a already padded tensor
    '''
    batch_size = len(xs)
    max_length = max(x.size(0) for x in xs)
    pad = xs[0].data.new(batch_size, max_length, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(batch_size):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

def cc_model(net):
    if torch.cuda.is_available():
        return nn.DataParallel(net).cuda()
    else:
        device = torch.device('cpu')
        return net.to(device)

def to_gpu(data, bos, eos, pad):
    token_ids, ilens, labels = data
    ilens = cc(torch.LongTensor(ilens))
    labels = cc(labels)
    ys = [cc(y) for y in token_ids]
    bos_t = cc(ys[0].data.new([bos]))
    eos_t = cc(ys[0].data.new([eos]))
    ys_in = [torch.cat([bos_t,y], dim=0) for y in ys]
    ys_out = [torch.cat([y,eos_t], dim=0) for y in ys]
    xs = pad_list(ys, pad_value=pad)
    ys_in = pad_list(ys_in, pad_value=pad)
    ys_out = pad_list(ys_out, pad_value=pad)
    return xs, ys, ys_in, ys_out, ilens, labels

def _seq_mask(seq_len, max_len):
    '''
    output will be a tensor, 1. means not masked, 0. means masked
    '''
    seq_len = torch.from_numpy(np.array(seq_len)) # batch of length
    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len).long() # [0,1,2,...,max_len]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) 
    # seq_range_expand is batch of [0,1,2,...max_len]
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_len_expand).float()

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def adjust_learning_rate(optimizer, lr):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr

def remove_pad_eos(sequences, eos=2):
    sub_sequences = []
    for sequence in sequences:
        try:
            eos_index = next(i for i, v in enumerate(sequence) if v == eos)
        except StopIteration:
            eos_index = len(sequence)
        sub_sequence = sequence[:eos_index]
        sub_sequences.append(sub_sequence)
    return sub_sequences

def to_sents(index_seq, vocab, non_lang_syms):
    word_list = idx2word(index_seq, vocab, non_lang_syms)
    sentences = word_list_2_str(word_list)
    return sentences

def idx2word(sequences, vocab, non_lang_syms):
    inverse_vocab = {v:k for k,v in vocab.items()}
    non_lang_syms_idx = [vocab[sym] for sym in non_lang_syms]
    output_seqs = []
    for sequence in sequences:
        output_seq = [inverse_vocab[idx] for idx in sequence if idx not in non_lang_syms_idx]
        output_seqs.append(output_seq)
    return output_seqs

def word_list_2_str(word_lists):
    sentences = []
    for word_list in word_lists:
        sentence = ' '.join([w if w != ' ' else '' for w in word_list])
        sentences.append(sentence)
    return sentences

def calculate_wer(hyps, refs):
    total_dis, total_len = 0., 0.
    for hyp, ref in zip(hyps, refs):
        dis = editdistance.eval(hyp, ref)
        total_dis += dis
        total_len += len(ref)
    return total_dis/total_len

def get_enc_context(enc_outputs, enc_lens):
    return torch.gather(enc_outputs,1,(enc_lens-1).view(-1,1).unsqueeze(2).repeat(1,1,enc_outputs.size(2))).squeeze(1)

def get_prediction_length(predictions, eos=2):
    ilen = []
    for prediction in predictions:
        try:
            eos_index = next(i for i, v in enumerate(prediction) if v == eos)
        except StopIteration:
            eos_index = len(prediction)
        ilen.append(eos_index)
    return cc(torch.LongTensor(ilen))
