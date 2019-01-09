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

def to_gpu(data):
    xs, ilens, ys, spks, envs, trans = data
    xs = cc(xs)
    ilens = cc(torch.LongTensor(ilens))
    ys = [cc(y) for y in ys]
    spks = cc(torch.IntTensor(spks))
    envs = cc(torch.IntTensor(envs))
    return xs, ilens, ys, spks, envs, trans

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
