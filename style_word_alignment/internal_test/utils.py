import torch 
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn
import gensim
from gensim.models import KeyedVectors
import pickle
from nltk.tokenize import word_tokenize

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
        #return nn.DataParallel(net).cuda()
        device = torch.device('cuda')
        return net.to(device)
    else:
        device = torch.device('cpu')
        return net.to(device)

def to_gpu(data, pad):
    token_ids, ilens, labels, aligns = data
    ilens = cc(torch.LongTensor(ilens))
    labels = cc(labels)
    inputs = pad_list([cc(y) for y in token_ids], pad_value=pad)
    aligns = pad_list([cc(a) for a in aligns], pad_value=0)
    return inputs, ilens, labels, aligns

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


def get_enc_context(enc_outputs, enc_lens):
    return torch.gather(enc_outputs,1,(enc_lens-1).view(-1,1).unsqueeze(2).repeat(1,1,enc_outputs.size(2))).squeeze(1)

def get_prediction_length(predictions, eos=2):
    ilen = []
    for prediction in predictions:
        try:
            eos_index = next(i for i, v in enumerate(prediction) if v == eos)
        except StopIteration:
            eos_index = len(prediction)
        if eos_index<=0:
            eos_index=1
        ilen.append(eos_index)
    return cc(torch.LongTensor(ilen))

def writefile(structure, path):
    with open(path, 'w') as f:
        for s in structure:
            f.write(s+'\n')
    return

def loadw2vweight(w2vpath, vocabdict, pad_idx, mask_name='<MASK>'):
    model = KeyedVectors.load_word2vec_format(w2vpath, binary=False)
    modelwordlist = model.index2entity
    pretrainweight = list()
    for i,(k,v) in enumerate(vocabdict.items()):
        if i == pad_idx:
            pretrainweight.append(np.zeros((100,),dtype=np.float32))
        else:
            if k in modelwordlist:
                pretrainweight.append(model[k])
            else:
                #pretrainweight.append(np.random.randn(100,).astype(np.float32))
                pretrainweight.append(np.zeros(100,).astype(np.float32))
    # add one extra vocab for <MASK>
    pretrainweight.append(np.zeros(100,).astype(np.float32))
    mask_index = len(vocabdict)
    vocabdict[mask_name]=mask_index
    pretrainweight = np.array(pretrainweight)
    return torch.FloatTensor(pretrainweight), vocabdict

def mergew2v(w2vpath, vocabdict, pad_idx, binary=False):
    model = KeyedVectors.load_word2vec_format(w2vpath, binary=binary)
    modelwordlist = model.index2entity
    pretrainweight = list()
    for i,(k,v) in enumerate(vocabdict.items()):
        if i == pad_idx:
            pretrainweight.append(np.zeros((100,),dtype=np.float32))
        else:
            if k in modelwordlist:
                pretrainweight.append(model[k])
            else:
                #pretrainweight.append(np.random.randn(100,).astype(np.float32))
                pretrainweight.append(np.zeros(100,).astype(np.float32))
    index = len(vocabdict)
    for w in modelwordlist:
        if w in vocabdict.keys():
            pass
        else:
            vocabdict[w]=(index+1)
            pretrainweight.append(model[w])
            index+=1
    pretrainweight = np.array(pretrainweight)
    return torch.FloatTensor(pretrainweight), vocabdict

def trim_emotion_alignment(emotion_align, ilens):
    '''
    make emotion_align fit the length
    '''
    output = list()
    for idx,a in enumerate(emotion_align):
        output.append(a[:ilens[idx]])
    return output
