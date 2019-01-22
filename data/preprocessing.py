#encoding=utf-8

import pickle
import collections
import os

vocab_dict = pickle.load(open('dict.p','rb'))
def sent2vec(vocab_dict, sentence):
    output=[]
    wordvec = sentence.split()
    for w in wordvec:
        if w not in vocab_dict.keys():
            output.append(vocab_dict['<UNK>'])
        else:
            output.append(vocab_dict[w])
    return output

def process_data(d_path,l_path,store_path_pos,store_path_neg):
    data = []
    with open(d_path,'r') as fdata:
        for line in fdata.readlines():
            data.append(' '.join(line.split('\n')).strip())
    label = []
    with open(l_path,'r') as flabel:
        for line in flabel.readlines():
            label.append(''.join(line.split('\n')).strip())
    pos=dict()
    neg=dict()
    total=dict()
    cnt_pos=0
    cnt_neg=0
    for i in range(len(data)):
        total[i]={
            'data':sent2vec(vocab_dict,data[i]),
            'label':int(label[i])
        }
        if label[i]=='0':
            neg[cnt_neg]={
                'data':sent2vec(vocab_dict,data[i]),
                'label':int(label[i])
            }
            cnt_neg+=1
        elif label[i]=='1':
            pos[cnt_pos]={
                'data':sent2vec(vocab_dict,data[i]),
                'label':int(label[i])
            }
            cnt_pos+=1
    #pickle.dump(pos,open(store_path_pos,'wb'))
    #pickle.dump(neg,open(store_path_neg,'wb'))
    pickle.dump(total,open(store_path_pos,'wb'))

process_data('raw/q_train.txt','raw/s_train.txt','train.p','neg_train.p')
process_data('raw/q_val.txt','raw/s_val.txt','val.p','neg_val.p')
process_data('raw/q_test.txt','raw/s_test.txt','test.p','neg_test.p')
