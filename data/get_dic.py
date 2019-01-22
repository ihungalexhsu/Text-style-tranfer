#encoding=utf-8

import pickle as pkl
import collections
import os


file_name=['raw/q_train.txt']
count = dict()
for i in file_name:
    with open(i,'r') as f:
        for line in f.readlines():
            line=line.split()
            for line_per in line:
                if line_per in count.keys():
                    count[line_per]+=1
                else:
                    count[line_per]=1

count = sorted(count.items(),key=lambda x:x[1], reverse=True)
print (len(count))
count_write = collections.OrderedDict()
count_write['<PAD>']=0
count_write['<BOS>']=1
count_write['<EOS>']=2
count_write['<UNK>']=3
index=4
for i in count:
    count_write[i[0]]=index
    index+=1
    if index>=60000:
        break

f = open(r"dict.p",'wb')
pkl.dump(count_write,f)
f.close()
non_lang_syms=['<PAD>','<BOS>','<EOS>']
pkl.dump(non_lang_syms,open('non_lang_syms.p','wb'))

