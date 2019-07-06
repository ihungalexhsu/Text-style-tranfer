import pickle
import os
import sys

pos_gt = list()
with open('positive_words.txt', 'r') as f:
    for lines in f.readlines():
        pos_gt.append(lines.strip('\n').strip())
neg_gt = list()
with open('negative_words.txt', 'r') as f:
    for lines in f.readlines():
        neg_gt.append(lines.strip('\n').strip())

pos_gen_file = sys.argv[1]
neg_gen_file = sys.argv[2]
pos_gen = pickle.load(open(pos_gen_file,'rb'))
neg_gen = pickle.load(open(neg_gen_file,'rb'))

count_pos_correct = 0
for w in pos_gen:
    if w in pos_gt:
        count_pos_correct+=1

count_neg_correct = 0
for w in neg_gen:
    if w in neg_gt:
        count_neg_correct+=1

print('positive word accuracy : ', float(count_pos_correct)/len(pos_gen))
print('negative word accuracy : ', float(count_neg_correct)/len(neg_gen))

