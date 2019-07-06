import pickle
import copy
import torch

vocab = pickle.load(open('../../data/yelp/picklefiles/vocab_dict.p','rb'))
pos_salience = pickle.load(open('yelp_stanford/yelp.salience.tf_idf.1.p','rb'))
neg_salience = pickle.load(open('yelp_stanford/yelp.salience.tf_idf.0.p','rb'))
pos_word = dict()
neg_word = dict()

for k,v in pos_salience:
    if v >= 15.0:
        pos_word[k]=v
    '''
    if v >= 9.0:
        pos_word[k]=v
    elif k=='good':
        pos_word[k]=v
    elif k=='like':
        pos_word[k]=v
    '''

for k,v in neg_salience:
    if v >= 15.0:
        neg_word[k]=v
    '''
    if v >= 10.0:
        neg_word[k]=v
    '''

def create_wordlist(salience):
    output = list()
    for phrase in salience.keys():
        ws = phrase.split(' ')
        for w in ws:
            if w not in output:
                output.append(w)
    return output

pos_word_list = create_wordlist(pos_word)
pickle.dump(pos_word_list, open('yelp_stanford/pos_word_list.p','wb'))
with open('yelp_stanford/pos_word_list.txt','w') as f:
    for l in pos_word_list:
        f.write(l)
        f.write('\n')

neg_word_list = create_wordlist(neg_word)
pickle.dump(neg_word_list, open('yelp_stanford/neg_word_list.p','wb'))
with open('yelp_stanford/neg_word_list.txt','w') as f:
    for l in neg_word_list:
        f.write(l)
        f.write('\n')

def highest_ngram(word_list, selected_words, n_gram=4):
    output = None
    max_value = 0.
    for n in range(1,n_gram):
        for idx in range(0, len(word_list)-n+1):
            tmp = ' '.join(word_list[idx:idx+n])
            if tmp in selected_words.keys():
                if selected_words[tmp] > max_value:
                    max_value = selected_words[tmp]
                    output=(idx, n)
    return output

def process_data(data, selected_words, vocab, label):
    output_dict= dict()
    out_idx = 0
    masked_list = list()
    for sents in data:
        word_list = sents.split(' ')
        mask = [0]*len(word_list)
        # create n-gram
        for n in range(1,5):
            for l in range(0, len(word_list)-n+1):
                tmp=' '.join(word_list[l:l+n])
                if tmp in selected_words:
                    mask[l:l+n]=[1]*n
        data = {
            'data': [vocab[w] if w in vocab.keys() else vocab['<UNK>'] for w in word_list],
            'label': label,
            'align': torch.ByteTensor(mask),
            'words': sents
        }
        output_dict[out_idx]=data
        out_idx+=1
        temp = list()
        for m,w in zip(mask, word_list):
            if m:
                temp.append('__')
            else:
                temp.append(w)
        masked_list.append(' '.join(temp))
    return output_dict, masked_list

def key_func(ele):
    return(len(ele.split(' ')))

pos_train = list()
with open('../../data/yelp/proc.sentiment.train.1', 'r') as f:
    for lines in f.readlines():
        pos_train.append(lines.strip('\n').strip())
post_pos_train, _ = process_data(pos_train, pos_word, vocab, 1)

pickle.dump(post_pos_train, open('yelp_stanford/pos_train_stanfordmask.p','wb'))
del pos_train
del post_pos_train

neg_train = list()
with open('../../data/yelp/proc.sentiment.train.0', 'r') as f:
    for lines in f.readlines():
        neg_train.append(lines.strip('\n').strip())
post_neg_train, _ = process_data(neg_train, neg_word, vocab, 0)
pickle.dump(post_neg_train, open('yelp_stanford/neg_train_stanfordmask.p','wb'))
del neg_train
del post_neg_train

pos_dev = list()
with open('../../data/yelp/proc.sentiment.dev.1', 'r') as f:
    for lines in f.readlines():
        pos_dev.append(lines.strip('\n').strip())
post_pos_dev, masked_list = process_data(pos_dev, pos_word, vocab, 1)
pickle.dump(post_pos_dev, open('yelp_stanford/pos_dev_stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)

with open('yelp_stanford/pos_dev_with_StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del pos_dev
del post_pos_dev

neg_dev = list()
with open('../../data/yelp/proc.sentiment.dev.0', 'r') as f:
    for lines in f.readlines():
        neg_dev.append(lines.strip('\n').strip())
post_neg_dev, masked_list = process_data(neg_dev, neg_word, vocab, 0)
pickle.dump(post_neg_dev, open('yelp_stanford/neg_dev_stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)
with open('yelp_stanford/neg_dev_with_StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del neg_dev
del post_neg_dev

pos_test = list()
with open('../../data/yelp/proc.sentiment.test.1', 'r') as f:
    for lines in f.readlines():
        pos_test.append(lines.strip('\n').strip())
post_pos_test, masked_list = process_data(pos_test, pos_word, vocab, 1)
pickle.dump(post_pos_test, open('yelp_stanford/pos_test_stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)
with open('yelp_stanford/pos_test_with_StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del pos_test
del post_pos_test

neg_test = list()
with open('../../data/yelp/proc.sentiment.test.0', 'r') as f:
    for lines in f.readlines():
        neg_test.append(lines.strip('\n').strip())
post_neg_test, masked_list = process_data(neg_test, neg_word, vocab, 0)
pickle.dump(post_neg_test, open('yelp_stanford/neg_test_stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)
with open('yelp_stanford/neg_test_StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del neg_test
del post_neg_test

pos_test = list()
with open('../../data/yelp/reference.1.input', 'r') as f:
    for lines in f.readlines():
        pos_test.append(lines.strip('\n').strip())
post_pos_test, masked_list = process_data(pos_test, pos_word, vocab, 1)
pickle.dump(post_pos_test, open('yelp_stanford/reference.1.stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)
with open('yelp_stanford/reference.1.StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del pos_test
del post_pos_test

neg_test = list()
with open('../../data/yelp/reference.0.input', 'r') as f:
    for lines in f.readlines():
        neg_test.append(lines.strip('\n').strip())
post_neg_test, masked_list = process_data(neg_test, neg_word, vocab, 0)
pickle.dump(post_neg_test, open('yelp_stanford/reference.0.stanfordmask.p','wb'))
sorted_list = sorted(masked_list, key=key_func, reverse=True)
with open('yelp_stanford/reference.0.StanMaskOut.txt','w') as f:
    for l in sorted_list:
        f.write(l)
        f.write('\n')
del neg_test
del post_neg_test

