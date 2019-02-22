#encoding=utf-8
import collections
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='Description: preprocessing of the style transfer data')

parser.add_argument('data_path', type=str, help='file path to data')
parser.add_argument('--dataset', dest='dataset', action='store', default='yelp',
                  type=str, choices=['yelp', 'amazon', 'imagecaption'])
parser.add_argument('--min_word_freq', dest='minfreq', action='store', type=int)
parser.add_argument('--max_vocab_number', dest='maxvocab', action='store', type=int)
parser.add_argument('--max_length', dest='maxlen', action='store', type=int)
args = parser.parse_args()
print(args)

def readfile2list(file_path):
    sentences = list()
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            sentences.append(line)
    return sentences

def get_len_distri(sentences):
    distri = collections.Counter()
    for s in sentences:
        distri[len(s.split())]+=1
    return distri

def trimlength(sentences, maxlen):
    output_sentences = list()
    for s in sentences:
        l = len(s.split())
        if l <= maxlen:
            if '_num_' in s:
                l-=1
            if ' . ' in s:
                l-=1
            if l > 1:
                output_sentences.append(s)
    return output_sentences

def build_vocab_dict(training_data, minfreq, maxvocab):
    vocab_dict = dict()
    vocab_dict['<PAD>']=0
    vocab_dict['<BOS>']=1
    vocab_dict['<EOS>']=2
    vocab_dict['<UNK>']=3
    word_count = collections.Counter()
    for s in training_data:
        ws = s.split()
        for w in ws:
            word_count[w]+=1
    # discard words that are not frequent enough
    new_word_count = collections.Counter()
    for k,v in word_count.items():
        if v >= minfreq:
            new_word_count[k]=v
    del word_count
    word_list_by_freq = new_word_count.most_common()
    while(len(word_list_by_freq) > (maxvocab-4)):
        word_list_by_freq.pop()
    cnt = 4
    for w,f in word_list_by_freq:
        vocab_dict[w]=cnt
        cnt+=1
    return vocab_dict

def writefile(structure, path, title, structure_type):
    if structure_type=='dict':
        with open(path, 'w') as f:
            f.write(title+'\n')
            for k,v in structure.items():
                f.write(str(k)+','+str(v)+'\n')
    if structure_type=='list':
        with open(path, 'w') as f:
            for s in structure:
                f.write(s+'\n')

def transform(sentences, vocab_dict):
    new_sentences = list()
    for s in sentences:
        new_ws = list()
        ws = s.split()
        for w in ws:
            if w in vocab_dict.keys():
                new_ws.append(w)
            else:
                new_ws.append('<UNK>')
        new_sentences.append(' '.join(new_ws))
    return new_sentences

def sent2vec(vocab_dict, sentence):
    output=[]
    wordvec = sentence.split()
    for w in wordvec:
        if w not in vocab_dict.keys():
            output.append(vocab_dict['<UNK>'])
        else:
            output.append(vocab_dict[w])
    return output

def savetopickle(pos_s, neg_s, pos_path, neg_path, all_path, vocab_dict):
    count=0
    pos, count = construct_dataformat(pos_s, vocab_dict, count, 1)
    neg, count = construct_dataformat(neg_s, vocab_dict, count, 0)
    total = {**pos, **neg}
    pickle.dump(pos,open(pos_path,'wb'))
    pickle.dump(neg,open(neg_path,'wb'))
    pickle.dump(total,open(all_path,'wb'))

def construct_dataformat(sentences, vocab_dict, idxstart, label):
    out = dict()
    for s in sentences:
        out[idxstart]={
            'data': sent2vec(vocab_dict, s),
            'label': label 
        }
        idxstart+=1
    return out, idxstart

def process_data(data_folder, store_folder, func_type, trim_or_not=True, vocab_dict=None):
    # read data
    if not os.path.exists(store_folder+'/picklefiles'):
        os.makedirs(store_folder+'/picklefiles')
    pstore_folder = os.path.join(store_folder, 'picklefiles')
    filenamelist = type2filename[func_type]
    pos_list = readfile2list(os.path.join(data_folder, filenamelist[0]))
    neg_list = readfile2list(os.path.join(data_folder, filenamelist[1]))
    # delete duplicate
    if args.dataset != 'imagecaption':
        pos_list = list(set(pos_list))
        neg_list = list(set(neg_list))
    total_list = pos_list+neg_list
    total_distri = get_len_distri(total_list)
    print ("before trimming distri: ", total_distri)
    
    if trim_or_not:
        pos_list = trimlength(pos_list, args.maxlen)
        neg_list = trimlength(neg_list, args.maxlen)
        total_list = pos_list+neg_list
        total_distri = get_len_distri(total_list)
        print ("after trimming distri: ", total_distri)
    
    if vocab_dict is None:
        vocab_dict = build_vocab_dict(total_list, args.minfreq, args.maxvocab)
        pickle.dump(vocab_dict, open(os.path.join(pstore_folder,'vocab_dict.p'),'wb'))
        writefile(vocab_dict, os.path.join(store_folder,'vocab2idx.txt'),'word,idx','dict')
    
    proc_pos = transform(pos_list, vocab_dict)
    proc_neg = transform(neg_list, vocab_dict)
    # save to file
    writefile(proc_pos, os.path.join(store_folder, 'proc.'+filenamelist[0]),'','list')
    writefile(proc_neg, os.path.join(store_folder, 'proc.'+filenamelist[1]),'','list')
    savetopickle(proc_pos, proc_neg,
                 os.path.join(pstore_folder, 'pos_'+func_type+'.p'),
                 os.path.join(pstore_folder, 'neg_'+func_type+'.p'),
                 os.path.join(pstore_folder, func_type+'.p'),
                 vocab_dict)
    return vocab_dict

def process_human_ref(data_folder, store_folder, vocab_dict):
    filenamelist = type2filename['ref']
    pos_file = readfile2list(os.path.join(data_folder, filenamelist[0]))
    neg_file = readfile2list(os.path.join(data_folder, filenamelist[1]))
    pos_input, pos_gt = splitref(pos_file)
    neg_input, neg_gt = splitref(neg_file)
    writefile(pos_input, os.path.join(store_folder, filenamelist[0]+'.input'),'','list')
    writefile(pos_gt, os.path.join(store_folder, filenamelist[0]+'.humanout'),'','list')
    writefile(neg_input, os.path.join(store_folder, filenamelist[1]+'.input'),'','list')
    writefile(neg_gt, os.path.join(store_folder, filenamelist[1]+'.humanout'),'','list')
    pstore_folder = os.path.join(store_folder, 'picklefiles')
    savetopickle(pos_input, neg_input,
                 os.path.join(pstore_folder, 'reference.1.p'),
                 os.path.join(pstore_folder, 'reference.0.p'),
                 os.path.join(pstore_folder, 'reference.all.p'),
                 vocab_dict)

def splitref(sentences):
    inputs = list()
    ground_truth = list()
    for sens in sentences:
        ss = sens.lower().split('\t')
        inputs.append(' '.join(ss[0].split()))
        ground_truth.append(' '.join(ss[1].split()))
    return inputs, ground_truth

type2filename={
    'train': ['sentiment.train.1','sentiment.train.0'],
    'dev': ['sentiment.dev.1','sentiment.dev.0'],
    'test': ['sentiment.test.1', 'sentiment.test.0'],
    'ref': ['reference.1', 'reference.0']
}

if __name__=="__main__":
    data_folder = os.path.join(args.data_path, args.dataset)
    if not os.path.exists('./'+args.dataset):
        os.makedirs('./'+args.dataset)
    store_folder = os.path.join('./', args.dataset)
    vocab_dict = process_data(data_folder, store_folder, 'train', True, None)
    process_data(data_folder, store_folder, 'dev', True, vocab_dict)
    process_data(data_folder, store_folder, 'test', False, vocab_dict)
    pstore_folder = os.path.join(store_folder, 'picklefiles')
    non_lang_syms=['<PAD>','<BOS>','<EOS>']
    pickle.dump(non_lang_syms,open(os.path.join(pstore_folder,'non_lang_syms.p'),'wb'))
    process_human_ref(data_folder, store_folder, vocab_dict)
