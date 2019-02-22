#encoding=utf-8
import collections
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='Description: this file mains to \
                                 generate data format that fit fastText usage')

parser.add_argument('data_path', type=str, help='file path to preprocessed data')
parser.add_argument('--dataset', dest='dataset', action='store', default='yelp',
                  type=str, choices=['yelp', 'amazon', 'imagecaption'])
args = parser.parse_args()
print(args)

def readfile2list(file_path):
    sentences = list()
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            sentences.append(line)
    return sentences

def process_data(data_folder, func_type):
    pos = readfile2list(os.path.join(data_folder, 'proc.sentiment.'+func_type+'.1'))
    neg = readfile2list(os.path.join(data_folder, 'proc.sentiment.'+func_type+'.0'))
    output = list()
    for s in pos:
        output.append("__label__1 "+s)
    for s in neg:
        output.append("__label__0 "+s)
    return output

def writefile(store_folder, filename, structure):
    with open(os.path.join(store_folder, filename), 'w') as f:
        for s in structure:
            f.write(s)

if __name__=="__main__":
    data_folder = os.path.join(args.data_path, args.dataset)
    if not os.path.exists('./'+args.dataset):
        os.makedirs('./'+args.dataset)
    store_folder = os.path.join('./', args.dataset)
    train = process_data(data_folder, 'train')
    dev = process_data(data_folder, 'dev')
    test = process_data(data_folder, 'test')
    writefile(store_folder, 'fastText.train', train)
    writefile(store_folder, 'fastText.dev', dev)
    writefile(store_folder, 'fastText.test', test)
