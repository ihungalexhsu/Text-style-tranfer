import torch 
from torch.utils.data import Dataset
import os
import pickle 
import numpy as np

class PickleDataset(Dataset):
    def __init__(self, pickle_path, config=None, sort=True):
        with open(pickle_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        # remove the utterance out of limit
        self.keys = self.get_keys(config, sort=sort)

    def get_keys(self, config, sort):
        if config:
            max_text_length = config['max_text_length']
            min_text_length = config['min_text_length']
            keys = [key for key in self.data_dict 
                    if len(self.data_dict[key]['data']) <= max_text_length and 
                    len(self.data_dict[key]['data']) >= min_text_length]
        else:
            keys = [key for key in self.data_dict]

        # sort by length
        if sort:
            keys = sorted(keys, key=lambda x: len(self.data_dict[x]['data']),reverse=True)
        return keys

    def __getitem__(self, index):
        utt_id = self.keys[index]
        token_ids = self.data_dict[utt_id]['data']
        label = self.data_dict[utt_id]['label']
        return token_ids, label

    def __len__(self):
        return len(self.keys)
    
    def get_raw_dist(self):
        dist = dict()
        for k in self.data_dict:
            l = len(self.data_dict[k]['data'])
            if l in dist.keys():
                dist+=1
            else:
                dist[l]=1
        sorted_k = sorted(dist)
        sorted_dist = dict()
        for k in sorted_k:
            sorted_dist[k]=dist[k]
        del dist
        return sorted_dist
