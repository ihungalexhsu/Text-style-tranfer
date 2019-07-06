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
                    if len(self.data_dict[key]['input']) <= max_text_length and 
                    len(self.data_dict[key]['input']) >= min_text_length]
        else:
            keys = [key for key in self.data_dict]

        # sort by length
        if sort:
            keys = sorted(keys, key=lambda x: len(self.data_dict[x]['input']), reverse=True)
        return keys

    def __getitem__(self, index):
        utt_id = self.keys[index]
        input_data = self.data_dict[utt_id]['input']
        output_data = self.data_dict[utt_id]['output']
        input_style = self.data_dict[utt_id]['input_style']
        output_style = self.data_dict[utt_id]['output_style']
        return input_data, output_data, input_style, output_style

    def __len__(self):
        return len(self.keys)
