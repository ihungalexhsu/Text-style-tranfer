import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import PickleDataset

def _collate_fn(l):
    l.sort(key=lambda x: len(x[0]), reverse=True)
    input_data = [torch.LongTensor(i) for i,_,_,_ in l]
    output_data = [torch.LongTensor(o) for _,o,_,_ in l]
    ilens = [len(i) for i,_,_,_ in l]
    labels = [label for _,_,_,label in l]
    labels = torch.LongTensor(labels)
    return input_data, output_data, ilens, labels 

def get_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn, num_workers=2)
