import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import PickleDataset

def _collate_fn(l):
    l.sort(key=lambda x: len(x[0]), reverse=True)
    token_ids = [torch.LongTensor(ids) for ids,_,_ in l]
    ilens = [len(ids) for ids,_,_ in l]
    labels = [label for _,label,_ in l]
    labels = torch.LongTensor(labels)
    aligns = [align for _,_,align in l]
    return token_ids, ilens, labels, aligns

def get_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn, num_workers=0)
