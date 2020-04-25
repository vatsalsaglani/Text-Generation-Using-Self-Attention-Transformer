import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import re 
import os 
from torch.utils.data import Dataset 
import numpy as np


class GeneratorDataset(Dataset):

    def __init__(self, data_array, seq_length):

        self.data_array = data_array
        self.seq_length = seq_length
        self.total_words = len(self.data_array)
        self.req_size = self.total_words - self.seq_length - 1

    def __len__(self):

        return self.req_size

    def __getitem__(self, ix):

        inp_seq = torch.from_numpy(np.array(self.data_array[ix:ix+self.seq_length]))
        op_seq = torch.from_numpy(np.array(self.data_array[ix+1:ix+self.seq_length+1]))

        return {'input': inp_seq.long(), 'output': op_seq}