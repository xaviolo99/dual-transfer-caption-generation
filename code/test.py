import os
import pickle
from collections import namedtuple

#from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from efficientnet_pytorch import EfficientNet

class Dataset(torch.utils.data.IterableDataset):
    
    def __init__(self, a=1):
        self.a = a

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.Tensor([1])

valid_data = Dataset()#Dataset(captions, IMAGES_PATH)
valid_generator = torch.utils.data.DataLoader([1, 2, 3], num_workers=1)#, batch_size=1, shuffle=False, num_workers=1)

for paths in valid_generator:
    print(paths)
    break
