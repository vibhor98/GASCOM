import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
from sentence_transformers.util import fullname, import_from_string

class MultiHeadAttention(nn.Module):

    def __init__(self, in_features: int, num_heads: int, batch_first: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(in_features, num_heads, batch_first=batch_first)

    def forward(self, Q, K, V):
        return self.multihead_attn(Q, K, V)

    def get_sentence_embedding_dimension(self) -> int:
        return self.in_features

    def get_config_dict(self):
        return {'in_features': self.in_features, 'num_heads': self.num_heads}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def __repr__(self):
        return "MultiHeadAttention({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = MultiHeadAttention(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
