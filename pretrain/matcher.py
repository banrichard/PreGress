import math

from torch import nn
import torch.nn.functional as F

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''

    def __init__(self, n_hid, n_out, temperature=0.1):
        super(Matcher, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.sqrt_hd = math.sqrt(n_out)
        self.drop = nn.Dropout(0.2)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.cache = None
        self.temperature = temperature

    def forward(self, x):
        tx = F.gelu(self.linear(x))
        return tx
        # if use_norm:
        #     return self.cosine(tx, ty) / self.temperature
        # else:
        #     return (tx * ty).sum(dim=-1) / self.sqrt_hd

    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)