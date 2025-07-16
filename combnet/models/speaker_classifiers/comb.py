import torch
from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
import combnet
import numpy as np
import math
import copy

class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)
    

class Break(torch.nn.Module):
    def forward(self, x):
        breakpoint()
        return x

class Abs(torch.nn.Module):
    def forward(self, x):
        return abs(x)


# TODO replace hardcoded values
class CombClassifier(torch.nn.Module):
    def __init__(self, n_filters, window_size=3, stride=3, comb_kwargs={}, input_layernorm=False, comb_abs=False):
        super().__init__()

        # comb_kwargs = copy.deepcopy(comb_kwargs)

        # if 'stride' in comb_kwargs:
        #     stride = comb_kwargs['stride']
        #     del comb_kwargs['stride']
        # else:
        #     stride = 3
        # if 'window_size' in comb_kwargs:
        #     window_size = comb_kwargs['window_size']
        #     del comb_kwargs['window_size']
        # else:
        #     window_size = stride

        dims = [3200]
        dims.append(
            # int((dims[-1]-(251-1)-1)/stride+1)
            # int((dims[-1]-window_size+1)//stride)
            int((dims[-1]-window_size)//stride+1)
        )
        dims.append(
            (dims[-1]-5+1)//3
        )
        dims.append(
            (dims[-1]-5+1)//3
        )
        dims.append(
            dims[-1]*60
        )
        dims.append(
            2048
        )
        dims.append(
            2048
        )
        dims.append(
            2048
        )
        print(dims)

        ln = torch.nn.LayerNorm(dims.pop(0))
        if not input_layernorm:
            ln = torch.nn.Identity()

        ab = torch.nn.Identity()
        if comb_abs:
            ab = Abs()

        self.layers = torch.nn.Sequential(
            ln,

            # FusedComb1d(1, n_filters, sr=combnet.SAMPLE_RATE, window_size=window_size, stride=stride, last_stride=False, **comb_kwargs),
            Comb1d(1, n_filters, sr=combnet.SAMPLE_RATE, **comb_kwargs),
            ab,
            # torch.nn.Conv1d(1, 80, 251),
            torch.nn.MaxPool1d(window_size, stride),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(n_filters, 60, 5),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(60, 60, 5),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            # batch x 60 x 107
            torch.nn.Flatten(1, 2),
            # batch x 6240

            # "DNN1" as per the original implementation
            torch.nn.LayerNorm(dims[0]),

            torch.nn.Linear(dims.pop(0), 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(2048, 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(2048, 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            # "DNN2" as per the original implementation
            torch.nn.Linear(2048, len(combnet.CLASS_MAP)),
            # torch.nn.Softmax(-1)
        )

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.layers[1].f]
        groups['main'] = list(self.layers[2:].parameters()) + list(self.layers[0].parameters())
        return groups

    def forward(self, audio):
        audio = audio.unfold(-1, 3200, 160)
        audio = audio.permute(0, 2, 1, 3)
        b, f = audio.shape[0], audio.shape[1]
        audio = audio.flatten(0, 1)
        probs = self.layers(audio)
        probs = probs.unflatten(0, (b, f))
        probs = probs.mean(1)
        # probs = torch.nn.functional.softmax(probs, -1)
        return probs