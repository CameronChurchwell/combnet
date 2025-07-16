import torch
from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
import combnet
import numpy as np
import math

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


# TODO replace hardcoded values
class ConvClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        dims = [3200]
        dims.append(
            (dims[-1]-251)//3
        )
        dims.append(
            (dims[-1]-5)//3
        )
        dims.append(
            (dims[-1]-5)//3
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

        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(dims.pop(0)),

            torch.nn.Conv1d(1, 80, 251),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(80, 60, 5),
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
            torch.nn.LayerNorm(dims.pop(0)),

            torch.nn.Linear(6420, 2048, bias=False),
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
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
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