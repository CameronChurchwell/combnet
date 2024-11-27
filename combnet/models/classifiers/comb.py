import torch
from combnet.modules import Comb1d, CombInterference1d
import combnet

# K = 12
# # K = 20
# C = 2
# class CombClassifier(torch.nn.Module):

#     def __init__(self, comb_fn=Comb1d, n_classes=2, n_filters=K):
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             comb_fn(1, n_filters, alpha=0.85, learn_alpha=False),
#             torch.nn.MaxPool1d(512, 256, 256),
#             torch.nn.AdaptiveMaxPool1d(1),
#             torch.nn.Flatten(1, 2),
#             # torch.nn.ReLU(),
#             torch.nn.Linear(n_filters, n_classes),
#             torch.nn.Softmax(1)
#         )

#     def forward(self, x):
#         return self.layers(x)

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

class CombClassifier(torch.nn.Module):
    def __init__(self, n_filters=12, comb_fn='Comb1d'):
        comb_fn = getattr(combnet.modules, comb_fn)
        super().__init__()
        self.layers = torch.nn.Sequential(
            comb_fn(1, n_filters),
            torch.nn.MaxPool1d(combnet.WINDOW_SIZE, combnet.HOPSIZE, combnet.WINDOW_SIZE//2),
            Unsqueeze(1),

            torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), (2, 2)), 
            torch.nn.ELU(),

            #TODO figure out how they got down to just 1 channel? This might be it, but better to double check...
            torch.nn.Flatten(1, 2),
            Permute(0, 2, 1),

            torch.nn.Linear(n_filters, 48),
            torch.nn.ELU(),

            Permute(0, 2, 1),

            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ELU(),
            torch.nn.Flatten(1, 2),

            torch.nn.Linear(48, 24),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, audio):
        return self.layers(audio)