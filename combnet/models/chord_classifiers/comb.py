import torch
# from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
import combnet

# from madmom.audio.filters import LogarithmicFilterbank

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

# class CombClassifier(torch.nn.Module):
#     def __init__(self, n_filters=12, comb_fn='Comb1d'):
#         comb_fn = getattr(combnet.modules, comb_fn)
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             comb_fn(1, n_filters),
#             torch.nn.MaxPool1d(combnet.WINDOW_SIZE, combnet.HOPSIZE, combnet.WINDOW_SIZE//2),
#             Unsqueeze(1),

#             torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2)),
#             torch.nn.ELU(),

#             torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
#             torch.nn.ELU(),

#             torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
#             torch.nn.ELU(),

#             torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
#             torch.nn.ELU(),

#             torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)), 
#             torch.nn.ELU(),

#             #TODO figure out how they got down to just 1 channel? This might be it, but better to double check...
#             torch.nn.Flatten(1, 2),
#             Permute(0, 2, 1),

#             torch.nn.Linear(n_filters*8, 48),
#             torch.nn.ELU(),

#             Permute(0, 2, 1),

#             torch.nn.AdaptiveAvgPool1d(1),
#             torch.nn.ELU(),
#             torch.nn.Flatten(1, 2),

#             torch.nn.Linear(48, 24),
#             torch.nn.Softmax(dim=1)
#         )

#     def parameter_groups(self):
#         groups = {}
#         groups['f0'] = [self.layers[0].f]
#         groups['main'] = list(self.layers[1:].parameters()) + [self.layers[0].a] + [self.layers[0].g]
#         return groups

#     def forward(self, audio):
#         return self.layers(audio)


class CombClassifier(torch.nn.Module):
    def __init__(self, n_filters=12, fused_comb_fn='FusedComb1d', comb_fn=None):
        super().__init__()
        if comb_fn is None:
            comb = getattr(combnet.modules, fused_comb_fn)(
                1, n_filters, sr=combnet.SAMPLE_RATE, window_size=combnet.WINDOW_SIZE, stride=combnet.HOPSIZE
            )
        else:
            assert fused_comb_fn is None
            comb = getattr(combnet.modules, comb_fn)(
                1, n_filters, sr=combnet.SAMPLE_RATE
            )
        # centers = None
        # import numpy as np
        # centers = torch.tensor(LogarithmicFilterbank(
        #     np.linspace(0, combnet.SAMPLE_RATE // 2, combnet.N_FFT//2+1),
        #     num_bands=12,
        #     fmin=65,
        #     fmax=2100,
        #     unique_filters=True
        # ).center_frequencies, dtype=torch.float32)
        n_classes = len(combnet.CLASS_MAP)
        print(n_filters)
        self.layers = torch.nn.Sequential(
            comb,
            torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(1, 2),
            torch.nn.Linear(n_filters, n_filters),
            torch.nn.ReLU(),
            torch.nn.Linear(n_filters, n_classes),
            torch.nn.Softmax(dim=1),
        )
        self.train()
        # self.layers[0].f.data = centers[:n_filters, None]
        torch.linspace(200, 500, self.layers[0].f.data.shape[0])[:, None]

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.layers[0].f]
        groups['main'] = list(self.layers[1:].parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, audio):
        return self.layers(audio)