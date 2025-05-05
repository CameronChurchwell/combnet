import torch
from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
import combnet

from madmom.audio.filters import LogarithmicFilterbank

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
    def __init__(self, n_filters=12, fused_comb_fn='FusedComb1d', n_conv_layers=5, linear_channels=None, comb_kwargs={}):
        super().__init__()
        fused_comb_fn = getattr(combnet.modules, fused_comb_fn)
        centers = None
        import numpy as np
        centers = torch.tensor(LogarithmicFilterbank(
            np.linspace(0, combnet.SAMPLE_RATE // 2, combnet.N_FFT//2+1),
            num_bands=24,
            fmin=65,
            fmax=2100,
            unique_filters=True
        ).center_frequencies, dtype=torch.float32)
        self.filters = torch.nn.Sequential(
            fused_comb_fn(1, n_filters, sr=combnet.SAMPLE_RATE, window_size=combnet.WINDOW_SIZE, stride=combnet.HOPSIZE,
                **comb_kwargs
            ),
        )
        if 'min_freq' not in comb_kwargs:
            self.filters[0].f.data = centers[:n_filters, None]
            if 'alpha' in comb_kwargs and comb_kwargs['alpha'] < 0:
                self.filters[0].f.data *= 2
        # if static_filters:
        #     # DEBUG
        #     # self.filters[0] = comb_fn(1, 2)
        #     # self.filters[0].f.data = torch.tensor([151., 371.])[:, None]
        #     # \DEBUG

        #     self.filters[0].train(False)
        #     self.filters[0].train = lambda s, b=None: s

        # self.filters[0].f.data *= 2

        self.train()
        # activation = torch.nn.ReLU
        activation = torch.nn.ELU

        if linear_channels is None:
            linear_channels = n_filters * 8

        self.layers = torch.nn.Sequential(*([
            torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2)),
            activation(),
        ] + sum([[
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            activation(),
        ] for _ in range(1, n_conv_layers)], start=[]) + [
            # Sum(1),
            # Break(),
            torch.nn.Flatten(1, 2),
            Permute(0, 2, 1),
            # Permute(0, 1, 3, 2),

            torch.nn.Linear(linear_channels, 48),
            activation(),

            Permute(0, 2, 1),
            # Permute(0, 1, 3, 2),
            # Sum(1),
            # torch.nn.Flatten(1, 2),

            torch.nn.AdaptiveAvgPool1d(1),
            activation(),
            torch.nn.Flatten(1, 2),

            torch.nn.Linear(48, 24),
            torch.nn.Softmax(dim=1)
        ]))
        self.window = torch.hann_window(combnet.WINDOW_SIZE)

    def to(self, device):
        self.window = self.window.to(device)
        self.filters = self.filters.to(device)
        return super().to(device)

    def _extract_features(self, audio):
        # features = self.filters @ spectrogram
        # features = torch.log(1+features)
        features = self.filters(audio)
        # features = torch.log(1+features)
        # features = features / abs(features).max()
        # features = features * 200
        # return torch.log(1+features)
        if combnet.COMB_ACTIVATION is not None:
            features = combnet.COMB_ACTIVATION(features)
        return features

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.filters[0].f]
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, audio):
        features = self._extract_features(audio)
        return self.layers(features.unsqueeze(1))


class CombLinearClassifier(CombClassifier):
    def __init__(self, n_filters=1024, n_input_channels=64, fused_comb_fn='FusedComb1d', n_conv_layers=5, comb_kwargs={}, linear_bias=True, linear_init=None):
        super().__init__(
            n_filters=n_filters,
            fused_comb_fn=fused_comb_fn,
            n_conv_layers=n_conv_layers,
            linear_channels=n_input_channels*8, # because n_filters is not the input size to the model, n_input_channels is.
            comb_kwargs=comb_kwargs,
        )
        # self.layers[0] = torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2))
        self.linear = torch.nn.Conv1d(n_filters, n_input_channels, 1, bias=linear_bias)
        if linear_init == 'identity':
            assert n_filters == n_input_channels
            self.linear.weight.data = torch.eye(n_input_channels).to(self.linear.weight.device)[..., None]

    # def to(self, device):
    #     self.window = self.window.to(device)
    #     self.filters = self.filters.to(device)
    #     return super().to(device)

    def _extract_features(self, audio):
        # features = self.filters @ spectrogram
        # features = torch.log(1+features)
        features = self.filters(audio)
        features = self.linear(features)

        # features = torch.log(1+features)
        # features = features / abs(features).max()
        # features = features * 200
        # return torch.log(1+features)
        if combnet.COMB_ACTIVATION is not None:
            features = combnet.COMB_ACTIVATION(features)
        else:
            features = torch.nn.functional.sigmoid(features)
        return features

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.filters[0].f]
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g
        groups['filters'] = list(self.linear.parameters())
        return groups

    def forward(self, audio):
        features = self._extract_features(audio)
        return self.layers(features.unsqueeze(1))