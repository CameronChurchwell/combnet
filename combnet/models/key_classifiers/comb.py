import torch
import combnet

from combnet.madmom import LogarithmicFilterbank

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
            torch.nn.Flatten(1, 2),
            Permute(0, 2, 1),

            torch.nn.Linear(linear_channels, 48),
            activation(),

            Permute(0, 2, 1),

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
        features = self.filters(audio)
        if combnet.COMB_ACTIVATION is not None:
            features = combnet.COMB_ACTIVATION(features)
        return features

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.filters[0].f]
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a]
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
        self.linear = torch.nn.Conv1d(n_filters, n_input_channels, 1, bias=linear_bias)
        if linear_init == 'identity':
            assert n_filters == n_input_channels
            self.linear.weight.data = torch.eye(n_input_channels).to(self.linear.weight.device)[..., None]

    def _extract_features(self, audio):
        features = self.filters(audio)
        features = self.linear(features)

        if combnet.COMB_ACTIVATION is not None:
            features = combnet.COMB_ACTIVATION(features)
        else:
            features = torch.nn.functional.sigmoid(features)
        return features

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.filters[0].f]
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a]
        groups['filters'] = list(self.linear.parameters())
        return groups

    def forward(self, audio):
        features = self._extract_features(audio)
        return self.layers(features.unsqueeze(1))