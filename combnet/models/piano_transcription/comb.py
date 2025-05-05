import torch
import combnet

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
    def __init__(self, n_filters=12, fused_comb_fn='FusedComb1d', comb_fn=None, comb_kwargs={}):
        super().__init__()
        if comb_fn is None:
            comb = getattr(combnet.modules, fused_comb_fn)(
                1, n_filters, sr=combnet.SAMPLE_RATE, window_size=combnet.WINDOW_SIZE, stride=combnet.HOPSIZE,
                **comb_kwargs
            )
        else:
            assert fused_comb_fn is None
            comb = getattr(combnet.modules, comb_fn)(
                1, n_filters, sr=combnet.SAMPLE_RATE,
                **comb_kwargs
            )
        n_classes = 12
        print(n_filters)
        self.layers = torch.nn.Sequential(
            comb,
            torch.nn.ELU(),
            # torch.nn.Flatten(1, 2),
            torch.nn.Conv1d(n_filters, n_filters, 1),
            # torch.nn.Sigmoid(),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_filters, n_classes, 1),
            # torch.nn.Sigmoid(),
        )
        self.train()

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.layers[0].f]
        groups['main'] = list(self.layers[1:].parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, audio):
        audio = torch.nn.functional.pad(audio, (combnet.WINDOW_SIZE//2, combnet.WINDOW_SIZE//2))
        return self.layers(audio)