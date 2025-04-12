import torch
from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
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

class ConvClassifier(torch.nn.Module):
    def __init__(self, kernel_size=16, n_channels=32, stride=4):
        super().__init__()
        self.train()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, n_channels, kernel_size, stride, padding=kernel_size//2),

            Unsqueeze(1),

            torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
            torch.nn.ELU(),

            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)), 
            torch.nn.ELU(),

            # Sum(1),
            # Break(),
            torch.nn.Flatten(1, 2),
            Permute(0, 2, 1),
            # Permute(0, 1, 3, 2),

            torch.nn.Linear(n_channels * 8, 48),
            torch.nn.ELU(),

            Permute(0, 2, 1),
            # Permute(0, 1, 3, 2),
            # Sum(1),
            # torch.nn.Flatten(1, 2),

            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ELU(),
            torch.nn.Flatten(1, 2),

            torch.nn.Linear(48, 24),
            torch.nn.Softmax(dim=1)
        )
        self.window = torch.hann_window(combnet.WINDOW_SIZE)

    def parameter_groups(self):
        groups = {}
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, audio):
        return self.layers(audio)