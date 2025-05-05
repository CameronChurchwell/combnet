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

class ConvClassifier(torch.nn.Module):
    def __init__(self, kernel_size=combnet.WINDOW_SIZE, n_channels=16):
        super().__init__()
        n_classes = 12
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, n_channels, kernel_size, stride=combnet.HOPSIZE, padding=kernel_size // 2),
            torch.nn.ELU(),
            # torch.nn.Flatten(1, 2),
            torch.nn.Conv1d(n_channels, n_channels, 1),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_channels, n_classes, 1),
            # torch.nn.Softmax(dim=1),
        )
        self.train()
        # self.layers[0].f.data = centers[:n_filters, None]
        # torch.linspace(200, 500, self.layers[0].f.data.shape[0])[:, None]

    def parameter_groups(self):
        groups = {}
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, audio):
        return self.layers(audio)