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
    def __init__(self, kernel_size=16, n_channels=32, stride=4):
        super().__init__()
        n_classes = len(combnet.CLASS_MAP)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, n_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            torch.nn.GELU(),
            torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(1, 2),
            torch.nn.Linear(n_channels, n_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_channels, n_classes),
            torch.nn.Softmax(dim=1),
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