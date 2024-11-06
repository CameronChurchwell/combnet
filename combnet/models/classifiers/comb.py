import torch
from combnet.modules import Comb1d, CombInterference1d


K = 12
# K = 20
C = 2
class CombClassifier(torch.nn.Module):

    def __init__(self, comb_fn=Comb1d, n_classes=2, n_filters=K):
        super().__init__()
        self.layers = torch.nn.Sequential(
            comb_fn(1, n_filters, alpha=0.85, learn_alpha=False),
            torch.nn.MaxPool1d(512, 256, 256),
            torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(1, 2),
            # torch.nn.ReLU(),
            torch.nn.Linear(n_filters, n_classes),
            torch.nn.Softmax(1)
        )

    def forward(self, x):
        return self.layers(x)