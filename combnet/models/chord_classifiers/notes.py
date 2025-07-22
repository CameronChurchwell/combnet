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

beadgcf = ["B", "E", "A", "D", "G", "C", "F"]
beadgcf_ext = beadgcf[:]
for note in beadgcf:
    beadgcf_ext.append(note +"b")

class NotesClassifier(torch.nn.Module):
    def __init__(self, note_count=3):
        super().__init__()
        n_classes = len(combnet.CLASS_MAP)
        input_size = len(beadgcf_ext)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, n_classes),
            torch.nn.Softmax(dim=1),
        )
        self.train()

    def parameter_groups(self):
        groups = {}
        groups['main'] = list(self.layers.parameters()) #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, notes):
        return self.layers(notes)