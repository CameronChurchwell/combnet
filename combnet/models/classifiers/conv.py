import torch
import combnet
import librosa

import torchaudio
import warnings

import numpy as np
from typing import Union, Optional
from numpy.typing import DTypeLike

# C = 2
# class ConvClassifier(torch.nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Conv1d(1, 32, 16, stride=4, padding=4),
#             torch.nn.GELU(),
#             torch.nn.Conv1d(32, 32, 16, stride=4, padding=4),
#             torch.nn.GELU(),
#             torch.nn.AdaptiveMaxPool1d(1),
#             torch.nn.Flatten(1, 2),
#             torch.nn.Linear(32, C),
#             torch.nn.Softmax(1)
#         )

#     def forward(self, x):
#         return self.layers(x)

def bins_to_freqs(bins):
    return bins * combnet.SAMPLE_RATE / combnet.N_FFT

def make_filters(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_bins: int,
    sample_rate: int
) -> torch.Tensor:
    """Create a frequency bin conversion matrix based on https://arxiv.org/pdf/1706.02921
    code is adapted from torchaudio melscale_fbank implementation"""

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = librosa.hz_to_midi(f_min)
    m_max = librosa.hz_to_midi(f_max)

    m_pts = torch.linspace(m_min, m_max, n_bins + 2)
    f_pts = torch.from_numpy(librosa.midi_to_hz(m_pts))

    # create filterbank
    fb = torchaudio.functional.functional._create_triangular_filterbank(all_freqs, f_pts)

    return fb

class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class ConvClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
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

            torch.nn.Linear(105, 48),
            torch.nn.ELU(),

            Permute(0, 2, 1),

            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ELU(),
            torch.nn.Flatten(1, 2),

            torch.nn.Linear(48, 24),
            torch.nn.Softmax(dim=1)
        )
        self.window = torch.hann_window(combnet.N_FFT)
        self.filters = make_filters(
            n_freqs=combnet.N_FFT//2+1,
            f_min=65,
            f_max=2100,
            n_bins=105,
            sample_rate=combnet.SAMPLE_RATE,
        ).T

    def to(self, device):
        self.window = self.window.to(device)
        self.filters = self.filters.to(device)
        return super().to(device)

    def forward(self, audio):
        if audio.dim() == 3:
            assert audio.shape[1] == 1
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=combnet.N_FFT,
            hop_length=combnet.HOP_LENGTH,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True)
        spec = abs(spec)
        spec = self.filters @ spec
        spec = torch.log(1+spec)

        return self.layers(spec.unsqueeze(1))