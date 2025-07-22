import torch
import combnet
import librosa

import torchaudio
import warnings

import numpy as np
from typing import Union, Optional
from numpy.typing import DTypeLike

from madmom.audio.filters import LogarithmicFilterbank

def bins_to_freqs(bins):
    return bins * combnet.SAMPLE_RATE / combnet.N_FFT

def _create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    import pdb; pdb.set_trace()
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb

def tonal_filters(
    n_freqs: int,
    midi_min: float,
    midi_max: float,
    sample_rate: int,
    step: int = 0.5
) -> torch.Tensor:
    """Create a frequency bin conversion matrix based on https://arxiv.org/pdf/1706.02921
    code is adapted from torchaudio melscale_fbank implementation"""

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # m_min = librosa.hz_to_midi(f_min)
    # m_max = librosa.hz_to_midi(f_max)

    m_pts = torch.arange(midi_min-step, midi_max+step, step)
    f_pts = torch.from_numpy(librosa.midi_to_hz(m_pts))

    # create filterbank
    fb = torchaudio.functional.functional._create_triangular_filterbank(all_freqs, f_pts)
    # fb = _create_triangular_filterbank(all_freqs, f_pts)

    return fb

def madmom_filters(bands_per_octave=24):
    return torch.tensor(LogarithmicFilterbank(
        np.linspace(0, combnet.SAMPLE_RATE // 2, combnet.N_FFT//2+1),
        num_bands=bands_per_octave,
        fmin=65,
        fmax=2100,
        unique_filters=True
    )).T

def evenly_spaced_filters(
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
    
class Sum(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(self.dim, keepdims=True)


class Break(torch.nn.Module):
    def forward(self, x):
        import pdb; pdb.set_trace()
        return x

class STFTClassifier(torch.nn.Module):
    def __init__(self, features='quartertones'):
        super().__init__()
        if features == 'quartertones':
            self.filters = tonal_filters(
                n_freqs=combnet.N_FFT//2+1,
                midi_min=36,
                midi_max=96,
                sample_rate=combnet.SAMPLE_RATE,
                step=0.5
            ).T
        elif features == 'semitones':
            self.filters = self.filters = tonal_filters(
                n_freqs=combnet.N_FFT//2+1,
                midi_min=36,
                midi_max=96,
                sample_rate=combnet.SAMPLE_RATE,
                step=1.0
            ).T
        elif features == 'chroma':
            self.filters = torch.tensor(librosa.filters.chroma(
                sr=combnet.SAMPLE_RATE,
                n_fft=combnet.N_FFT,
                n_chroma=12
            ))
        elif features == '105':
            self.filters = evenly_spaced_filters(
                n_freqs=combnet.N_FFT//2+1,
                f_min=65,
                f_max=2100,
                n_bins=105,
                sample_rate=combnet.SAMPLE_RATE,
        ).T
        elif features == 'madmom':
            self.filters = madmom_filters()
        else:
            raise ValueError(f'Unknown features: {features}')

        self.layers = torch.nn.Sequential(
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

            torch.nn.Flatten(1, 2),
            Permute(0, 2, 1),

            torch.nn.Linear(self.filters.shape[0] * 8, 48),
            torch.nn.ELU(),

            Permute(0, 2, 1),

            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ELU(),
            torch.nn.Flatten(1, 2),

            torch.nn.Linear(48, 24),
            torch.nn.Softmax(dim=1)
        )
        self.window = torch.hann_window(combnet.WINDOW_SIZE)

    def to(self, device):
        self.window = self.window.to(device)
        self.filters = self.filters.to(device)
        return super().to(device)

    def _extract_features(self, spectrogram):
        features = self.filters @ spectrogram
        features = torch.log(1+features)
        return features

    def forward(self, spectrogram):
        features = self._extract_features(spectrogram)
        return self.layers(features.unsqueeze(1))