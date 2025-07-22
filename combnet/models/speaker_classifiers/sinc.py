import torch
from combnet.modules import Comb1d, CombInterference1d, FusedComb1d
import combnet
import numpy as np
import math

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

class Breakpoint(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        import pdb; pdb.set_trace()
        pass

# Code taken from (https://github.com/mravanelli/SincNet/blob/master/dnn_models.py)
# The MIT License (MIT)

# Copyright (c) 2019 Mirco Ravanelli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class SincLayer(torch.nn.Module):
    """
    SincNet Layer adapted from https://github.com/mravanelli/SincNet/blob/master/dnn_models.py
    All rights to original authors.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=80, kernel_size=251, sample_rate=None):
        super().__init__()
        if sample_rate is None: sample_rate = combnet.SAMPLE_RATE

        min_low_hz=50
        min_band_hz=50

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        self.stride = 1
        self.padding = 0
        self.dilation = 1

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms: torch.Tensor):
        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])

        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        band_pass = band_pass / (2*band[:,None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return torch.abs(torch.nn.functional.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1))


class SincNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        dims = [3200]
        dims.append(
            (dims[-1]-251)//3
        )
        dims.append(
            (dims[-1]-5)//3
        )
        dims.append(
            (dims[-1]-5)//3
        )
        dims.append(
            dims[-1]*60
        )
        dims.append(
            2048
        )
        dims.append(
            2048
        )
        dims.append(
            2048
        )

        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(dims.pop(0)),

            SincLayer(80, 251),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(80, 60, 5),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(60, 60, 5),
            torch.nn.MaxPool1d(3),
            torch.nn.LayerNorm(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            # batch x 60 x 107
            torch.nn.Flatten(1, 2),
            # batch x 6240

            # "DNN1" as per the original implementation
            torch.nn.LayerNorm(dims.pop(0)),

            torch.nn.Linear(6420, 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(2048, 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(2048, 2048, bias=False),
            torch.nn.BatchNorm1d(dims.pop(0)),
            torch.nn.LeakyReLU(0.2),

            # "DNN2" as per the original implementation
            torch.nn.Linear(2048, len(combnet.CLASS_MAP)),
        )

    def parameter_groups(self):
        groups = {}
        groups['main'] = list(self.layers.parameters())
        return groups

    def forward(self, audio):
        audio = audio.unfold(-1, 3200, 160)
        audio = audio.permute(0, 2, 1, 3)
        b, f = audio.shape[0], audio.shape[1]
        audio = audio.flatten(0, 1)
        probs = self.layers(audio)
        probs = probs.unflatten(0, (b, f))
        probs = probs.mean(1)
        return probs