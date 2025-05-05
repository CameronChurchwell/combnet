from functools import lru_cache
import torch
from combnet.functional import lc_batch_comb
from combnet.filters import *
import torchaudio

# Wrap between K and N
def wrap(x, K, N):
    return ((x - K) % (N - K)) + K

# Range to use for regularization
r = torch.linspace( 50, 8000, 1024)
dr = torch.logspace( 0, -2, 1024) # Decay for high frequencies to allow harmonics crossing

class Comb1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=.6,
        gain=None,
        use_bias = False,
        learn_alpha=False,
        groups=1,
        learn_gain=False,
        sr=16000,
        comb_fn=None,
    ):
        super().__init__()

        if comb_fn is None:
            self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton_fused
            # self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit
        elif isinstance(comb_fn, str):
            raise NotImplementedError("TODO implement this lookup")
        else:
            self.comb_fn = comb_fn

        self.sr = sr

        self.d = (out_channels,in_channels)
        self.la = learn_alpha

        self.f = torch.nn.Parameter( torch.rand( out_channels, in_channels)*(500-50)+50, requires_grad=True)

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(alpha*torch.ones( self.d), requires_grad=True)
            else:
                self.a = torch.nn.Parameter( torch.rand( out_channels, in_channels)*(.5-.4)+.4, requires_grad=True)
        else:
            self.a = alpha*torch.ones( self.d)

        if use_bias:
            self.b = torch.nn.Parameter( torch.zeros((1,out_channels,1)))
        else:
            self.b = torch.tensor(0)

    def regularization_losses(self):
        regularization = torch.tensor(0., device=self.f.device)
        if not hasattr(self, 'r') or self.r.device != self.f.device:
            self.r = r.to(self.f.device)
            self.dr = dr.to(self.f.device)
        for i in range(0, self.f.shape[0]):
            for j in range(0, i):
                if i == j: continue
                f1 = self.f[i, 0]
                f2 = self.f[j, 0]
                w1 = self.dr*torch.exp( -(wrap( self.r, -f1/2, f1/2)/80)**2) # harmonic bumps for f1
                w2 = self.dr*torch.exp( -(wrap( self.r, -f2/2, f2/2)/80)**2) # harmonic bumps for f2
                regularization += torch.dot( w1/w1.std(), w2/w2.std())
        return regularization, (self.g.clamp(min=0.01) ** 0.5).sum()
        # return torch.tensor(0.0, device=self.f.device), torch.tensor(0.0, device=self.f.device)

    def __call__( self, x):
        d = x.device
        # return lc_batch_comb(x, self.f.to(d), self.a.to(d), self.sr, self.g.to(d)) + self.b.to(d)
        # return fractional_comb_fir_multitap(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp_explicit(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        if self.training:
            return self.comb_fn(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        else: # TODO Debug
            with torch.no_grad():
                return self.comb_fn(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fiir(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)

@lru_cache
def design_fir_highpass(num_taps, cutoff_hz, sample_rate):
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    n = torch.arange(num_taps) - (num_taps - 1) / 2
    h = -torch.sinc(2 * normalized_cutoff * n)
    h[(num_taps - 1) // 2] += 1
    window = torch.hamming_window(num_taps, periodic=False)
    h = h * window
    return h.flip(0)[None, None]

class FusedComb1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=.9,
        gain=None,
        use_bias = False,
        learn_alpha=False,
        groups=1,
        learn_gain=False,
        sr=16000,
        comb_fn=None,
        window_size=None,
        reduction='max',
        stride=None,
        last_stride=True, # include last partial stride
        min_freq=None,
        max_freq=None,
        min_bin=None,
        max_bin=None,
        n_taps=10,
    ):
        self.n_taps = n_taps
        super().__init__()
        assert reduction in ['max', 'sum', 'mean']
        self.reduction = reduction
        if comb_fn is None:
            if reduction == 'max':
                self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton_fused
            elif reduction == 'sum':
                self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
            elif reduction == 'mean':
                self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
            # self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit
        elif isinstance(comb_fn, str):
            comb_fn = getattr(combnet.filters, comb_fn)
            self.comb_fn = comb_fn
        else:
            self.comb_fn = comb_fn

        if window_size is None:
            window_size = combnet.WINDOW_SIZE
        if stride is None:
            stride = combnet.HOPSIZE
        self.window_size = window_size
        self.stride = stride

        self.last_stride = last_stride

        self.sr = sr

        self.d = (out_channels,in_channels)
        self.la = learn_alpha

        scaling_parameters = [min_freq, max_freq, min_bin, max_bin]
        if min_freq is not None and max_freq is not None: # Is this a good way to do this?
            if min_bin is not None and max_bin is not None:
                assert True not in [p is None for p in scaling_parameters]
                nbins = max_bin-min_bin
                fratio = max_freq/min_freq
                if isinstance(alpha, float) and alpha < 0:
                    import warnings
                    warnings.warn('using negative alpha scaling function')
                    @torch.compile
                    def scaling_function(f: torch.Tensor): # f = output_channels x input_channels
                        # f = min_freq * (fratio ** ((max_bin * torch.nn.functional.sigmoid(f) - min_bin) / nbins))
                        s = torch.nn.functional.sigmoid(f)
                        p = nbins*s+min_bin
                        o = min_freq * fratio ** ((p-min_bin) / nbins)
                        return o*2
                else:
                    @torch.compile
                    def scaling_function(f: torch.Tensor): # f = output_channels x input_channels
                        # f = min_freq * (fratio ** ((max_bin * torch.nn.functional.sigmoid(f) - min_bin) / nbins))
                        s = torch.nn.functional.sigmoid(f)
                        p = nbins*s+min_bin
                        o = min_freq * fratio ** ((p-min_bin) / nbins)
                        return o
            else:
                assert min_bin is None and max_bin is None
                fratio = max_freq/min_freq
                @torch.compile
                def scaling_function(f: torch.Tensor):
                    s = torch.nn.functional.sigmoid(f)
                    o = min_freq * fratio ** ((s))
                    return o
            self.scaling_function = scaling_function
        else:
            self.scaling_function = None

        if self.scaling_function:
            if combnet.F0_INIT_METHOD == 'random':
                self.f = torch.nn.Parameter(3*(torch.rand(out_channels, in_channels) * 2 - 1), requires_grad=True)
            elif combnet.F0_INIT_METHOD == 'equal':
                assert in_channels == 1 # TODO generalize
                # self.f = torch.nn.Parameter(torch.linspace(-1, 1, out_channels)[:, None], requires_grad=True)
                self.f = torch.nn.Parameter(torch.linspace(-3, 3, out_channels)[:, None], requires_grad=True)
            else:
                raise ValueError(f'unknown initialization method {combnet.F0_INIT_METHOD}')
        else:
            assert combnet.F0_INIT_METHOD == 'random'
            self.f = torch.nn.Parameter(torch.rand(out_channels, in_channels)*(500-50)+50, requires_grad=True)

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(alpha*torch.ones( self.d), requires_grad=True)
            else:
                self.a = torch.nn.Parameter( torch.rand( out_channels, in_channels)*(.5-.4)+.4, requires_grad=True)
        else:
            self.a = alpha*torch.ones( self.d)

        if use_bias:
            self.b = torch.nn.Parameter( torch.zeros((1,out_channels,1)))
        else:
            self.b = torch.tensor(0)

    def regularization_losses(self):
        regularization = torch.tensor(0., device=self.f.device)
        if not hasattr(self, 'r') or self.r.device != self.f.device:
            self.r = r.to(self.f.device)
            self.dr = dr.to(self.f.device)
        for i in range(0, self.f.shape[0]):
            for j in range(0, i):
                if i == j: continue
                f1 = self.f[i, 0]
                f2 = self.f[j, 0]
                w1 = self.dr*torch.exp( -(wrap( self.r, -f1/2, f1/2)/80)**2) # harmonic bumps for f1
                w2 = self.dr*torch.exp( -(wrap( self.r, -f2/2, f2/2)/80)**2) # harmonic bumps for f2
                regularization += torch.dot( w1/w1.std(), w2/w2.std())
        return regularization, (self.g.clamp(min=0.01) ** 0.5).sum()
        # return torch.tensor(0.0, device=self.f.device), torch.tensor(0.0, device=self.f.device)

    # def forward(self, x):
    #     return self(x)

    # @torch.compile()
    def __call__(self, x):
        d = x.device
        # return lc_batch_comb(x, self.f.to(d), self.a.to(d), self.sr, self.g.to(d)) + self.b.to(d)
        # return fractional_comb_fir_multitap(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp_explicit(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        if self.scaling_function:
            f = self.scaling_function(self.f.to(d))
        else:
            f = self.f.to(d)
        # if self.training:
        if self.reduction == 'max':
            out = self.comb_fn(
                x,
                f,
                self.a.to(d),
                self.sr,
                self.window_size,
                self.stride,
                n_taps=self.n_taps
            ) + self.b.to(d)
        elif self.reduction == 'sum':
            y = self.comb_fn(x, f, self.a.to(d), self.sr)
            out = torch.nn.functional.avg_pool1d(y, self.window_size, self.stride) * self.window_size
        elif self.reduction == 'mean':
            y = self.comb_fn(x, f, self.a.to(d), self.sr)
            out = torch.nn.functional.avg_pool1d(y, self.window_size, self.stride)
        if not self.last_stride:
            out_length = (x.shape[-1] - self.window_size) // self.stride
            out = out[..., :out_length]
        return out
        # else: # TODO Debug
        #     with torch.no_grad():
        #         return self.comb_fn(x, f, self.a.to(d), self.sr, self.window_size, self.stride) + self.b.to(d)
        # return fractional_comb_fiir(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)

# Comb1dFIIR = partial(Comb1d, comb_fn=combnet.filters.fractional_comb_fiir)

class CombInterference1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, alpha=.6, gain=None,
        use_bias = False, learn_alpha=False, groups = 1, learn_gain=False, sr=16000):
        super().__init__()

        self.sr = sr

        self.d = (out_channels,in_channels)
        self.la = learn_alpha

        self.f = torch.nn.Parameter( torch.rand( out_channels, in_channels)*(500-50)+50, requires_grad=True)

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(alpha*torch.ones( self.d), requires_grad=True)
            else:
                self.a = torch.nn.Parameter( torch.rand( out_channels, in_channels)*(.5-.4)+.4, requires_grad=True)
        else:
            self.a = alpha*torch.ones( self.d)

        if use_bias:
            self.b = torch.nn.Parameter( torch.zeros((1,out_channels,1)))
        else:
            self.b = torch.tensor(0)


    def __call__( self, x):
        x = fractional_anticomb_interference_fiir(x, self.f, self.a.to(x.device), self.sr)
        x = fractional_comb_fiir(x, self.f, self.a.to(x.device), self.sr) #+ self.b
        return x
    
class CombResidual1d(CombInterference1d):

    def __call__(self, x):
        x = fractional_anticomb_interference_fiir(x, self.f, self.a.to(x.device), self.sr, residual_mode=True)
        x = fractional_comb_fiir(x, self.f, self.a.to(x.device), self.sr) #+ self.b
        return x