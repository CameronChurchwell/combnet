import torch
from combnet.functional import lc_batch_comb
from combnet.filters import *

# Wrap between K and N
def wrap(x, K, N):
    return ((x - K) % (N - K)) + K

# Range to use for regularization
r = torch.linspace( 50, 8000, 1024)
dr = torch.logspace( 0, -2, 1024) # Decay for high frequencies to allow harmonics crossing

class Comb1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha=.6, gain=None,
        use_bias = False, learn_alpha=False, groups=1, learn_gain=False, sr=16000, comb_fn=None):
        super().__init__()

        if comb_fn is None:
            self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
            # self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit
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
        # return lc_batch_comb( x, self.f.to(d), self.a.to(d), self.sr, self.g.to(d)) + self.b.to(d)
        # return fractional_comb_fir_multitap(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp_explicit(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        return self.comb_fn(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fiir( x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)


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