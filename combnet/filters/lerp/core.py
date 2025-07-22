import torch

def single_fractional_comb_fir_lerp(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0, dtype=torch.float)
    l = sr / f0
    l0 = torch.floor(l).to(int)
    l1 = torch.ceil(l).to(int)
    k = torch.frac(l)
    # l = int(round(sr/f0))
    y = torch.zeros_like(x)
    y += x
    # y[..., l:] += a*y[..., :-l]
    y[..., l0:] += (1-k) * a*x[..., :-l0]
    y[..., l1:] += k * a*x[..., :-l1]
    return y

def single_fractional_comb_fir_multitap_lerp(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0, dtype=torch.float)
    l = sr / f0
    y = torch.zeros_like(x)
    y += x
    n_taps = 10
    for i in range(1, n_taps):
        l_current_tap = l * i
        l0 = torch.floor(l_current_tap).to(int)
        l1 = torch.ceil(l_current_tap).to(int)
        k = torch.frac(l_current_tap)
        y[..., l0:] += (1-k) * (a**i)*x[..., :-l0]
        y[..., l1:] += k * (a**i)*x[..., :-l1]
    return y

def fractional_comb_fir_multitap_lerp(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]
    a = a.expand(f0.shape)

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr / f0 # out_channels x in_channels
    y = torch.zeros(x.shape[0], f0.shape[0], x.shape[-1], device=x.device, dtype=x.dtype) # batch x out_channels x time
    y += x.sum(1, keepdims=True)
    n_taps = 10
    for t in range(1, n_taps+1):
        l_current_tap = l * t
        # l_current_tap = l
        l0 = torch.floor(l_current_tap).to(int)
        l1 = torch.ceil(l_current_tap).to(int)
        k = torch.frac(l_current_tap)


        for o in range(0, k.shape[0]):
            for i in range(0, k.shape[1]):
                a_current = (a[o, i] ** t)
                y[:, o, l0[o, i]:] += (1-k[o, i]) * a_current*x[:, i, :-l0[o, i]]
                y[:, o, l1[o, i]:] += k[o, i] * a_current*x[:, i, :-l1[o, i]]
    return y

class _explicit_lerp(torch.autograd.Function):
    """
    Performs linear interpolation of comb filter to allow gradient computation
    """

    @staticmethod
    @torch.no_grad
    def forward(ctx, x, y, a, l, n_taps):
        ctx.save_for_backward(x, a, l)
        ctx.n_taps = n_taps
        for t in range(1, n_taps+1):
            l_current_tap = l * t
            # l_current_tap = l
            l0 = torch.floor(l_current_tap).to(int)
            l1 = torch.ceil(l_current_tap).to(int)
            k = torch.frac(l_current_tap)

            for o in range(0, k.shape[0]):
                for i in range(0, k.shape[1]):
                    a_current = a[o, i] ** t
                    y[:, o, l0[o, i]:] += (1-k[o, i]) * a_current*x[:, i, :-l0[o, i]]
                    y[:, o, l1[o, i]:] += k[o, i] * a_current*x[:, i, :-l1[o, i]]
        return y

    @staticmethod
    @torch.no_grad
    def backward(ctx, output_gradient: torch.Tensor):
        x, a, l = ctx.saved_tensors
        n_taps = ctx.n_taps

        # batch x out_channels x in_channels x time
        dy_dl = torch.zeros((x.shape[0], l.shape[0], l.shape[1], output_gradient.shape[2]), device=l.device, dtype=l.dtype)

        for t in range(1, n_taps+1):
            l_current_tap = l * t
            # l_current_tap = l
            l0 = torch.floor(l_current_tap).to(int)
            l1 = torch.ceil(l_current_tap).to(int)
            k = torch.frac(l_current_tap)

            for o in range(0, k.shape[0]):
                for i in range(0, k.shape[1]):
                    c_current = t * (a[o, i] ** t)
                    dy_dl[:, o, i, l0[o, i]:] -= (c_current*x[:, i, :-l0[o, i]])
                    dy_dl[:, o, i, l1[o, i]:] += (c_current*x[:, i, :-l1[o, i]])

        dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)

        return None, None, None, dLoss_dl, None


def fractional_comb_fir_multitap_lerp_explicit(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]
    a = a.expand(f0.shape)

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr / f0 # out_channels x in_channels
    y = torch.zeros(x.shape[0], f0.shape[0], x.shape[-1], device=x.device, dtype=x.dtype) # batch x out_channels x time
    y += x.sum(1, keepdims=True)
    n_taps = 10
    return _explicit_lerp.apply(x, y, a, l, n_taps)