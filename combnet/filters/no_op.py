import torch

def comb_no_op(x, f0, a, sr):
    """A dummy method for testing baseline memory usage""" # TODO remove
    return None

def comb_no_op_y(x, f0, a, sr):
    """A dummy method for testing baseline memory usage""" # TODO remove
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

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels
    y = torch.zeros((x.shape[0], f0.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
    return y