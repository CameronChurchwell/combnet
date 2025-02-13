import torch
import torch.nn.grad
import triton
import triton.language as tl
import combnet
from functools import partial
# import triton_viz

# from torch_xla.experimental.custom_kernel import jax_import_guard
# jax_import_guard() # only needed on TPU, which we don't support
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

sinc = torch.sinc
# sinc = sinc_safe
# def sinc(x):
#     return torch.where(x==0., 1., torch.sin(x)/x)

def sinc_safe(x):
    x = torch.where(x==0., 1e-9, x)
    return torch.sin(x)/x

def sparse_sinc(x):
    sinced = sinc(x)
    outputs = torch.where(
        (-4.5<x) & (x<=4.5),
        # (-1.5<x) & (x<=1.5),
        sinced,
        0.
    )
    return outputs

# Simplify calling a convolution
def convolve( x, y):
    from torch.nn.functional import conv1d
    return conv1d( x[None,None], y[None,None], padding='same')[0,0]

# Take 1: Use an IIR comb filter, must use scan to avoid jit dealing with variable loop count
# Works fine, but has a DC peak and is insanely slow
def single_fractional_comb_iir( x, f0, a, sr):
    # Make the filter, essentially a fractional (sinc) delay in the feedback path
    l = sr/f0
    t = torch.arange( sr//20) - l + 1 # Fixed the buffer size to a constant to avoid jit complaints, I assume we don't care for sub 20Hz
    a = a * torch.sinc( t) * torch.exp( -.1 * t**2)
    # a = a.at[0].set( -.02)

    # Core scan routine utilizing a ring buffer
    c = torch.zeros( len( a))
    y = torch.zeros( len( x))
    for i in torch.arange( len( x)):
        y[i] = x[i] + a.dot( c)
        c = torch.roll( c, shifts=1)
        c[0] = y[i]

    # Do it
    return y

def single_comb_iir_faithful(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    for i in range(l, x.shape[-1]):
        y[..., i] += a*y[..., i-l]
    return y

def single_comb_fir(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    y[..., l:] += a*y[..., :-l]
    return y

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

def single_comb_fir_multitap(x, f0, a, sr):
    y = x
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    for i in range(1, 10):
        y[..., i*l:] += (a**i)*x[..., :-i*l]
    return y

def single_fractional_comb_fir_multitap(x, f0, a, sr):
    x = x.squeeze()
    l = sr/f0
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device)

    f = torch.zeros(sr//10, device=x.device)
    f[-1] = 1.
    # TODO you can definitely remove this loop, but it might be less
    #  memory efficient and still not that much faster... The slow part is the conv
    n_taps = 10
    for i in range(1, n_taps+1):
        f += (a ** i) * torch.sinc(t-i*l)
    # from matplotlib import pyplot as plt
    # plt.plot(f); plt.gcf().set_size_inches(10, 7.5); plt.show()
    x = torch.nn.functional.pad(x, (sr//10-1, 0))
    y = torch.nn.functional.conv1d(
        x[None,None], 
        f[None,None],
    )[0,0]
    return y
    # return convolve(x, f)

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
    Stands in for the following operation:
    ```
    #TODO fill in
    ```
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

# @triton_viz.trace
@triton.jit
def _lerp_forward_kernel(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    y, # batch x out_channels x time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    block_batch: tl.constexpr = 2,
    block_in_channels: tl.constexpr = 2,
    block_out_channels: tl.constexpr = 2,
    block_time: tl.constexpr = 512,
):
    """Computes a `block_batch x block_out_channels x block_time` block of y
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    accumulator = tl.zeros((block_batch, block_out_channels, block_time), dtype=tl.float32) # block_batch x block_out_channels x block_time

    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1

    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    for ic_counter in range(0, in_channels, block_in_channels):
        in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
        channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
        channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
        delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
        gains = tl.load(a + channel_indices, channel_mask, 1.0) # 1 x block_out_channels x block_in_channels x 1
        base_gains = tl.load(a + channel_indices, channel_mask, 1.0)

        # iterate over taps
        for tap in range(1, n_taps+1, 1):
            # calculate lerp ratio of floor and ceil
            fractional_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
            floor_delays = tl.floor(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
            ceil_delays = tl.ceil(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
            k = fractional_delays - floor_delays # 1 x block_out_channels x block_in_channels x 1

            # l0
            # apply delays to indices
            tap_indices = indices - floor_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            # x_tile = tl.load(x + x_indices, mask) * ((1-k)* (tl.exp(tl.log(gains) * tap)))
            x_tile = tl.load(x + x_indices, mask) * ((1-k)*gains)
            accumulator += tl.sum(x_tile, 2)

            # l1
            # apply delays to indices
            tap_indices = indices - ceil_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            # x_tile = tl.load(x + x_indices, mask) * (k * (tl.exp(tl.log(gains) * tap))) # block_out_channels x block_in_channels x block_time
            x_tile = tl.load(x + x_indices, mask) * (k*gains) # block_out_channels x block_in_channels x block_time
            accumulator += tl.sum(x_tile, 2)

            gains *= base_gains

    # store tile in y
    y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    # if t_id == 0:
    #     tl.device_print('y_indices', y_indices)
    y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # if (n_id == 0) & (o_id == 0) & (t_id == 0): # works?
    # if (n_id == 0) & (t_id == 0): # works??
    # if (t_id == 0):
    tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

@triton.jit
def _lerp_backward_kernel(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    gradient, # batch x out_channels x in_channels x time
    # output_gradient, # batch x out_channels x time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    block_batch: tl.constexpr = 1,
    block_in_channels: tl.constexpr = 1,
    block_out_channels: tl.constexpr = 1,
    block_time: tl.constexpr = 512,
):
    """Computes a `block_batch x block_out_channels x block_in_channels x block_time` block of y
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    accumulator = tl.zeros((block_batch, block_out_channels, block_in_channels, block_time), dtype=tl.float32)

    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1

    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    # if t_id == 0:
    #     tl.device_print('out_channel_indices', out_channel_indices)
    for ic_counter in range(0, in_channels, block_in_channels):
        in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
        channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
        channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
        delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
        gains = tl.load(a + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1

        # iterate over taps
        for tap in range(1, n_taps+1, 1):
            # calculate lerp ratio of floor and ceil
            fractional_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
            floor_delays = tl.floor(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
            ceil_delays = tl.ceil(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1

            # l0
            # apply delays to indices
            tap_indices = indices - floor_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            tl.device_assert(gains > 0)
            x_tile = tl.load(x + x_indices, mask) * (tap * (tl.exp(tl.log(gains) * tap)))
            accumulator -= x_tile

            # l1
            # apply delays to indices
            tap_indices = indices - ceil_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * (tap * (tl.exp(tl.log(gains) * tap))) # block_out_channels x block_in_channels x block_time
            accumulator += x_tile

        # og_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
        # og_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # og_tile = tl.load(output_gradient + og_indices, og_mask)

        # partial_gradient = tl.sum(tl.sum(accumulator * og_tile, 3, keep_dims=True), 0, keep_dims=True)

        # g_indices = in_channel_indices + out_channel_indices * in_channels
        # g_mask = (out_channel_indices < out_channels) & (in_channel_indices < in_channels)
        # tl.atomic_add(gradient+g_indices, partial_gradient, g_mask)

        g_indices = indices + in_channel_indices * time + out_channel_indices * in_channels * time + batch_indices * out_channels * in_channels * time
        g_mask = (indices < time) & (in_channel_indices < in_channels) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # tl.atomic_add(gradient+g_indices, accumulator, g_mask, 'relaxed') # relaxed seems to be faster with no downsides
        tl.atomic_add(gradient+g_indices, accumulator, g_mask)


    # store tile in y
    # y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    # if t_id == 0:
    #     tl.device_print('y_indices', y_indices)
    # y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

class _explicit_lerp_triton(torch.autograd.Function):
    """
    Stands in for the following operation:
    ```
    #TODO fill in
    ```
    """

    @staticmethod
    @torch.no_grad
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, l: torch.Tensor, n_taps):
        # TODO this is not strong enough, need to check strides multiply up to dims
        assert x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and l.is_contiguous()
        assert x.device == y.device == a.device == l.device
        ctx.save_for_backward(x, a, l)
        ctx.n_taps = n_taps
        def grid(META):
            grid_shape = (
                triton.cdiv(y.shape[0], META["block_batch"]),
                triton.cdiv(y.shape[1], META["block_out_channels"]),
                triton.cdiv(y.shape[2], META["block_time"])
            )
            return grid_shape
        _lerp_forward_kernel[grid](
            x=x, # batch x in_channels x time
            l=l, # out_channels x in_channels
            a=a, # out_channels x in_channels
            y=y, # batch x out_channels x time
            batch_size=y.shape[0],
            n_taps=n_taps,
            out_channels=y.shape[1],
            in_channels=x.shape[1],
            time=x.shape[-1],
        )
        return y

    @staticmethod
    @torch.no_grad
    def backward(ctx, output_gradient: torch.Tensor):
        x, a, l = ctx.saved_tensors
        n_taps = ctx.n_taps

        # batch x out_channels x in_channels x time
        dy_dl = torch.zeros((x.shape[0], l.shape[0], l.shape[1], output_gradient.shape[2]), device=l.device, dtype=l.dtype)

        def grid(META):
            grid_shape = (
                triton.cdiv(x.shape[0], META["block_batch"]),
                triton.cdiv(l.shape[0], META["block_out_channels"]),
                triton.cdiv(x.shape[-1], META["block_time"])
            )
            return grid_shape
        _lerp_backward_kernel[grid](
            x=x, # batch x in_channels x time
            l=l, # out_channels x in_channels
            a=a, # out_channels x in_channels
            gradient=dy_dl, # batch x out_channels x time
            # output_gradient=output_gradient.contiguous(),
            batch_size=x.shape[0],
            n_taps=n_taps,
            out_channels=l.shape[0],
            in_channels=l.shape[1],
            time=x.shape[-1],
        )

        # dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)
        dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)

        return None, None, None, dLoss_dl, None

def fractional_comb_fir_multitap_lerp_explicit_triton(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]
    x = x.contiguous()

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]
    a = a.expand(f0.shape).contiguous()

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr / f0 # out_channels x in_channels
    n_taps = 10
    y = torch.zeros(x.shape[0], f0.shape[0], x.shape[-1], device=x.device, dtype=x.dtype) # batch x out_channels x time
    y = _explicit_lerp_triton.apply(x, y, a, l, n_taps)
    y = y + x.sum(1, keepdims=True)
    return y

@triton.jit
def _lerp_forward_kernel_fused(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    y, # batch x out_channels x time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    out_time: int,
    window_size: int,
    stride: int,
    block_batch: tl.constexpr = 2,
    block_in_channels: tl.constexpr = 1,
    block_out_channels: tl.constexpr = 8,
    block_time: tl.constexpr = 512,
):
    """
    Computes a `block_batch x block_out_channels x 1` block of y
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    result = tl.zeros((block_batch, block_out_channels), dtype=tl.float32)
    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1
    for block_window_offset in range(0, window_size, block_time):
        indices = tl.arange(0, block_time)[None, None, None] + t_id * stride + block_window_offset # 1 x 1 x 1 x block_time
        accumulator = tl.zeros((block_batch, block_out_channels, block_time), dtype=tl.float32) # block_batch x block_out_channels x block_time

        for ic_counter in range(0, in_channels, block_in_channels):
            in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
            channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
            channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
            delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
            gains = tl.load(a + channel_indices, channel_mask, 1.0) # 1 x block_out_channels x block_in_channels x 1
            base_gains = tl.load(a + channel_indices, channel_mask, 1.0) # 1 x block_out_channels x block_in_channels x 1

            # iterate over taps
            for tap in range(0, n_taps+1, 1):
                # calculate lerp ratio of floor and ceil
                fractional_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
                floor_delays = tl.floor(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
                ceil_delays = tl.ceil(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
                k = fractional_delays - floor_delays # 1 x block_out_channels x block_in_channels x 1

                # l0
                # apply delays to indices
                tap_indices = indices - floor_delays # 1 x block_out_channels x block_in_channels x block_time
                # apply channel and batch and channel offsets to indices
                x_indices = tap_indices + in_channel_indices * time \
                    + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
                # create mask based on indices components and bounds
                mask = (tap_indices >= 0) \
                    & (tap_indices < time) \
                    & (in_channel_indices < in_channels) \
                    & (batch_indices < batch_size) \
                    & (indices < t_id * stride + window_size)
                # x_tile = tl.load(x + x_indices, mask) * ((1-k)* (tl.exp(tl.log(gains) * tap)))
                x_tile = tl.load(x + x_indices, mask) * ((1-k)*gains)
                accumulator += tl.sum(x_tile, 2)

                # l1
                # apply delays to indices
                tap_indices = indices - ceil_delays # 1 x block_out_channels x block_in_channels x block_time
                # apply channel and batch and channel offsets to indices
                x_indices = tap_indices + in_channel_indices * time \
                    + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
                # create mask based on indices components and bounds
                mask = (tap_indices >= 0) \
                    & (tap_indices < time) \
                    & (in_channel_indices < in_channels) \
                    & (batch_indices < batch_size) \
                    & (indices < t_id * stride + window_size)
                x_tile = tl.load(x + x_indices, mask) * (k*gains) # block_out_channels x block_in_channels x block_time
                accumulator += tl.sum(x_tile, 2)

                gains *= base_gains

        # compute max over block_time chunk and compare/store with result
        result = tl.maximum(tl.max(tl.abs(accumulator), axis=2), result)

    # store tile in y
    y_indices = t_id + out_channel_indices * out_time + batch_indices * out_channels * out_time
    # if t_id == 0:
    #     tl.device_print('y_indices', y_indices)
    y_mask = (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # if (n_id == 0) & (o_id == 0) & (t_id == 0): # works?
    # if (n_id == 0) & (t_id == 0): # works??
    # if (t_id == 0):
    # tl.static_print(y_indices.shape, result.shape, y_mask.shape)
    # tl.static_assert(False)
    tl.store(y+y_indices, result[:, :, None, None], y_mask)

@triton.jit
def _lerp_backward_kernel_fused(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    gradient, # batch x out_channels x in_channels x time
    # output_gradient, # batch x out_channels x time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    block_batch: tl.constexpr = 1,
    block_in_channels: tl.constexpr = 1,
    block_out_channels: tl.constexpr = 1,
    block_time: tl.constexpr = 512,
):
    """Computes a `block_batch x block_out_channels x block_in_channels x block_time` block of y
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    accumulator = tl.zeros((block_batch, block_out_channels, block_in_channels, block_time), dtype=tl.float32)

    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1

    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    # if t_id == 0:
    #     tl.device_print('out_channel_indices', out_channel_indices)
    for ic_counter in range(0, in_channels, block_in_channels):
        in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
        channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
        channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
        delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
        gains = tl.load(a + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1

        # iterate over taps
        for tap in range(1, n_taps+1, 1):
            # calculate lerp ratio of floor and ceil
            fractional_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
            floor_delays = tl.floor(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
            ceil_delays = tl.ceil(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1

            # l0
            # apply delays to indices
            tap_indices = indices - floor_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            tl.device_assert(gains > 0)
            x_tile = tl.load(x + x_indices, mask) * (tap * (tl.exp(tl.log(gains) * tap)))
            accumulator -= x_tile

            # l1
            # apply delays to indices
            tap_indices = indices - ceil_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * (tap * (tl.exp(tl.log(gains) * tap))) # block_out_channels x block_in_channels x block_time
            accumulator += x_tile

        # og_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
        # og_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # og_tile = tl.load(output_gradient + og_indices, og_mask)

        # partial_gradient = tl.sum(tl.sum(accumulator * og_tile, 3, keep_dims=True), 0, keep_dims=True)

        # g_indices = in_channel_indices + out_channel_indices * in_channels
        # g_mask = (out_channel_indices < out_channels) & (in_channel_indices < in_channels)
        # tl.atomic_add(gradient+g_indices, partial_gradient, g_mask)

        g_indices = indices + in_channel_indices * time + out_channel_indices * in_channels * time + batch_indices * out_channels * in_channels * time
        g_mask = (indices < time) & (in_channel_indices < in_channels) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # tl.atomic_add(gradient+g_indices, accumulator, g_mask, 'relaxed') # relaxed seems to be faster with no downsides
        tl.atomic_add(gradient+g_indices, accumulator, g_mask)


    # store tile in y
    # y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    # if t_id == 0:
    #     tl.device_print('y_indices', y_indices)
    # y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

class _explicit_lerp_triton_fused(torch.autograd.Function):
    """
    Stands in for the following operation:
    ```
    #TODO fill in
    ```
    """

    @staticmethod
    @torch.no_grad
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        l: torch.Tensor,
        n_taps,
        window_size,
        stride
    ):
        # TODO this is not strong enough, need to check strides multiply up to dims
        assert x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and l.is_contiguous()
        assert x.device == y.device == a.device == l.device
        ctx.save_for_backward(x, a, l)
        ctx.n_taps = n_taps
        ctx.window_size = window_size
        ctx.stride = stride
        def grid(META):
            grid_shape = (
                triton.cdiv(y.shape[0], META["block_batch"]),
                triton.cdiv(y.shape[1], META["block_out_channels"]),
                y.shape[2], # one program per "stride"
            )
            return grid_shape
        _lerp_forward_kernel_fused[grid](
            x=x, # batch x in_channels x time
            l=l, # out_channels x in_channels
            a=a, # out_channels x in_channels
            y=y, # batch x out_channels x time
            batch_size=y.shape[0],
            n_taps=n_taps,
            out_channels=y.shape[1],
            in_channels=x.shape[1],
            time=x.shape[-1],
            out_time=y.shape[-1],
            window_size=window_size,
            stride=stride,
        )
        return y

    # @staticmethod
    # @torch.no_grad
    # def backward(ctx, output_gradient: torch.Tensor):
    #     x, a, l = ctx.saved_tensors
    #     n_taps = ctx.n_taps

    #     # batch x out_channels x in_channels x time
    #     dy_dl = torch.zeros((x.shape[0], l.shape[0], l.shape[1], output_gradient.shape[2]), device=l.device, dtype=l.dtype)

    #     def grid(META):
    #         grid_shape = (
    #             triton.cdiv(x.shape[0], META["block_batch"]),
    #             triton.cdiv(l.shape[0], META["block_out_channels"]),
    #             triton.cdiv(x.shape[-1], META["block_time"])
    #         )
    #         return grid_shape
    #     _lerp_backward_kernel_fused[grid](
    #         x=x, # batch x in_channels x time
    #         l=l, # out_channels x in_channels
    #         a=a, # out_channels x in_channels
    #         gradient=dy_dl, # batch x out_channels x time
    #         # output_gradient=output_gradient.contiguous(),
    #         batch_size=x.shape[0],
    #         n_taps=n_taps,
    #         out_channels=l.shape[0],
    #         in_channels=l.shape[1],
    #         time=x.shape[-1],
    #     )

    #     # dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)
    #     dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)

    #     return None, None, None, dLoss_dl, None

def fractional_comb_fir_multitap_lerp_explicit_triton_fused(x, f0, a, sr, window_size=None, stride=None):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]
    x = x.contiguous()

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]
    a = a.expand(f0.shape).contiguous()

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    if window_size is None:
        window_size = 1024
    if stride is None:
        stride = 256

    #TODO assert window_size and stride are powers of 2?
    l = sr / f0 # out_channels x in_channels
    n_taps = 10
    out_time = (x.shape[-1] - window_size) // stride + 1
    y = torch.zeros(x.shape[0], f0.shape[0], out_time, device=x.device, dtype=x.dtype) # batch x out_channels x time
    y = _explicit_lerp_triton_fused.apply(x, y, a, l, n_taps, window_size, stride)
    return y

def _lerp_forward_pallas(
    # x_ref, # batch x in_channels x time
    # l_ref, # out_channels x in_channels
    # a_ref, # out_channels x in_channels
    y_ref, # batch x out_channels x time
    # n_taps: int,
):
    """Computes a `block_batch x block_out_channels x block_time` block of y
    """
    n_id = pl.program_id(0)
    o_id = pl.program_id(1)
    t_id = pl.program_id(2)
    y_ref[...] = jnp.zeros(y_ref.shape)

    # indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    # accumulator = tl.zeros((block_batch, block_out_channels, block_time), dtype=tl.float32) # block_batch x block_out_channels x block_time

    # batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1

    # out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    # for ic_counter in range(0, in_channels, block_in_channels):
    #     in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
    #     channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
    #     channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
    #     delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
    #     gains = tl.load(a + channel_indices, channel_mask, 1.0) # 1 x block_out_channels x block_in_channels x 1

    #     # iterate over taps
    #     for tap in range(1, n_taps+1, 1):
    #         # calculate lerp ratio of floor and ceil
    #         fractional_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
    #         floor_delays = tl.floor(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
    #         ceil_delays = tl.ceil(fractional_delays).to(tl.int32) # 1 x block_out_channels x block_in_channels x 1
    #         k = fractional_delays - floor_delays # 1 x block_out_channels x block_in_channels x 1

    #         # l0
    #         # apply delays to indices
    #         tap_indices = indices - floor_delays # 1 x block_out_channels x block_in_channels x block_time
    #         # apply channel and batch and channel offsets to indices
    #         x_indices = tap_indices + in_channel_indices * time \
    #             + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
    #         # create mask based on indices components and bounds
    #         mask = (tap_indices >= 0) \
    #             & (tap_indices < time) \
    #             & (in_channel_indices < in_channels) \
    #             & (batch_indices < batch_size)
    #         x_tile = tl.load(x + x_indices, mask) * ((1-k)* (tl.exp(tl.log(gains) * tap)))
    #         accumulator += tl.sum(x_tile, 2)

    #         # l1
    #         # apply delays to indices
    #         tap_indices = indices - ceil_delays # 1 x block_out_channels x block_in_channels x block_time
    #         # apply channel and batch and channel offsets to indices
    #         x_indices = tap_indices + in_channel_indices * time \
    #             + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
    #         # create mask based on indices components and bounds
    #         mask = (tap_indices >= 0) \
    #             & (tap_indices < time) \
    #             & (in_channel_indices < in_channels) \
    #             & (batch_indices < batch_size)
    #         x_tile = tl.load(x + x_indices, mask) * (k * (tl.exp(tl.log(gains) * tap))) # block_out_channels x block_in_channels x block_time
    #         accumulator += tl.sum(x_tile, 2)

    # # store tile in y
    # y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    # # if t_id == 0:
    # #     tl.device_print('y_indices', y_indices)
    # y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # # if (n_id == 0) & (o_id == 0) & (t_id == 0): # works?
    # # if (n_id == 0) & (t_id == 0): # works??
    # # if (t_id == 0):
    # tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

class _explicit_lerp_pallas(torch.autograd.Function):
    """
    Stands in for the following operation:
    ```
    #TODO fill in
    ```
    """

    @staticmethod
    @torch.no_grad
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, l: torch.Tensor, n_taps):
        assert x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and l.is_contiguous()
        assert x.device == y.device == a.device == l.device
        ctx.save_for_backward(x, a, l)
        ctx.n_taps = n_taps
        # def grid(META):
        #     grid_shape = (
        #         triton.cdiv(y.shape[0], META["block_batch"]),
        #         triton.cdiv(y.shape[1], META["block_out_channels"]),
        #         triton.cdiv(y.shape[2], META["block_time"])
        #     )
        #     # print(grid_shape)
        #     return grid_shape
        pallas_kernel = pl.pallas_call(
            _lerp_forward_pallas,
            out_shape=jax.ShapeDtypeStruct((y.shape[0], y.shape[1], y.shape[2]), jnp.float32),
            out_specs=pl.BlockSpec(
                (2, 2, 512), lambda i, j, k: (i, j, k)
            ),
            grid=(1, 1, 4)
        )
        @partial(jax.jit, backend='gpu')
        def pallas_kernel_call():
            return pallas_kernel()
        y_p = pallas_kernel_call()
        breakpoint()
        return y

    @staticmethod
    @torch.no_grad
    def backward(ctx, output_gradient: torch.Tensor):
        return None, None, None, None, None
        # # return _explicit_lerp.backward(ctx, output_gradient)
        # x, a, l = ctx.saved_tensors
        # n_taps = ctx.n_taps

        # # batch x out_channels x in_channels x time
        # dy_dl = torch.zeros((x.shape[0], l.shape[0], l.shape[1], output_gradient.shape[2]), device=l.device, dtype=l.dtype)

        # # for t in range(1, n_taps+1):
        # #     l_current_tap = l * t
        # #     l0 = torch.floor(l_current_tap).to(int)
        # #     l1 = torch.ceil(l_current_tap).to(int)

        # #     for o in range(0, l.shape[0]):
        # #         for i in range(0, l.shape[1]):
        # #             c_current = t * (a[o, i] ** t)
        # #             dy_dl[:, o, i, l0[o, i]:] -= (c_current*x[:, i, :-l0[o, i]])
        # #             dy_dl[:, o, i, l1[o, i]:] += (c_current*x[:, i, :-l1[o, i]])

        # # gradient = torch.zeros(l.shape[0], l.shape[1], device=l.device, dtype=l.dtype)

        # def grid(META):
        #     grid_shape = (
        #         triton.cdiv(x.shape[0], META["block_batch"]),
        #         triton.cdiv(l.shape[0], META["block_out_channels"]),
        #         triton.cdiv(x.shape[-1], META["block_time"])
        #     )
        #     # print(grid_shape)
        #     return grid_shape
        # _lerp_backward_kernel[grid](
        #     x=x, # batch x in_channels x time
        #     l=l, # out_channels x in_channels
        #     a=a, # out_channels x in_channels
        #     gradient=dy_dl, # batch x out_channels x time
        #     # output_gradient=output_gradient.contiguous(),
        #     batch_size=x.shape[0],
        #     n_taps=n_taps,
        #     out_channels=l.shape[0],
        #     in_channels=l.shape[1],
        #     time=x.shape[-1],
        # )

        # # dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)
        # dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)

        # return None, None, None, dLoss_dl, None

def fractional_comb_fir_multitap_lerp_explicit_pallas(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]
    x = x.contiguous()

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]
    a = a.expand(f0.shape).contiguous()

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr / f0 # out_channels x in_channels
    n_taps = 10
    y = torch.zeros(x.shape[0], f0.shape[0], x.shape[-1], device=x.device, dtype=x.dtype) # batch x out_channels x time
    y = _explicit_lerp_pallas.apply(x, y, a, l, n_taps)
    y = y + x.sum(1, keepdims=True)
    return y

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

@torch.no_grad
@torch.compile(mode='reduce-overhead')
def comb_fir_multitap(x, f0, a, sr):
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
    l = (sr//f0).to(int) # out_channels x in_channels
    y = torch.zeros((x.shape[0], l.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
    n_taps=10
    for t in range(1, n_taps+1):
        for o in range(0, l.shape[0]):
            for i in range(0, l.shape[1]):
                y[:, o, t*l[o, i]:] += (a[o, i]**t)*x[:, i, :-t*l[o, i]]
    return y

@triton.jit
def _multitap_forward_kernel(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    y, # batch x out_channels x time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    block_batch: tl.constexpr = 1,
    block_in_channels: tl.constexpr = 1,
    block_out_channels: tl.constexpr = 1,
    block_time: tl.constexpr = 512,
):
    """Computes a `block_batch x block_out_channels x block_time` block of y
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    accumulator = tl.zeros((block_batch, block_out_channels, block_time), dtype=tl.float32) # block_batch x block_out_channels x block_time

    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1

    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1
    for ic_counter in range(0, in_channels, block_in_channels):
        in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
        channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
        channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
        delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
        gains = tl.load(a + channel_indices, channel_mask, 1.0) # 1 x block_out_channels x block_in_channels x 1

        # iterate over taps
        for tap in range(1, n_taps+1, 1):
            # calculate lerp ratio of floor and ceil
            tap_delays = tap * delays # 1 x block_out_channels x block_in_channels x 1
            # apply delays to indices
            tap_indices = indices - tap_delays # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * (tl.exp(tl.log(gains) * tap))
            accumulator += tl.sum(x_tile, 2)

    # store tile in y
    y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
    tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

@torch.no_grad()
def comb_fir_multitap_triton(x, f0, a, sr):
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
    l = (sr//f0).to(int) # out_channels x in_channels
    y = torch.zeros((x.shape[0], l.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
    n_taps=10
    assert x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and l.is_contiguous()
    assert x.device == y.device == a.device == l.device
    def grid(META):
        grid_shape = (
            triton.cdiv(y.shape[0], META["block_batch"]),
            triton.cdiv(y.shape[1], META["block_out_channels"]),
            triton.cdiv(y.shape[2], META["block_time"])
        )
        return grid_shape
    _multitap_forward_kernel[grid](
        x=x, # batch x in_channels x time
        l=l, # out_channels x in_channels
        a=a, # out_channels x in_channels
        y=y, # batch x out_channels x time
        batch_size=y.shape[0],
        n_taps=n_taps,
        out_channels=y.shape[1],
        in_channels=x.shape[1],
        time=x.shape[-1],
    )
    y = y + x.sum(1, keepdims=True)
    return y

def fractional_comb_fir_multitap(x, f0, a, sr):
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

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    n_taps=10

    # Tensor method
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    f = (gains * sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

    # Loop method
    # f = torch.zeros((f0.shape[0], f0.shape[1], sr//10), device=x.device) # out_channels x in_channels x kernel_size
    # for i in range(1, n_taps+1):
    #     delay = (i * l)[..., None] # out_channels x in_channels x 1
    #     gain = (a ** i)[..., None] # out_channels x in_channels x 1
    #     time = t[None, None] # 1 x 1 x kernel_size
    #     shifted_time = time - delay # out_channels x in_channels x kernel_size
    #     f += gain * torch.sinc(shifted_time)

    f[..., -1] = 1. # original signal (x[i])

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f, # out_channels x in_channels x kernel_size
    ) # batch x out_channels x time

    return y


# cudagraphs inductor onnxrt openxla tvm
@torch.compile(mode='max-autotune') # reduce overhead (also try cudagraph)
def fractional_comb_fir_multitap_torch_compile(x, f0, a, sr):
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

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    n_taps=10

    # Tensor method
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    f = (gains * sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

    # Loop method
    # f = torch.zeros((f0.shape[0], f0.shape[1], sr//10), device=x.device) # out_channels x in_channels x kernel_size
    # for i in range(1, n_taps+1):
    #     delay = (i * l)[..., None] # out_channels x in_channels x 1
    #     gain = (a ** i)[..., None] # out_channels x in_channels x 1
    #     time = t[None, None] # 1 x 1 x kernel_size
    #     shifted_time = time - delay # out_channels x in_channels x kernel_size
    #     f += gain * torch.sinc(shifted_time)

    # f[..., -1] = 1. # original signal (x[i])

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f, # out_channels x in_channels x kernel_size
    ) # batch x out_channels x time

    y += x

def fractional_comb_fir_multitap_pseudo_sparse(x, f0, a, sr):
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

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Tensor method
    taps = torch.arange(1, 11, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    f = (gains * sparse_sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

    # Loop method
    # f = torch.zeros((f0.shape[0], f0.shape[1], sr//10), device=x.device) # out_channels x in_channels x kernel_size
    # for i in range(1, 11):
    #     delay = (i * l)[..., None] # out_channels x in_channels x 1
    #     gain = (a ** i)[..., None] # out_channels x in_channels x 1
    #     time = t[None, None] # 1 x 1 x kernel_size
    #     shifted_time = time - delay # out_channels x in_channels x kernel_size
    #     f += gain * torch.sinc(shifted_time)

    f[..., -1] = 1. # original signal (x[i])

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f, # out_channels x in_channels x kernel_size
    ) # batch x out_channels x time

    return y


def fractional_comb_fir_multitap_sparse(x, f0, a, sr):
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

    # TODO make it work with more than 1 input channel
    assert f0.shape[1] == 1

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Construct the filters
    n_taps = 10
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    sinced = sparse_sinc(shifted_time)

    # out_channels*in_channels*n_taps x 4
    centers = (shifted_time.permute(1, 2, 0, 3).ceil()==0).argwhere()

    try:
        centers = centers[:, 3].reshape(f0.shape[0], f0.shape[1], n_taps) # out_channels x in_channels x n_taps
    except:
        import pdb; pdb.set_trace()

    if sinced.isnan().any():
        import pdb; pdb.set_trace()
    f = (gains * sinced).sum(0) # out_channels x in_channels x kernel_size
    # f[..., -1] = 1. # original signal (x[i]) # not needed if we just sum the original signal in later

    # x_unpadded = x

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    offsets = delays.squeeze().round()

    # This is really stupid and there has to be a better way to compute these kernels, but we'll do that later

    num_nonzero = (abs(f)>0).sum(2).unique().item()

    kernel_block_size = num_nonzero//n_taps

    block_radius = kernel_block_size // 2

    f_condensed = f[abs(f)>0].reshape(
        f.shape[0], f.shape[1], n_taps, kernel_block_size) # out_channels x in_channels x n_taps x kernel_block_size
    f_condensed = f_condensed.flip(2) # What? Why? Well, the above line puts the blocks in reverse order to tap index
    f_condensed = f_condensed.permute(0, 2, 1, 3) # out_channels x n_taps x in_channels x kernel_block_size
    f_condensed = f_condensed.flatten(0, 1) # out_channels*n_taps x in_channels x kernel_block_size

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f_condensed, # out_channels*n_taps x in_channels x kernel_block_size
    ) # batch x out_channels*n_taps x time

    y = y.unflatten(1, (-1, n_taps)) # batch x out_channels x n_taps x time

    assert (num_nonzero//n_taps) % 2 == 1 # TODO relax this constraint
    output_length = x.shape[-1] - f.shape[-1] + 1

    # TODO figure out input_channels...
    assert centers.shape[1] == 1
    centers = centers[:, 0] # output_channels x n_taps
    offsets = centers - block_radius # output_channels x n_taps
    # Now we just have to grab the correct slices and sum them together

    # First attempt: Create index tensor and use gather
    # Uses a lot of vram (for the index tensor) and is very slow. The actual gather operation is fast though
    # # indices = torch.arange(0, output_length)[None, None, :].to(x.device) #1 x 1 x time
    # indices = torch.arange(0, output_length, device=x.device)[None, None, :] #1 x 1 x time
    # indices = indices + centers[:, :, None] - block_radius # output_channels x n_taps x time
    # indices = indices[None].expand(y.shape[0], -1, -1, -1) # batch x output_channels x n_taps x time
    # y = torch.gather(y, 3, indices) # batch x output_channels x n_taps x time'
    # y = y.sum(2)

    # Second attempt: Use unfolding and advanced indexing.
    # Backward pass tries to use like 1TB of vram, probably because of unfold
    # unfolded = y.unfold(3, output_length, 1) # batch x output_channels x n_taps x offsets x time'
    # channels_indices = torch.arange(0, unfolded.shape[1])[:, None].expand(unfolded.shape[1], n_taps) # output_channels x n_taps
    # taps_indices = torch.arange(0, n_taps)[None].expand(unfolded.shape[1], n_taps) # output_channels x n_taps
    # y = unfolded[:, channels_indices, taps_indices, offsets] # batch x output_channels x n_taps x time'
    # y = y.sum(2)

    # Third attempt: Just use 2 Python loops
    # Works and is faster than the non-sparse version, but still slow
    # output = torch.zeros(y.shape[0], y.shape[1], output_length, device=x.device) # batch x output_channels x time'
    # for c in range(y.shape[1]):
    #     for t in range(y.shape[2]):
    #         output[:, c, :] += y[:, c, t, offsets[c, t]:offsets[c, t]+output_length]
    # y = output

    # Fourth attempt: Just use 1 Python loop
    # works but is still slow and dumb 
    output = torch.zeros(y.shape[0], y.shape[1], output_length, device=x.device) # batch x output_channels x time'
    for t in range(y.shape[2]):
        output[:, :, :] += y[:, :, t, offsets[:, t]:offsets[:, t]+output_length]
    y = output

    # sum in the original signal
    y += x[..., -y.shape[-1]:]
    # y += x_unpadded

    return y

def fractional_comb_fir_multitap_sparse_lowmem(x, f0, a, sr):
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

    # TODO make it work with more than 1 input channel
    assert f0.shape[1] == 1

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Construct the filters
    n_taps = 10
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    sinced = sparse_sinc(shifted_time)

    # out_channels*in_channels*n_taps x 4
    centers = (shifted_time.permute(1, 2, 0, 3).ceil()==0).argwhere()

    try:
        centers = centers[:, 3].reshape(f0.shape[0], f0.shape[1], n_taps) # out_channels x in_channels x n_taps
    except:
        import pdb; pdb.set_trace()

    if sinced.isnan().any():
        import pdb; pdb.set_trace()
    f = (gains * sinced).sum(0) # out_channels x in_channels x kernel_size

    x_unpadded = x

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    with torch.no_grad():
        f_mask = f!=0
        num_nonzero = (f_mask).sum(2).unique().item()
        kernel_block_size = num_nonzero//n_taps
        block_radius = kernel_block_size // 2
        output_length = x.shape[-1] - f.shape[-1] + 1
        assert kernel_block_size % 2 == 1 # TODO relax this constraint
        # TODO figure out input_channels...
        assert centers.shape[1] == 1
        centers = centers[:, 0] # output_channels x n_taps
        offsets = centers - block_radius # output_channels x n_taps
        offsets = offsets.permute(1, 0) # n_taps x output_channels

    f_condensed = f[f_mask].reshape(
        f.shape[0], f.shape[1], n_taps, kernel_block_size) # out_channels x in_channels x n_taps x kernel_block_size
    f_condensed = f_condensed.flip(2) # What? Why? Well, the above line puts the blocks in reverse order to tap index
    # f_condensed = f_condensed.permute(0, 2, 1, 3) # out_channels x n_taps x in_channels x kernel_block_size
    f_condensed = f_condensed.permute(2, 0, 1, 3) # n_taps x out_channels x in_channels x kernel_block_size

    y = torch.zeros((x.shape[0], f0.shape[0], output_length), device=x.device) # batch x output_channels x time'

    # for i in range(0, n_taps):
    #     temp_out = torch.nn.functional.conv1d(
    #         x,
    #         f_condensed[:, i],
    #     ) # batch x out_channels x time
    #     off = offsets[:, i]
    #     y += temp_out[:, :, off:off+output_length]

    for f_slice, off in zip(f_condensed, offsets):
        temp_out = torch.nn.functional.conv1d(
            x,
            f_slice,
        ) # batch x out_channels x time
        try:
            y += temp_out[..., off:off+output_length]
        except:
            import pdb; pdb.set_trace()

    # sum in the original signal
    # y += x[..., -y.shape[-1]:]
    y += x_unpadded

    return y

class _explicit_conv_loop(torch.autograd.Function):
    """
    Stands in for the following operation:
    ```
    for f_slice, off in zip(f_condensed, offsets):
        temp_out = torch.nn.functional.conv1d(
            x,
            f_slice,
        ) # batch x out_channels x time
        y += temp_out[..., off:off+output_length]
    ```
    """

    @staticmethod
    @torch.no_grad
    def forward(ctx, f_condensed, x, offsets, output_length):
        ctx.save_for_backward(f_condensed, x, offsets)
        ctx.output_length = output_length
        y = torch.zeros((x.shape[0], f_condensed.shape[1], output_length), device=x.device)
        for f_slice, off in zip(f_condensed, offsets):
            temp_out = torch.nn.functional.conv1d(
                x,
                f_slice,
                stride=1
            ) # batch x out_channels x time
            y += temp_out[..., off:off+output_length]
        ctx.full_output_length = temp_out.shape[-1]
        return y

    @staticmethod
    @torch.no_grad
    def backward(ctx, output_gradient: torch.Tensor):
        f_condensed, x, offsets = ctx.saved_tensors
        output_length = ctx.output_length
        input_length = output_length + f_condensed.shape[-1]-1

        # Attempt 1: use a loop here as well
        grad_f_condensed = torch.zeros_like(f_condensed, device=x.device)

        f_shape = f_condensed[0].shape
        for i, off in enumerate(offsets):
            grad_f_condensed[i] = torch.nn.grad.conv1d_weight(
                x[..., off:off+input_length],
                f_shape,
                output_gradient,
                stride=1
            )

        # Attempt 2: abuse unfold memory view
        # assert offsets.shape[1] == 1
        # x = x.unfold(2, input_length, step=1).flatten(1, 2)
        # x = x[:, offsets[:, 0]]
        # f_shape = (f_condensed.shape[1], f_condensed.shape[0], f_condensed.shape[3])
        # grad_f_condensed = torch.nn.grad.conv1d_weight(
        #     x,
        #     f_shape,
        #     output_gradient,
        #     stride=1
        # )
        # grad_f_condensed = grad_f_condensed.unflatten(1, (f_condensed.shape[0], -1)).permute(1, 0, 2, 3)

        # Attempt TODO: abuse unfold and regularity of taps
        # ... But first we have to enfore regularity of taps
        # assert offsets.shape[1] == 1
        # offset = -int(offsets.squeeze().diff().unique().item())
        # x = x.unfold(2, input_length, step=offset)

        # x = x[:, :, 1:offsets.shape[0]+1]
        # # x = x[:, :, 1:1+1]

        # x = x.flatten(1, 2)

        # f_shape = (f_condensed.shape[1], f_condensed.shape[0], f_condensed.shape[3])
        # # f_shape = (f_condensed.shape[1], 1, f_condensed.shape[3])

        # x = x.contiguous()
        # assert x.shape is not None
        # grad_f_condensed = torch.nn.grad.conv1d_weight(
        #     x,
        #     f_shape,
        #     output_gradient,
        #     stride=1
        # )
        # grad_f_condensed = grad_f_condensed.unflatten(1, (f_condensed.shape[0], -1)).permute(1, 0, 2, 3)



        # return None, None, None, None
        return grad_f_condensed, None, None, None

# def fractional_comb_fir_multitap_sparse_lowmem_convloop(x, f0, a, sr):
#     if x.dim() == 1: # time
#         x = x[None, None]
#     elif x.dim() == 2: # channels x time
#         x = x[None]

#     assert x.dim() == 3 # batch x channels x time

#     if not isinstance(f0, torch.Tensor):
#         f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
#     if f0.dim() == 0:
#         f0 = f0[None, None]
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
#     if a.dim() == 0:
#         a = a[None, None]

#     assert f0.dim() == 2 # out_channels x in_channels
#     assert a.dim() == 2 # out_channels x in_channels

#     # TODO make it work with more than 1 input channel
#     assert f0.shape[1] == 1

#     l = sr/f0 # out_channels x in_channels
#     t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

#     # Construct the filters
#     n_taps = 10
#     taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
#     delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
#     gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
#     time = t[None, None, None] # 1 x 1 x 1 x kernel_size
#     shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
#     sinced = sparse_sinc(shifted_time)

#     # out_channels*in_channels*n_taps x 4
#     centers = (shifted_time.permute(1, 2, 0, 3).ceil()==0).argwhere()

#     try:
#         centers = centers[:, 3].reshape(f0.shape[0], f0.shape[1], n_taps) # out_channels x in_channels x n_taps
#     except:
#         import pdb; pdb.set_trace()

#     if sinced.isnan().any():
#         import pdb; pdb.set_trace()
#     f = (gains * sinced).sum(0) # out_channels x in_channels x kernel_size

#     # x_unpadded = x

#     x = torch.nn.functional.pad(x, (sr//10-1, 0))

#     with torch.no_grad():
#         f_mask = f!=0
#         offsets = delays.squeeze().round()
#         num_nonzero = (f_mask).sum(2).unique().item()
#         kernel_block_size = num_nonzero//n_taps
#         block_radius = kernel_block_size // 2
#         output_length = x.shape[-1] - f.shape[-1] + 1
#         assert kernel_block_size % 2 == 1 # TODO relax this constraint
#         # TODO figure out input_channels...
#         assert centers.shape[1] == 1
#         centers = centers[:, 0] # output_channels x n_taps
#         offsets = centers - block_radius # output_channels x n_taps
#         offsets = offsets.permute(1, 0) # n_taps x output_channels

#     f_condensed = f[f_mask].reshape(
#         f.shape[0], f.shape[1], n_taps, kernel_block_size) # out_channels x in_channels x n_taps x kernel_block_size
#     f_condensed = f_condensed.flip(2) # What? Why? Well, the above line puts the blocks in reverse order to tap index
#     # f_condensed = f_condensed.permute(0, 2, 1, 3) # out_channels x n_taps x in_channels x kernel_block_size
#     f_condensed = f_condensed.permute(2, 0, 1, 3) # n_taps x out_channels x in_channels x kernel_block_size

#     y = _explicit_conv_loop.apply(f_condensed, x, offsets, output_length)

#     # sum in the original signal
#     y += x[..., -y.shape[-1]:]
#     # y += x_unpadded

#     return y

def fractional_comb_fir_multitap_sparse_lowmem_convloop(x, f0, a, sr):
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

    # TODO make it work with more than 1 input channel
    assert f0.shape[1] == 1

    l = sr/f0 # out_channels x in_channels

    # Construct the filters
    n_taps = 10
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    aligned_delays = taps * l.round()[None, ..., None] # n_taps x out_channels x in_channels x 1
    kernel_block_size = 9
    kernel_radius = kernel_block_size // 2
    kernel_block_indices = torch.arange(kernel_radius, -kernel_radius-1, -1, device=x.device)[None, None, None] # 1 x 1 x 1 x kernel_block_size
    sinced = torch.sinc(kernel_block_indices + aligned_delays - delays) # n_taps x out_channels x in_channels x kernel_block_size

    f = (gains * sinced) # n_taps x out_channels x in_channels x kernel_size

    kernel_size = sr//10
    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    output_length = x.shape[-1] - kernel_size + 1
    offsets = aligned_delays.to(int)[:, :, 0, 0]
    offsets = kernel_size - offsets - kernel_radius # n_taps x out_channels

    y = _explicit_conv_loop.apply(f, x, offsets, output_length) # batch x out_channels x time'

    # sum in the original signal
    y += x[..., -y.shape[-1]:]
    # y += x_unpadded

    return y

class _fractional_comb_fir_multitap_sparse_lowmem_explicit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, f0, a, sr):
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

        # TODO make it work with more than 1 input channel
        assert f0.shape[1] == 1

        # save these for later use
        ctx.save_for_backward(x, f0, a)
        ctx.sr = sr

        l = sr/f0 # out_channels x in_channels
        t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

        # Construct the filters
        n_taps = 2
        taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
        delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
        gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
        time = t[None, None, None] # 1 x 1 x 1 x kernel_size
        shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
        sinced = sparse_sinc(shifted_time)

        # out_channels*in_channels*n_taps x 4
        centers = (shifted_time.permute(1, 2, 0, 3).ceil()==0).argwhere()

        try:
            centers = centers[:, 3].reshape(f0.shape[0], f0.shape[1], n_taps) # out_channels x in_channels x n_taps
        except:
            import pdb; pdb.set_trace()

        if sinced.isnan().any():
            import pdb; pdb.set_trace()
        f = (gains * sinced).sum(0) # out_channels x in_channels x kernel_size

        x_unpadded = x

        x = torch.nn.functional.pad(x, (sr//10-1, 0))

        with torch.no_grad():
            f_mask = f!=0
            offsets = delays.squeeze().round()
            num_nonzero = (f_mask).sum(2).unique().item()
            kernel_block_size = num_nonzero//n_taps
            block_radius = kernel_block_size // 2
            output_length = x.shape[-1] - f.shape[-1] + 1
            assert kernel_block_size % 2 == 1 # TODO relax this constraint
            # TODO figure out input_channels...
            assert centers.shape[1] == 1
            centers = centers[:, 0] # output_channels x n_taps
            offsets = centers - block_radius # output_channels x n_taps
            offsets = offsets.permute(1, 0) # n_taps x output_channels

        f_condensed = f[f_mask].reshape(
            f.shape[0], f.shape[1], n_taps, kernel_block_size) # out_channels x in_channels x n_taps x kernel_block_size
        f_condensed = f_condensed.flip(2) # What? Why? Well, the above line puts the blocks in reverse order to tap index
        # f_condensed = f_condensed.permute(0, 2, 1, 3) # out_channels x n_taps x in_channels x kernel_block_size
        f_condensed = f_condensed.permute(2, 0, 1, 3) # n_taps x out_channels x in_channels x kernel_block_size

        y = torch.zeros((x.shape[0], f0.shape[0], output_length), device=x.device) # batch x output_channels x time'

        # for i in range(0, n_taps):
        #     temp_out = torch.nn.functional.conv1d(
        #         x,
        #         f_condensed[:, i],
        #     ) # batch x out_channels x time
        #     off = offsets[:, i]
        #     y += temp_out[:, :, off:off+output_length]

        for f_slice, off in zip(f_condensed, offsets):
            temp_out = torch.nn.functional.conv1d(
                x,
                f_slice,
            ) # batch x out_channels x time
            y += temp_out[..., off:off+output_length]

        # sum in the original signal
        # y += x[..., -y.shape[-1]:]
        y += x_unpadded

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # torch.nn.grad.conv1d_weight
        return None, None, None, None

fractional_comb_fir_multitap_sparse_lowmem_explicit = _fractional_comb_fir_multitap_sparse_lowmem_explicit.apply

def fractional_comb_fir_multitap_sparse_triton(x, f0, a, sr):
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

    # TODO make it work with more than 1 input channel
    assert f0.shape[1] == 1
    # TODO make it work with more than 1 output channel
    assert f0.shape[0] == 1

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Construct the filters
    n_taps = 10
    taps = torch.arange(1, n_taps+1, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    sinced = sparse_sinc(shifted_time)

    # out_channels*in_channels*n_taps x 4
    centers = (shifted_time.permute(1, 2, 0, 3).ceil()==0).argwhere()

    try:
        centers = centers[:, 3].reshape(f0.shape[0], f0.shape[1], n_taps) # out_channels x in_channels x n_taps
    except:
        import pdb; pdb.set_trace()

    if sinced.isnan().any():
        import pdb; pdb.set_trace()
    f = (gains * sinced).sum(0) # out_channels x in_channels x kernel_size
    # f[..., -1] = 1. # original signal (x[i]) # not needed if we just sum the original signal in later

    x_unpadded = x
    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = combnet.SparseConv1d.forward(x.to(torch.float16), f.to(torch.float16), stride=1).to(torch.float32)

    y += x_unpadded

    return y


def single_fractional_comb_modulo(x, f0, a, sr):
    x = x.squeeze()
    l = sr/f0
    # l = (sr//f0)
    # import pdb; pdb.set_trace()
    t = torch.arange(0, sr//20)

    # f = t % l
    f = torch.remainder(t, l)
    a = (a ** (t/l))

    # f = (torch.sinc(f)) * a
    import pdb; pdb.set_trace()
    f = torch.sinc(f) * a
    # f = torch.sinc(f - f % 1.) * a

    # from matplotlib import pyplot as plt
    # plt.plot(f); plt.gcf().set_size_inches(10, 7.5); plt.show()

    # f = torch.flip(f, (0,))

    return convolve(x, f)

def single_comb_iir_fast(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    step = l
    for i in range(l, x.shape[-1]-step, step):
        y[..., i:i+step] += a*y[..., i-l:i-l+step]
    return y

def single_fractional_comb_iir_faithful(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device) - l).to(x.device)
    f = a * torch.sinc( t) #* torch.exp( -.1 * t**2)
    # f[0] = 1 # a[0] should be 1!
    f = torch.flip(f, (-1,))
    size = sr//10
    y = torch.zeros_like(x)
    y = torch.nn.functional.pad(y, (size-1, 0))
    for i in range(0, x.shape[-1]):
        # if i > 48:
        #     import pdb; pdb.set_trace()
        y[..., i+size-1] = (y[..., i:i+size] * f[None, None, :]).sum() + x[..., i]
    return y[..., size-1:]

# Take 2: Use an IIR comb filter, but apply using spectral division
# Much faster than above, and has a controlled DC bump, but a little hacky of course
def single_fractional_comb_fiir( x, f0, a, sr):
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device) - l).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) / torch.fft.rfft( f, n=l))[:x.shape[-1]] # change to fast conv instead?

def fractional_comb_fiir(x, f0, a, sr):
    if x.dim() == 2: # audio_channels x time
        x = x[None, None]
    if x.dim() == 3: # batch x audio_channels x time
        x = x[:, None]
    assert x.dim() == 4 # batch x feature_channels x audio_channels x time
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None, None]
    assert f0.dim() == 2
    # TODO something might be wrong here with the shape of `a`
    assert a.dim() == 0 or a.dim() == 2
    if a.dim() == 2:
        a = a[..., None]
    # Make the filter
    l = sr/f0
    t = (torch.arange(sr//10, device=x.device)[None, None, :] - l[..., None]).to(x.device)
    f = -a * torch.sinc(t) * torch.exp(-.1 * t**2)
    f[..., 0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones(l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # from matplotlib import pyplot as plt
    # plt.plot(1/abs(torch.fft.rfft( f, n=l).squeeze())); plt.show()
    x_fft = torch.fft.rfft(x, n=l)
    f_fft = torch.fft.rfft(f, n=l)
    filtered_fft = (lp * x_fft / f_fft[None]).sum(2)
    return torch.fft.irfft(filtered_fft)[:x.shape[-1]] # change to fast conv instead?

def fractional_anticomb_interference_fiir(x, f0, a, sr, residual_mode=False):
    assert x.dim() == 3
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if a.dim() == 2:
        a = a[..., None]
    assert f0.dim() == 2
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[None, None, :] - l[..., None]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[..., 0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    filter = torch.fft.rfft(f, n=l)
    if residual_mode:
        filter = filter.cumprod(0)
        filter = torch.cat([torch.ones(1, filter.shape[1], filter.shape[2], device=filter.device), filter[:-1]], 0)
    else:
        idx = ~torch.eye(f0.shape[0], dtype=bool, device=filter.device)
        idx = idx.unsqueeze(2).repeat(1, 1, f0.shape[1])
        filter = filter.unsqueeze(0).repeat(f0.shape[0], 1, 1, 1)
        filter = filter[idx].view(f0.shape[0]-1, f0.shape[0], f0.shape[1], filter.shape[-1])
        filter = filter.prod(0)
    # filter = filter.prod(0, keepdims=True) / filter #TODO might cause gradient problems
    # scale to compensate for multiple filters
    if filter.shape[0] >= 3:
        filter = filter ** (1/(filter.shape[0]-1)) #TODO check if necessary
    return torch.fft.irfft(((lp * torch.fft.rfft( x, n=l))[:, None] * filter[None]))[:x.shape[-1]] # change to fast conv instead?

def fractional_anitcomb_fiir( x, f0, a, sr):
    # Make the filter
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None]
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[:, None] - l[None, :]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!
    # import pdb; pdb.set_trace()
    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # f /= abs(f).max()
    spectral_filter = torch.fft.rfft( f.T, n=l).prod(0)
    # from matplotlib import pyplot as plt
    # import pdb; pdb.set_trace()
    # spectral_filter /= abs(spectral_filter).max()
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) * spectral_filter)[:x.shape[-1]] # change to fast conv instead?

def single_fractional_comb_anti_fiir( x, f0, a, sr):
    # Make the filter
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None]
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[:, None] - l[None, :]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!
    # import pdb; pdb.set_trace()
    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # f /= abs(f).max()
    spectral_filter = torch.fft.rfft( f.T, n=l).prod(0)
    # from matplotlib import pyplot as plt
    # import pdb; pdb.set_trace()
    # spectral_filter /= abs(spectral_filter).max()
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) * spectral_filter)[:x.shape[-1]] # change to fast conv instead?


# Take 3: Use an explicit FIR comb filter
# f0 is intimately tied to a for loop, so this does not have a gradient
def single_fractional_comb_fir( x, f0, a, sr):
    n = 640 #int( 768*a)
    t = torch.linspace( 0, 2*torch.pi*n/sr, n+1)[:-1]

    # Take 1, straightforward take
    a = 0
    for f in torch.arange( f0, sr/2, f0):
        a += torch.cos( f*t)

    # Soften up the filter and use convolution
    a *= torch.hann_window( n)
    return convolve( x, a)




# Take 4: Use an FIR Dirichlet kernel comb filter
# Works fine, but is numerically unstable around k*pi even after adding a bunch of tricks
def single_fractional_comb_diric( x, f0, a, sr):
    T = 1025 #(640*a).astype( int)

    x = x.squeeze()

    # Vanilla Dirichlet kernel implementation (peaks galore!)
    def diric( x, N):
        x = x % (2*torch.pi)
        return torch.sin( N*x) / (N*torch.sin( x))

    # Replace k*pi areas with a polynomial approximation to avoid numerical instability
    def diric2( x, N):
        x = x % (2*torch.pi)
        th = .001
        c1 = (x<th)
        c2 = abs(x-torch.pi)<th
        c3 = (x > (2*torch.pi-th))
        c4 = ~(c1|c2|c3)
        return c1 * (1+(1-N**2)*x**2/6) + \
            c2 * (1+(1-N**2)*(x-torch.pi)**2/6) + \
            c3 * (1+(1-N**2)*(x-2*torch.pi)**2/6) + \
            c4 * torch.sin( N*x) / (N*torch.sin( x))

    # Auto-select n to not alias
    n = int( 2*((sr/f0)//2)-1)

    # r = torch.arange( 0, T*torch.pi, torch.pi).to(x.device)
    a = diric2( f0*(torch.arange( 0, T*torch.pi, torch.pi, device=x.device)+1e-3)/sr, n) * torch.kaiser_window( T, False, beta=16., device=x.device)
    # a = diric2( f0*(torch.arange( 0, T*torch.pi, torch.pi)+1e-3)/sr, n)
    # a = diric( f0*(r+1e-3)/sr, n)
    # from matplotlib import pyplot as plt
    # plt.show()
    # plt.plot(a); plt.show()
    return convolve( x, a)


# Take 5: Use an overdriven sine as the filter kernel
# Cannot control the harmonics well, but is numerically stable
def single_fractional_comb_fir_od( x, f0, a, sr):
    n = 768
    t = torch.linspace( 0, 2*torch.pi*n/sr, n+1)[:-1]

    a = torch.tanh( 4*torch.cos( f0*t))

    a *= torch.hanning( n)
    return convolve( x, a, mode='same')