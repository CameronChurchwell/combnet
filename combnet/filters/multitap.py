import torch
import triton
import triton.language as tl

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
            # calculate delays
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
    f = (gains * torch.sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

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
    f = (gains * torch.sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

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

def sparse_sinc(x):
    sinced = torch.sinc(x)
    outputs = torch.where(
        (-4.5<x) & (x<=4.5),
        # (-1.5<x) & (x<=1.5),
        sinced,
        0.
    )
    return outputs

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

    # TODO generalize?
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

    assert (num_nonzero//n_taps) % 2 == 1
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
        assert centers.shape[1] == 1 # TODO generalize?
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