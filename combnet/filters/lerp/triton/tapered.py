import torch
import triton
import triton.language as tl

@triton.jit
def _lerp_forward_kernel_fused_tapered(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    y, # batch x out_channels x out_time
    m, # batch x out_channels x out_time
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
    result_indices = tl.zeros((block_batch, block_out_channels), dtype=tl.int32)
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
                # left taper for l0
                tap_indices = indices - floor_delays - 1 # 1 x block_out_channels x block_in_channels x block_time
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
                x_tile = tl.load(x + x_indices, mask) * ((1-k)*gains) / 2
                accumulator += tl.sum(x_tile, 2)
                # right taper for l0
                tap_indices = indices - floor_delays + 1 # 1 x block_out_channels x block_in_channels x block_time
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
                x_tile = tl.load(x + x_indices, mask) * ((1-k)*gains) / 2
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
                # left taper for l1
                # apply delays to indices
                tap_indices = indices - ceil_delays - 1 # 1 x block_out_channels x block_in_channels x block_time
                # apply channel and batch and channel offsets to indices
                x_indices = tap_indices + in_channel_indices * time \
                    + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
                # create mask based on indices components and bounds
                mask = (tap_indices >= 0) \
                    & (tap_indices < time) \
                    & (in_channel_indices < in_channels) \
                    & (batch_indices < batch_size) \
                    & (indices < t_id * stride + window_size)
                x_tile = tl.load(x + x_indices, mask) * (k*gains) / 2 # block_out_channels x block_in_channels x block_time
                accumulator += tl.sum(x_tile, 2)
                # right taper for l1
                # apply delays to indices
                tap_indices = indices - ceil_delays + 1 # 1 x block_out_channels x block_in_channels x block_time
                # apply channel and batch and channel offsets to indices
                x_indices = tap_indices + in_channel_indices * time \
                    + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
                # create mask based on indices components and bounds
                mask = (tap_indices >= 0) \
                    & (tap_indices < time) \
                    & (in_channel_indices < in_channels) \
                    & (batch_indices < batch_size) \
                    & (indices < t_id * stride + window_size)
                x_tile = tl.load(x + x_indices, mask) * (k*gains) / 2 # block_out_channels x block_in_channels x block_time
                accumulator += tl.sum(x_tile, 2)

                gains *= base_gains

        # compute max over block_time chunk and compare/store with result
        max_values = tl.max(tl.abs(accumulator), axis=2)
        max_indices = tl.argmax(tl.abs(accumulator), axis=2)
        result_indices = tl.where(max_values > result, max_indices, result_indices) # this line must go above the following line
        result = tl.maximum(max_values, result)

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

    # result indices are with respect to the start of the window...
    # so we add the offset to the start of the window

    result_indices += t_id * stride

    tl.store(y+y_indices, result[:, :, None, None], y_mask)
    tl.store(m+y_indices, result_indices[:, :, None, None], y_mask)

@triton.jit
def _lerp_backward_kernel_fused_tapered(
    x, # batch x in_channels x time
    l, # out_channels x in_channels
    a, # out_channels x in_channels
    gradient, # batch x out_channels x in_channels x out_time
    pool_indices, # batch x out_channels x out_time
    batch_size: int,
    n_taps: int,
    out_channels: int,
    in_channels: int,
    time: int,
    out_time: int,
    block_batch: tl.constexpr = 4,
    block_in_channels: tl.constexpr = 1,
    block_out_channels: tl.constexpr = 16,
    # block_time: tl.constexpr = 512,
):
    """
    Computes a `block_batch x block_out_channels x block_in_channels x 1` block of gradient
    """
    n_id = tl.program_id(0)
    o_id = tl.program_id(1)
    t_id = tl.program_id(2)

    accumulator = tl.zeros((block_batch, block_out_channels, block_in_channels, 1), dtype=tl.float32)

    batch_indices = tl.arange(0, block_batch)[:, None, None, None] + n_id * block_batch # block_batch x 1 x 1 x 1
    out_channel_indices = tl.arange(0, block_out_channels)[None, :, None, None] + o_id * block_out_channels # 1 x block_out_channels x 1 x 1

    # indices = tl.arange(0, block_time)[None, None, None] + t_id * block_time # 1 x 1 x 1 x block_time
    idx = out_channel_indices * out_time + batch_indices * out_channels * out_time + t_id
    pool_mask = (out_channel_indices < out_channels) & (batch_indices < batch_size)
    indices = tl.load(pool_indices+idx, pool_mask)

    # tl.device_print('indices', indices)
    # tl.device_assert(False)

    for ic_counter in range(0, in_channels, block_in_channels):
        in_channel_indices = tl.arange(0, block_in_channels)[None, None, :, None] + ic_counter * block_in_channels # 1 x 1 x block_in_channels x 1
        channel_indices = out_channel_indices * in_channels + in_channel_indices # 1 x block_out_channels x block_in_channels x 1
        channel_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels)
        delays = tl.load(l + channel_indices, channel_mask) # 1 x block_out_channels x block_in_channels x 1
        gains = tl.load(a + channel_indices, channel_mask, 1) # 1 x block_out_channels x block_in_channels x 1
        base_gains = tl.load(a + channel_indices, channel_mask, 1) # 1 x block_out_channels x block_in_channels x 1
        tl.device_assert(gains > 0, 'gains must be > 0')

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
            x_tile = tl.load(x + x_indices, mask) * tap * gains
            accumulator -= x_tile
            # left taper for l0
            tap_indices = indices - floor_delays - 1 # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * tap * gains / 2
            accumulator -= x_tile
            # right taper for l0
            tap_indices = indices - floor_delays + 1 # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * tap * gains / 2
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
            x_tile = tl.load(x + x_indices, mask) * tap * gains # block_out_channels x block_in_channels x block_time
            accumulator += x_tile
            # left taper for l1
            # apply delays to indices
            tap_indices = indices - ceil_delays - 1# 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * tap * gains / 2 # block_out_channels x block_in_channels x block_time
            accumulator += x_tile
            # right taper for l1
            # apply delays to indices
            tap_indices = indices - ceil_delays + 1 # 1 x block_out_channels x block_in_channels x block_time
            # apply channel and batch and channel offsets to indices
            x_indices = tap_indices + in_channel_indices * time \
                + batch_indices * in_channels * time # block_batch x block_out_channels x block_in_channels x block_time
            # create mask based on indices components and bounds
            mask = (tap_indices >= 0) \
                & (tap_indices < time) \
                & (in_channel_indices < in_channels) \
                & (batch_indices < batch_size)
            x_tile = tl.load(x + x_indices, mask) * tap * gains / 2 # block_out_channels x block_in_channels x block_time
            accumulator += x_tile

            gains *= base_gains

        # og_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
        # og_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # og_tile = tl.load(output_gradient + og_indices, og_mask)

        # partial_gradient = tl.sum(tl.sum(accumulator * og_tile, 3, keep_dims=True), 0, keep_dims=True)

        # g_indices = in_channel_indices + out_channel_indices * in_channels
        # g_mask = (out_channel_indices < out_channels) & (in_channel_indices < in_channels)
        # tl.atomic_add(gradient+g_indices, partial_gradient, g_mask)

        g_indices = t_id + in_channel_indices * out_time + out_channel_indices * in_channels * out_time + batch_indices * out_channels * in_channels * out_time
        g_mask = (in_channel_indices < in_channels) & (out_channel_indices < out_channels) & (batch_indices < batch_size)
        # tl.atomic_add(gradient+g_indices, accumulator, g_mask, 'relaxed') # relaxed seems to be faster with no downsides
        tl.atomic_add(gradient+g_indices, accumulator, g_mask)


    # store tile in y
    # y_indices = indices + out_channel_indices * time + batch_indices * out_channels * time
    # if t_id == 0:
    #     tl.device_print('y_indices', y_indices)
    # y_mask = (indices < time) & (out_channel_indices < out_channels) & (batch_indices < batch_size)

    # tl.store(y+y_indices, accumulator[:, :, None, :], y_mask)

class _explicit_lerp_triton_fused_tapered(torch.autograd.Function):
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
        m: torch.Tensor,
        a: torch.Tensor,
        l: torch.Tensor,
        n_taps,
        window_size,
        stride
    ):
        # TODO this is not strong enough, need to check strides multiply up to dims
        assert x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and l.is_contiguous() and m.is_contiguous()
        assert x.device == y.device == a.device == l.device
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
        _lerp_forward_kernel_fused_tapered[grid](
            x=x, # batch x in_channels x time
            l=l, # out_channels x in_channels
            a=a, # out_channels x in_channels
            y=y, # batch x out_channels x time
            m=m, # batch x out_channels x time
            batch_size=y.shape[0],
            n_taps=n_taps,
            out_channels=y.shape[1],
            in_channels=x.shape[1],
            time=x.shape[-1],
            out_time=y.shape[-1],
            window_size=window_size,
            stride=stride,
        )
        ctx.save_for_backward(x, a, l, m)
        return y

    @staticmethod
    @torch.no_grad
    def backward(ctx, output_gradient: torch.Tensor):
        x, a, l, m = ctx.saved_tensors
        n_taps = ctx.n_taps

        # batch x out_channels x in_channels x out_time
        dy_dl = torch.zeros((x.shape[0], l.shape[0], l.shape[1], output_gradient.shape[2]), device=l.device, dtype=l.dtype)

        def grid(META):
            grid_shape = (
                triton.cdiv(x.shape[0], META["block_batch"]),
                triton.cdiv(l.shape[0], META["block_out_channels"]),
                # triton.cdiv(x.shape[-1], META["block_time"])
                m.shape[2]
            )
            return grid_shape
        _lerp_backward_kernel_fused_tapered[grid](
            x=x, # batch x in_channels x time
            l=l, # out_channels x in_channels
            a=a, # out_channels x in_channels
            pool_indices=m, # batch x out_channels x out_time
            gradient=dy_dl, # batch x out_channels x out_time
            # output_gradient=output_gradient.contiguous(),
            batch_size=x.shape[0],
            n_taps=n_taps,
            out_channels=l.shape[0],
            in_channels=l.shape[1],
            time=x.shape[-1],
            out_time = m.shape[-1],
        )

        # dLoss_dl = torch.einsum('not,noit->oi', output_gradient, dy_dl)
        dLoss_dl = torch.einsum('nop,noip->oi', output_gradient, dy_dl)

        # This line is so dumb
        return None, None, None, None, dLoss_dl, None, None, None

def fractional_comb_fir_multitap_lerp_explicit_triton_fused_tapered(x, f0, a, sr, window_size=None, stride=None, n_taps=10):
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
    out_time = (x.shape[-1] - window_size) // stride + 1
    y = torch.zeros(x.shape[0], f0.shape[0], out_time, device=x.device, dtype=x.dtype) # batch x out_channels x time
    m = torch.zeros(x.shape[0], f0.shape[0], out_time, device=x.device, dtype=torch.int32) # batch x out_channels x time
    y = _explicit_lerp_triton_fused_tapered.apply(x, y, m, a, l, n_taps, window_size, stride)
    return y