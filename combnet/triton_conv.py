import torch
import triton
import triton.language as tl
from time import time

# unpack the given idx given the order of axis of the desired 3-dim tensor
# You could view it as the reverse of flatten the idx of 3 axis in a tensor to 1-dim idx.
# order is the order of axes in tensor, innermost dimension outward
# shape is the 3D tensor's shape
def _unpack(idx, order, shape):
    if torch.is_tensor(idx):
        _12 = torch.div(idx, shape[order[0]], rounding_mode="trunc")
        _0 = idx % shape[order[0]]
        _2 = torch.div(_12, shape[order[1]], rounding_mode="trunc")
        _1 = _12 % shape[order[1]]
    else:
        _12 = idx // shape[order[0]]
        _0 = idx % shape[order[0]]
        _2 = _12 // shape[order[1]]
        _1 = _12 % shape[order[1]]
    return _0, _1, _2

@triton.jit
def _kernel_delta_x(
    x,
    w,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    # pointer inc for x
    delta_x_ptr,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h,
    stride_w,
    output_padding_h,
    output_padding_w,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h
    off_x_w = off_y_w * stride_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    delta_x_ptrs = delta_x_ptr + off_x_crs
    off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
    x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]

    mask_x = (
        (off_x_n < BATCH)
        & (off_x_h >= 0)
        & (off_x_h < IN_H)
        & (off_x_w >= 0)
        & (off_x_w < IN_W)
    )[:, None] & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        delta_x_ptrs += BLOCK_K
        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

@triton.jit
def _kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    x_channel_stride, x_time_stride,
    w_out_stride, w_in_stride, w_kernel_stride,
    y_channel_stride, y_time_stride,
    output_length,
    INPUT_CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)
    block_start = block_id * BLOCK_SIZE

    kernel_indices = tl.arange(0, KERNEL_SIZE)
    block_offsets = tl.arange(0, BLOCK_SIZE)
    input_channel_indices = tl.arange(0, INPUT_CHANNELS)

    # w = tl.load(w_ptr + kernel_indices)[:, None]
    w_indices = kernel_indices[None] + input_channel_indices[:, None] * w_in_stride # input_channels x kernel_size
    w = tl.load(w_ptr + w_indices)

    # indices for x loading
    x_time_indices = block_start * x_time_stride + kernel_indices[None, :, None] + block_offsets[None, None] * x_time_stride
    x_indices = x_time_indices + input_channel_indices[:, None, None] * x_channel_stride
    # mask for x loading
    # mask = x_indices > 0
    # mask = (block_start + block_offsets[None, None]) < output_length
    x = tl.load(x_ptr + x_indices)

    tl.static_print(w.shape, x.shape)

    output = tl.sum(tl.sum(w[:, :, None]*x, 0), 0)

    tl.static_print(output.shape)

    tl.store(y_ptr + block_start + block_offsets, output)


def conv(
    x: torch.Tensor, # length
    w: torch.Tensor, # kernel_size
    stride: int = 1,
):
    torch.cuda.set_device(x.device)

    output_channels, input_channels, kernel_size = w.shape
    _, length = x.shape

    assert x.shape[0] == input_channels

    assert kernel_size <= length
    output_length = (length-kernel_size)//stride + 1
    y = torch.zeros(output_channels, output_length, device=x.device)

    # compute strides
    #TODO some of these will be 1 and are therefore unnecessary
    x_channel_stride, x_time_stride = x.stride()
    w_out_stride, w_in_stride, w_kernel_stride = w.stride()
    y_channel_stride, y_time_stride = y.stride()
    x_time_stride *= stride

    grid = lambda meta: (output_channels, triton.cdiv(output_length, meta['BLOCK_SIZE']),)
    # grid = lambda meta: (1, triton.cdiv(output_length, meta['BLOCK_SIZE']),)
    # grid = (1, 1,)
    # grid = (output_channels, 1)
    _kernel[grid](
        x_ptr=x,
        w_ptr=w,
        y_ptr=y,
        x_channel_stride=x_channel_stride, x_time_stride=x_time_stride,
        w_out_stride=w_out_stride, w_in_stride=w_in_stride, w_kernel_stride=w_kernel_stride,
        y_channel_stride=y_channel_stride, y_time_stride=y_time_stride,
        output_length=output_length,
        INPUT_CHANNELS=input_channels,
        KERNEL_SIZE=kernel_size,
        BLOCK_SIZE=1024
    )

    return y


if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.randn(1, 2, 4096*1024).to(device)
    w = torch.randn(1, 2, 16).to(device)

    print(x.shape, w.shape)

    stride = 2

    torch.cuda.synchronize()
    start = time()
    y_target = torch.nn.functional.conv1d(
        x,
        w,
        stride=stride,
    )
    torch.cuda.synchronize()
    end = time()
    print(f'torch: {end-start}')

    torch.cuda.synchronize()
    start = time()
    y = conv(
        x[0],
        w,
        stride=stride,
    )
    torch.cuda.synchronize()
    end = time()
    print(f'triton: {end-start}')

    # print(f"max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024:0.3f} GB")

    print(y_target)
    print(y)


    # breakpoint()