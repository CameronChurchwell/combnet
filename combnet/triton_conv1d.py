import torch
from torch._inductor.graph import GraphLowering # THIS CANNOT BE REMOVED. PAIN.
from torch._inductor.kernel.conv import convolution
import triton
import triton.language as tl

# unpack the given idx given the order of axis of the desired 3-dim tensor
# You could view it as the reverse of flatten the idx of 3 axis in a tensor to 1-dim idx.
# order is the order of axes in tensor, innermost dimension outward
# shape is the 3D tensor's shape
# (now adapted to 2D)
def _unpack(idx, order, shape):
    # if torch.is_tensor(idx):
    _12 = torch.div(idx, shape[order[0]], rounding_mode="trunc")
    _0 = idx % shape[order[0]]
    # _2 = torch.div(_12, shape[order[1]], rounding_mode="trunc")
    _1 = _12 % shape[order[1]]
    # else:
    #     _12 = idx // shape[order[0]]
    #     _0 = idx % shape[order[0]]
    #     _2 = _12 // shape[order[1]]
    #     _1 = _12 % shape[order[1]]
    # return _0, _1, _2
    return _0, _1

import pdb

@triton.autotune(configs=[
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=2
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=2
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=2
    # ),

    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=2
    ),
    # triton.Config(
    #     {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=2
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=2
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=2
    # ),

    # triton.Config(
    #     {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=4
    # ),

    # triton.Config(
    #     {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=8
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=4, num_warps=8
    # ),

    # triton.Config(
    #     {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=2, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=2, num_warps=4
    # ),
    # triton.Config(
    #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=2, num_warps=4
    # ),
    ],
    key=[
        'SPARSE_KERNEL_T',
        'OUT_T',
    ]
)
@triton.jit
def _kernel_sparse(
    x,
    w,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xt,
    stride_wn,
    stride_wc,
    stride_wt,
    stride_yn,
    stride_yc,
    stride_yt,
    # pointer inc for x
    delta_x_ptr,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_T,
    KERNEL_N,
    KERNEL_T,
    SPARSE_KERNEL_T,
    OUT_T,
    # parameters of conv
    stride,
    output_padding,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_T] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nt = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nt = pid_nt * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nt // OUT_T
    off_y_t = off_y_nt % OUT_T

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_t = off_y_t * stride
    off_x_nt = off_x_n * stride_xn + off_x_t * stride_xt
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_T
    CRS_ACTUAL = IN_C * SPARSE_KERNEL_T
    # load inc ptr of x, upade x_ptrs
    delta_x_ptrs = delta_x_ptr + off_x_crs
    off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=(off_x_crs < CRS_ACTUAL))
    x_ptrs = x + off_x_nt[:, None] + off_x_crs_unpacked[None, :]

    mask_x = (
        (off_x_n < BATCH)
        & (off_x_t >= 0)
        & (off_x_t < IN_T)
    )[:, None] & (off_x_crs < CRS_ACTUAL)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS_ACTUAL)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0) # MxK
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0) # KxN

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
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS_ACTUAL, other=0)
        x_ptrs = x + off_x_nt[:, None] + off_x_crs_unpacked[None, :]

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_t >= 0)
            & (off_x_t < IN_T)
        )[:, None] & (off_x_crs < CRS_ACTUAL)[None, :]
        mask_w = (off_x_crs < CRS_ACTUAL)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nt = pid_nt * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nt // (OUT_T)
    off_y_t = off_y_nt % (OUT_T)
    # consider output padding

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_t[:, None] * stride_yt
        + off_y_k[None, :] * stride_yc
    )

    # # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_t < OUT_T + output_padding)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

    return


class SparseConv1d:
    kernel = _kernel_sparse

    # for the contigous order of w ptr, what"s the corresponding
    # ptr changes for x in a sliding window
    @staticmethod
    def _delta_x_ptr(
        IN_C,
        KERNEL_T,
        stride_wc,
        stride_wt,
        stride_xc,
        stride_xt,
        device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_2d = [stride_wc, stride_wt]
        order = sorted(range(len(stride_w_2d)), key=stride_w_2d.__getitem__)
        window_size = IN_C * KERNEL_T

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_T])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_t = window_unpack[order[1]]
        r_inc = window_unpack_c
        delta_x = (
            window_unpack_t * stride_xt + r_inc * stride_xc
        )
        return delta_x

    @staticmethod
    def _call(
        x,
        w,
        stride: int,
        groups: int = 1,
        output_padding: int = 0,
    ):
        torch.cuda.set_device(x.device)
        kernel_t = None
        delta_x = None

        # Sparsify
        # kernel_t = w.shape[-1]
        # binary_mask = w!=0
        # w = w[binary_mask].reshape(w.shape[0], w.shape[1], -1)
        # aw = binary_mask.argwhere()
        # delta_x = aw[:, 2] + aw[:, 1] * x.stride()[1]

        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape

        # indicies for the layeout
        xn, xc, xt = 0, 1, 2
        yn, yc, yt = 0, 1, 2
        wn, wc, wt = 0, 1, 2

        # out_channel, in_channel, kernel_height, kernel_width
        if kernel_t is not None:
            kernel_length = kernel_t
        else:
            kernel_length = shape_w[wt]
        input_length = shape_x[xt]
        in_channel = shape_w[wc] * groups

        assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
        assert (
            shape_x[xc] == in_channel
        ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

        # output shape
        shape_y = [0] * 3
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        # shape_y[yt] = (
        #     input_length * (kernel_length - 1)
        #     - 1
        #     + stride
        # ) // stride + 2 * output_padding
        shape_y[yt] = (input_length - kernel_length) // stride + 1

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_T = shape_x[xt]
        KERNEL_N = shape_w[wn]
        SPARSE_KERNEL_T = shape_w[wt]
        if kernel_t is not None:
            KERNEL_T = kernel_t
        else:
            KERNEL_T = shape_w[wt]
        OUT_T = shape_y[yt]

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()
        stride_y = y.stride()

        # allocate tmp
        # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
        # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
        # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
        # accumulator types
        ACC_TYPE = (
            tl.float32
            if x.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )
        if delta_x is None:
            delta_x = SparseConv1d._delta_x_ptr(
                IN_C,
                KERNEL_T,
                stride_w[wc],
                stride_w[wt],
                stride_x[xc],
                stride_x[xt],
                device,
            )

        # launch kernel, 2-dim, batch*h*w, kernel
        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_T, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
            )

        _kernel_sparse[grid](
            x,
            w,
            y,
            # stride nchw for x,w,y tensor
            stride_x[xn],
            stride_x[xc],
            stride_x[xt],
            stride_w[wn],
            stride_w[wc],
            stride_w[wt],
            stride_y[yn],
            stride_y[yc],
            stride_y[yt],
            # pointer inc for x
            delta_x,
            # Tensor dimensions
            BATCH,
            IN_C,
            IN_T,
            KERNEL_N,
            KERNEL_T,
            SPARSE_KERNEL_T,
            OUT_T,
            # conv parameters
            stride,
            output_padding,
            groups,
            # Metaparameters
            ACC_TYPE=ACC_TYPE,
            # BLOCK_M=32,
            # BLOCK_N=32,
            # BLOCK_K=32,
        )
        return y

    @staticmethod
    def forward(
        x,
        w,
        stride=1,
    ):
        return SparseConv1d._call(
            x,
            w,
            stride,
        )


conv = SparseConv1d.forward

# @triton.jit
# def _dumb(x_ptr):
#     x_indices = tl.arange(0, 32)
#     x_indices = x_indices[None] * 32 + x_indices[:, None]
#     x = tl.load(x_ptr + x_indices)
#     x = tl.dot(x, x)
#     tl.store(x_ptr + x_indices, x)

if __name__ == '__main__':
    breakpoint()
    device = torch.device('cuda:0')
    # x = torch.randn(1, 1, 4096*1024, device=device, dtype=torch.float16)
    # w = torch.randn(1, 1, 16, device=device, dtype=torch.float16)

    x = torch.tensor([
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 2, 3, 4, 5, 6, 7, 8]
    ], device=device, dtype=torch.float16)[None]
    w = torch.tensor([
        [1, 0, 0, 2],
        [0, 3, 0, -1],
    ], device=device, dtype=torch.float16)[None]

    print(x.shape, w.shape)

    stride = 1

    print(x.shape, w.shape)

    output_padding = 0
    groups = 1

    y_target = torch.nn.functional.conv1d(
        x,
        w,
        stride=stride,
        groups=groups
    )

    # full_kernel_t = w.shape[-1]
    # binary_mask = w!=0
    # w_dense = w[binary_mask].reshape(w.shape[0], w.shape[1], -1)
    # aw = binary_mask.argwhere()
    # channels = aw[:, 1]
    # delta_x = aw[:, 2] + aw[:, 1] * x.stride()[1]

    # y_torch = convolution(
    #     x,
    #     w,
    #     bias: TensorBox,
    #     stride: List[int],
    #     padding: List[int],
    #     dilation: List[int],
    #     transposed: bool,
    #     output_padding: List[int],
    #     groups: int,
    # ):

    y = conv(
        x,
        w,
        stride=stride,
    )

    # grid = (1,)
    # x = torch.randn(32, 32, device=device).to(torch.float16)
    # print(x)
    # print(x@x)
    # _dumb[grid](x)
    # print(x)

    print(y_target.shape, y.shape)

    print(y_target)
    print(y)
    # print(abs(y_target-y).max(), abs(y_target-y).argmax(), abs(y_target-y).mean())

    # breakpoint()