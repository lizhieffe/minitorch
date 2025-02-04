# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, Tuple

import numba

# Required to use cuda in numba. https://github.com/googlecolab/colabtools/issues/5081
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

from numba import cuda
from numba.cuda import jit as _jit
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import Function
from .fast_conv import tensor_conv1d
import time

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

# Cuda has limit of max 1024 threads per block. Here we set this to 8 because we are building 3D block.
THREADS_PER_BLOCK = 8

def _tensor_conv1d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """

  
    The grid x, y, z axis corresponds to the [B, T, COUT] dim of the output.


    """

    # Get shapes
    #
    # input: `batch, in_channels, width`
    # weight: `out_channels, in_channels, k_width`
    # output: `batch, out_channels, width`
    #
    batch, in_channels, width = input_shape
    out_channels, _, k_width = weight_shape
    _, _, out_width = out_shape
    # Note, width and out_width doesn't necessarily to be equal. E.g. in the 
    # first _tensor_conv1d_cuda() call in the backward(), the width >= out_width.
    assert batch == out_shape[0]
    assert out_channels == out_shape[1]
    assert in_channels == weight_shape[1]

    # TODO: why
    input_batch_stride = input_strides[0] if input_shape[0] > 1 else 0

    # get the index in the grid
    # [B, T, COUT]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y
    pk = cuda.threadIdx.z

    if j >= out_width:
        return

    # Allocate shared memory.
    BLOCK_DIM = 8
    input_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), numba.float64) # [B, T, KW * C_IN]
    weight_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64) # [KW * C_IN, C_OUT]

    total = 0.0
    for conv_i in range(0, k_width * in_channels, BLOCK_DIM):        

        if pk + conv_i < k_width * in_channels and i < batch:
            k_width_i = (pk + conv_i) // in_channels
            in_channels_i = (pk + conv_i) % in_channels

            # Make sure the conv doesn't go out bound of T
            if reverse:
                k_width_i = k_width - k_width_i - 1

                if j - k_width_i >= 0:
                    input_pos = input_batch_stride * i + input_strides[1] * in_channels_i + input_strides[2] * (j - k_width_i)
                    input_shared[i][j][pk] = input[input_pos]
                else:
                    input_shared[i][j][pk] = 0.0
            else:
                if j + k_width_i < width:
                    input_pos = input_batch_stride * i + input_strides[1] * in_channels_i + input_strides[2] * (j + k_width_i)
                    input_shared[i][j][pk] = input[input_pos]
                else:
                    input_shared[i][j][pk] = 0.0

        if pi + conv_i < k_width * in_channels and k < out_channels and pj == 0:
            k_width_i = (pi + conv_i) // in_channels
            in_channels_i = (pi + conv_i) % in_channels

            # if k < out_channels:
            if reverse:
                k_width_i = k_width - k_width_i - 1

            weight_pos = weight_strides[0] * k + weight_strides[1] * in_channels_i + weight_strides[2] * k_width_i
            weight_shared[pi][k] = weight[weight_pos]

        numba.cuda.syncthreads()

        # The Numa simulator doesn't seem to support syncthreads().
        # When using Numba simulator, comment out syncthreads() and use sleep.
        #
        # time.sleep(1)
        
        if i < batch and k < out_channels:
            for iii in range(BLOCK_DIM):
                if iii + conv_i < k_width * in_channels:
                    total += input_shared[i][j][iii] * weight_shared[iii][k]
        # This is needed because other thread may enter the next loop earlier and change the shared mem.
        numba.cuda.syncthreads()

        # The Numa simulator doesn't seem to support syncthreads().
        # When using Numba simulator, comment out syncthreads() and use sleep.
        #
        # time.sleep(1)

    if i < batch and k < out_channels:
        out_pos = out_strides[0] * i + out_strides[1] * k + out_strides[2] * j
        out[out_pos] = total

# tensor_conv1d_cuda = jit(_tensor_conv1d_cuda)

def tensor_conv1d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    fn = jit(_tensor_conv1d_cuda)
    blockspergrid = (
        (out_shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out_shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out_shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    fn[blockspergrid, threadsperblock](
        out, out_shape, out_strides, out_size, input, input_shape, input_strides, weight, weight_shape, weight_strides, reverse
    )

# class Conv1dCudaFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """Compute a 1D Convolution
#
#         Args:
#         ----
#             ctx : Context
#             input : batch x in_channel x h x w
#             weight : out_channel x in_channel x kh x kw
#
#         Returns:
#         -------
#             batch x out_channel x h x w
#
#         """
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, w = input.shape
#         out_channels, in_channels2, kw = weight.shape
#         assert in_channels == in_channels2
#
#         # Run convolution
#         output = input.zeros((batch, out_channels, w))
#
#         # One block per batch, extra rows, extra col
#         blockspergrid = (
#             (output.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (output.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (output.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#         )
#         threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
#         tensor_conv1d_cuda[blockspergrid, threadsperblock](
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#
#         return output
#
#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, w = input.shape
#         out_channels, in_channels, kw = weight.shape
#         grad_weight = grad_output.zeros((in_channels, out_channels, kw))
#         new_input = input.permute(1, 0, 2)
#         new_grad_output = grad_output.permute(1, 0, 2)
#
#         blockspergrid = (
#             (grad_weight.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (grad_weight.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (grad_weight.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#         )
#         threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
#         tensor_conv1d_cuda[blockspergrid, threadsperblock](
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *new_input.tuple(),
#             *new_grad_output.tuple(),
#             False,
#         )
#
#         grad_weight = grad_weight.permute(1, 0, 2)
#
#         grad_input = input.zeros((batch, in_channels, w))
#         new_weight = weight.permute(1, 0, 2)
#
#
#         blockspergrid = (
#             (grad_input.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (grad_input.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (grad_input.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#         )
#         threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
#         tensor_conv1d_cuda[blockspergrid, threadsperblock](
#             *grad_input.tuple(), grad_input.size, *grad_output.tuple(), *new_weight.tuple(), True
#         )
#
#         return grad_input, grad_weight
