# type: ignore
# Currently pyright doesn't support numba.cuda

import time
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
from .tensor_functions import Function

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
                    input_shared[pi][pj][pk] = input[input_pos]
                else:
                    input_shared[pi][pj][pk] = 0.0
            else:
                if j + k_width_i < width:
                    input_pos = input_batch_stride * i + input_strides[1] * in_channels_i + input_strides[2] * (j + k_width_i)
                    input_shared[pi][pj][pk] = input[input_pos]
                else:
                    input_shared[pi][pj][pk] = 0.0

        if pi + conv_i < k_width * in_channels and k < out_channels and pj == 0:
            k_width_i = (pi + conv_i) // in_channels
            in_channels_i = (pi + conv_i) % in_channels

            # if k < out_channels:
            if reverse:
                k_width_i = k_width - k_width_i - 1

            weight_pos = weight_strides[0] * k + weight_strides[1] * in_channels_i + weight_strides[2] * k_width_i
            weight_shared[pi][pk] = weight[weight_pos]

        numba.cuda.syncthreads()

        # The Numa simulator doesn't seem to support syncthreads().
        # When using Numba simulator, comment out syncthreads() and use sleep.
        #
        # time.sleep(1)

        if i < batch and k < out_channels:
            for iii in range(BLOCK_DIM):
                if iii + conv_i < k_width * in_channels:
                    total += input_shared[pi][pj][iii] * weight_shared[iii][pk]
        # This is needed because other thread may enter the next loop earlier and change the shared mem.
        numba.cuda.syncthreads()

        # The Numa simulator doesn't seem to support syncthreads().
        # When using Numba simulator, comment out syncthreads() and use sleep.
        #
        # time.sleep(1)

    if i < batch and k < out_channels:
        out_pos = out_strides[0] * i + out_strides[1] * k + out_strides[2] * j
        out[out_pos] = total






# Copied https://github.com/Cornell-Tech-ML/mle-module-4-93c3173d-HarshiniDonepudi/blob/master/cuda_conv.py

# def tensor_conv1d(
#         out: Tensor,
#         out_shape: Shape,
#         out_strides: Strides,
#         out_size: int,
#         input: Tensor,
#         input_shape: Shape,
#         input_strides: Strides,
#         weight: Tensor,
#         weight_shape: Shape,
#         weight_strides: Strides,
#         reverse: bool,
# ) -> None:
#     """
#     1D Cuda Convolution implementation.
#     Given input tensor of
#        `batch, in_channels, width`
#     and weight tensor
#        `out_channels, in_channels, k_width`
#     Computes padded output of
#        `batch, out_channels, width`
#     `reverse` decides if weight is anchored left (False) or right.
#     (See diagrams)
#     Args:
#         out (array): storage for `out` tensor.
#         out_shape (array): shape for `out` tensor.
#         out_strides (array): strides for `out` tensor.
#         out_size (int): size of the `out` tensor.
#         input (array): storage for `input` tensor.
#         input_shape (array): shape for `input` tensor.
#         input_strides (array): strides for `input` tensor.
#         weight (array): storage for `input` tensor.
#         weight_shape (array): shape for `input` tensor.
#         weight_strides (array): strides for `input` tensor.
#         reverse (bool): anchor weight at left or right
#     """
#     batch_, out_channels, out_width = out_shape
#     batch, in_channels, width = input_shape
#     out_channels_, in_channels_, kw = weight_shape
#
#     assert (
#             batch == batch_
#             and in_channels == in_channels_
#             and out_channels == out_channels_
#     )
#     s1 = weight_strides
#     s2 = input_strides
#
#     # Block arrangement is going to be size of output: (size_out, )
#     pos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     i = cuda.local.array(MAX_DIMS, numba.int32)
#     to_index(pos, out_shape, i)
#
#     # Im going to have kw * in_channels threads per block: [in_channels, kw]
#     # Initialize the shared memories
#     Shared_Input = cuda.local.array((in_channels_, kw), numba.int32)
#     Shared_Weights = cuda.local.array((in_channels_, kw), numba.int32)
#     # And get the local indexes of the threads
#     local_i = cuda.threadIdx.x
#     local_j = cuda.threadIdx.y
#
#     # Time to fill the memories. The weight one is straightforward
#     Shared_Weights[local_i, local_j] = weight[
#         i[1] * s1[0] + local_i * s1[1] + local_j * s1[2]
#         ]
#     # The input memory is not as straightforward
#     if reverse is False:
#         Pos = i[2] + local_j
#         if Pos < width:
#             Shared_Input[local_i, local_j] = input[
#                 i[0] * s2[0] + local_i * s2[1] + Pos * s2[2]
#                 ]
#         else:
#             Shared_Input[local_i, local_j] = 0
#         # Once the shared memories are initialized we just compute the sum and accumulate
#         Res = 0.0
#         # Wait for all threads to reach this point
#         numba.cuda.syncthreads()
#         Res += Shared_Input[local_i, local_j] * Shared_Weights[local_i, local_j]
#     else:
#         Pos = id[2] - local_i
#         if Pos >= 0:
#             Shared_Input[local_i, local_j] = input[
#                 id[0] * s2[0] + local_i * s2[1] + Pos * s2[2]
#                 ]
#         else:
#             Shared_Input[local_i, local_j] = 0
#         Res = 0.0
#         # Wait for all threads to reach this point
#         numba.cuda.syncthreads()
#         Res += Shared_Input[local_i, local_j] * Shared_Weights[local_i, local_j]
#
#
# class Conv1dFun(Function):
#     """
#     Compute a 1D Convolution.
#     Args:
#         ctx: Context.
#         input (:class:'Tensor'): batch x in_channel x h x w.
#         weight (:class:'Tensor'): out_channel x in_channel x kh x kw.
#     Returns:
#         (:class:'Tensor'): batch x out_channel x h x w.
#     """
#
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, w = input.shape
#         out_channels, in_channels2, kw = weight.shape
#         assert in_channels == in_channels2
#
#         # Run convolution
#         output = input.zeros((batch, out_channels, w))
#         tensor_conv1d(
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#         return output
#
#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, w = input.shape
#         out_channels, in_channels2, kw = weight.shape
#         assert in_channels == in_channels2
#
#         grad_weight = grad_output.zeros((out_channels, in_channels, kw))
#         new_input = input.permute(1, 0, 2)
#         new_grad_output = grad_output.permute(1, 0, 2)
#         tensor_conv1d(
#             grad_weight, new_grad_output, True
#         grad_weight.shape,
#         grad_weight.strides,
#         grad_weight.size,
#         new_input,
#         new_input.shape,
#         new_input.strides,
#         new_grad_output,
#         new_grad_output.shape,
#         new_grad_output.strides,
#         False,
#         )
#         grad_weight = grad_weight.permute(1, 0, 2)
#         grad_input = input.zeros((batch, in_channels, w))
#         new_weight = weight.permute(1, 0, 2)
#         tensor_conv1d(
#             grad_input,
#             grad_input.shape,
#             grad_input.strides,
#             grad_input.size,
#             grad_output,
#             grad_output.shape,
#             grad_output.strides,
#             new_weight,
#             new_weight.shape,
#             new_weight.strides,
#             True,
#         )
#         return grad_input, grad_weight
#
#
# conv1d = cuda.jit()(Conv1dFun)



_tensor_conv1d_cuda_jit = jit(_tensor_conv1d_cuda)

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
    blockspergrid = (
        (out_shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out_shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out_shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    _tensor_conv1d_cuda_jit[blockspergrid, threadsperblock](
        out, out_shape, out_strides, out_size, input, input_shape, input_strides, weight, weight_shape, weight_strides, reverse
    )



#######################################################################################################################
#######################################################################################################################
######################################################################################################################
#
# This section is an old interface for conv1d. It is used for test purpose only.
#
tensor_conv1d_cuda_old = jit(_tensor_conv1d_cuda)

class Conv1dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (output.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (output.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (output.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        tensor_conv1d_cuda_old[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)

        blockspergrid = (
            (grad_weight.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_weight.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_weight.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        tensor_conv1d_cuda_old[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )

        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)


        blockspergrid = (
            (grad_input.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_input.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_input.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        tensor_conv1d_cuda_old[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size, *grad_output.tuple(), *new_weight.tuple(), True
        )

        return grad_input, grad_weight
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
