import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor, SimpleBackend
import numpy as np
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors
from .strategies import assert_close, small_floats

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
CudaTensorBackend = minitorch.TensorBackend(minitorch.CudaOps)


@pytest.mark.task4_1
def test_conv1d_simple_0() -> None:
    t = minitorch.tensor([0, 1, 2, 3], backend=CudaTensorBackend).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]], backend=CudaTensorBackend).view(1, 1, 3)
    out = minitorch.conv1d(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1
#
@pytest.mark.task4_1
def test_conv1d_simple_1() -> None:
    # input: `batch, in_channels, width`
    # weight: `out_channels, in_channels, k_width`
    # output: `batch, out_channels, width`

    t = minitorch.tensor([0, 1, 2, 3], backend=FastTensorBackend).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]], backend=FastTensorBackend).view(1, 1, 3)
    _, t2_shape, t2_strides = t2.tuple()

    t_cuda = minitorch.tensor([0, 1, 2, 3], backend=CudaTensorBackend).view(1, 1, 4)
    t_cuda.requires_grad_(True)
    t2_cuda = minitorch.tensor([[1, 2, 3]], backend=CudaTensorBackend).view(1, 1, 3)

    assert_close_tensor(minitorch.conv1d(t_cuda, t2_cuda), minitorch.conv1d(t, t2))

@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6), backend=FastTensorBackend), tensors(shape=(3, 2, 2), backend=FastTensorBackend))
@settings(max_examples=3)
def test_conv1d_simple_2(input: Tensor, weight: Tensor) -> None:
    out = minitorch.conv1d(input, weight)

    input_cuda = input.contiguous()
    input_cuda.backend = CudaTensorBackend
    weight_cuda = weight.contiguous()
    weight_cuda.backend = CudaTensorBackend
    out_cuda = minitorch.conv1d(input_cuda, weight_cuda)

    assert_close_tensor(out_cuda, out)

    # It fails if not swap the backend. It is probably because both the fast and cuda backend use jit fn and the different
    # jit compilers are incompatible.
    out.backend = SimpleBackend
    out_cuda.backend = SimpleBackend
    assert_close_tensor(out, out_cuda)

@pytest.mark.task4_1
# @given(tensors(shape=(6, 6, 6)), tensors(shape=(4, 6, 4)))
@given(tensors(shape=(6, 6, 6), backend=FastTensorBackend), tensors(shape=(4, 6, 4), backend=FastTensorBackend))
@settings(max_examples=50)
def test_conv1d_simple_3(input: Tensor, weight: Tensor) -> None:
    out = minitorch.Conv1dCudaFun.apply(input, weight)

    input_cuda = input.contiguous()
    input_cuda.backend = CudaTensorBackend
    weight_cuda = weight.contiguous()
    weight_cuda.backend = CudaTensorBackend
    out_cuda = minitorch.conv1d(input_cuda, weight_cuda)

    assert_close_tensor(out_cuda, out)

    # It fails if not swap the backend. It is probably because both the fast and cuda backend use jit fn and the different
    # jit compilers are incompatible.
    out.backend = SimpleBackend
    out_cuda.backend = SimpleBackend
    assert_close_tensor(out, out_cuda)

@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6), backend=FastTensorBackend), tensors(shape=(1, 1, 4), backend=FastTensorBackend))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.conv1d, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6), backend=FastTensorBackend), tensors(shape=(3, 2, 2), backend=FastTensorBackend))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.conv1d, input, weight)

@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6), backend=CudaTensorBackend), tensors(shape=(1, 1, 4), backend=CudaTensorBackend))
def test_conv1d_cuda(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.conv1d, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6), backend=CudaTensorBackend), tensors(shape=(3, 2, 2), backend=CudaTensorBackend))
@settings(max_examples=50)
def test_conv1d_channel_cuda(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.conv1d, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)
