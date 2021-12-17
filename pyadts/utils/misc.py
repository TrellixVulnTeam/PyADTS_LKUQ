import math


def conv_output_size(input_size: int, kernel_size: int, padding: int = 0, stride: int = 1, dilation: int = 1):
    """
    A helper function that computes the output size of an 1d convolutional operation.

    Args:
        input_size: Input size
        kernel_size: Size of the convolving kernel
        padding: Padding added to both sides of the input. Default: 0
        stride: Stride of the convolution. Default: 1
        dilation: Spacing between kernel elements. Default: 1

    Returns:

    """
    return math.floor((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride + 1)


def conv_transpose_output_size(input_size: int, kernel_size: int, padding: int = 0, stride: int = 1, dilation: int = 1,
                               output_padding: int = 0):
    """
    A helper function that computes the output size of an 1d transposed convolutional operation.

    Args:
        input_size: Input size
        kernel_size: Size of the convolving kernel
        padding: Padding added to both sides of the input. Default: 0
        stride: Stride of the convolution. Default: 1
        dilation: Spacing between kernel elements. Default: 1
        output_padding: The value controls the additional size added to one side of the output shape.

    Returns:

    """
    return (input_size - 1) * stride - padding * 2 + dilation * (kernel_size - 1) + output_padding + 1
