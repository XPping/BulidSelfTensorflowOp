from __future__ import division, print_function


import numpy as np

"""
reference: https://blog.csdn.net/Mrhiuser/article/details/52672824
"""

def im2col(data, kernel_size, stride):
    """
    :param data: shape:(batch_size, in_channels, height, width)
    :param kernel: (k_height, k_width)
    :stride:
    :return:
    """
    batch_size, channels, height, width = data.shape[0:4]
    k_height, k_width = kernel_size
    o_width = (width - k_width + 1) // stride
    o_height = (height - k_height + 1) // stride

    data = data.reshape(-1)
    output = np.zeros(batch_size * channels*k_height*k_width * o_height*o_width)
    for i in range(batch_size):
        for j in range(channels):
            for h in range(0, height-k_height+1, stride):
                for w in range(0, width-k_width+1, stride):
                    for k in range(k_height):
                        for kk in range(k_width):
                            output_index = i*channels*k_height*k_width*o_height*o_width \
                                           + j*k_height*k_width*o_height*o_width \
                                           + (k*k_width+kk)*(o_width*o_height) + h*int((width-k_width+1)/stride)+w
                            data_index = i*channels*height*width + j*height*width + (h+k)*width + (w+kk)
                            # print(output_index, data_index)
                            output[output_index] = data[data_index]
    """
    output = np.zeros((batch_size, o_height*o_width, channels*k_height*k_width))
    for i in range(batch_size):
        for j in range(channels):
            for h in range(0, height-k_height+1, stride):
                for w in range(0, width-k_width+1, stride):
                    for k in range(k_height):
                        tmp = j*k_height*k_width+k*k_width
                        output[i, h//stride*o_width+w, tmp:tmp+k_width] \
                            = data[i, j, h+k, w*k_width:(w+1)*k_width]
    result = np.zeros((batch_size, channels*k_height*k_width, o_height*o_width))

    for i in range(batch_size):
        result[i] = np.transpose(output[i])
    """
    output = output.reshape((batch_size, channels*k_height*k_width, o_height*o_width))
    return output

def kernel2col(kernel):
    """
    :param kernel: shape:(out_channels, in_channels, k_width, k_height)
    :return:
    """
    out_channels, in_channels, k_width, k_height = kernel.shape[0:4]
    output = np.zeros((out_channels, in_channels * k_width * k_height), dtype=kernel.dtype)

    for out_c in range(out_channels):
        tmp = kernel[out_c, :, :, :].reshape(-1)
        output[out_c] = tmp

    return output

def conv2d(data, kernel, stride=1):
    """
    :param data: shape:(batch_size, in_channels, height, width)
    :param kernel: shape:(out_channels, in_channels, k_height, k_width)
    :param stride:
    :return:
    """
    if len(data.shape) != 4 or len(kernel.shape) != 4:
        raise "the shape of data, kernel is error!"
    batch_size, in_channels, height, width = data.shape[0:4]
    out_channels, in_channels, k_height, k_width = kernel.shape[0:4]
    o_width = (width - k_width + 1) // stride
    o_height = (height - k_height + 1) // stride

    output = np.zeros((batch_size, out_channels, o_height, o_width))

    data = im2col(data, (k_width, k_height), stride)
    kernel = kernel2col(kernel)
    for i in range(batch_size):
        output[i] = np.dot(kernel, data[i]).reshape(-1, o_height, o_width)
    return output

if __name__ == "__main__":
    data = np.zeros((2, 3, 4, 4))
    for i in range(3):
        data[:, i, :, :] = np.arange(16).reshape(4, 4)
    kernel = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            kernel[i, j, :, :] = (np.ones(9)*(i+1)).reshape(3, 3)
    output = conv2d(data, kernel, stride=1)
    print(output)