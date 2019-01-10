import tensorflow as tf
import numpy as np
def testZeroOut():
    zero_out_module = tf.load_op_library("./self_conv2d.so")
    data = np.zeros((2, 3, 4, 4))
    for i in range(3):
        data[:, i, :, :] = np.arange(16).reshape(4, 4)
    kernel = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            kernel[i, j, :, :] = (np.ones(9) * (i + 1)).reshape(3, 3)

    sess = tf.Session()
    with sess.as_default():
        result = zero_out_module.zero_out(data, kernel, 1)
        print(result)
        print(result.eval())
if __name__ == "__main__":
    testZeroOut()