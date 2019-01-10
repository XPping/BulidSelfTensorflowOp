# caffe_conv2d.py
This is caffe-conv2d implements by python-numpy, which is used for test  

# self_conv2d_op.cc
This is conv2d-op by tensorflow-C++

# BUILD
This is the bazel file

# self_conv2d.so
This is the DLL of self_conv2d_op.cc. First, you must bazel the source-code of tensorflow. Then, put the self_conv2d_op.cc and BUILD into the "tensorflow/core/user_ops". Finally, run "bazel build --config opt //tensorflow/core/user_ops:self_conv2d.so" in home dir of tensorflow.

# test_self_conv2d_op.py
This is used for self-conv2d test.

# Result
![image](https://github.com/XPping/BulidSelfTensorflowOp/raw/master/self_conv2d_op/images/python_self_conv2d.jpg) 
![image](https://github.com/XPping/BulidSelfTensorflowOp/raw/master/self_conv2d_op/images/self_conv2d.jpg) 
