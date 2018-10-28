# BulidSelfTensorflowOp

# Environment
Python2.7 + tensorflow1.8

# Examples
Examples is an attempt to implement https://www.tensorflow.org/extend/adding_an_op, including CPU-Op and GPU-Op.  
The command to create a dynamic library for CPU-Op is as bellow:  

    1. TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )  
    2. TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )  
    3. g++ -std=c++11 -shared `zero_out.cc` -o `zero_out.so` -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 
 
The command for GPU-Op:  

    1. TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )  
    2. TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )  
    3. nvcc -std=c++11 -c -o `cuda_op_kernel.cu.o` `cuda_op_kernel.cu.cc` -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  
    4. g++ -std=c++11 -shared -o `cuda_op_kernel.so` `cuda_op_kernel.cc` `cuda_op_kernel.cu.o` -L /usr/local/cuda-8.0/lib64/ -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB -ltensorflow_framework  
