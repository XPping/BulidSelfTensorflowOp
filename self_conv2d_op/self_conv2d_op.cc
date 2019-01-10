

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/kernels/transpose_functor.h"
//#include "tensorflow/core/kernels/conv_ops.h"

namespace tensorflow{
typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Scalar>
struct Conv2dOpKernel{
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using MatrixMap = Eigen::Map<Matrix>;

    static ConstMatrixMap ConstTensorToEigenMatrix(const Tensor& t){
        return ConstMatrixMap(t.flat<Scalar>().data(), t.dim_size(0), t.dim_size(1));
    }

    static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
        return ConstMatrixMap(t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
            t.dim_size(1), t.dim_size(2));
    }

    static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
        return MatrixMap(t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
            t->dim_size(1), t->dim_size(2));
    }

    static void Run(const Tensor& in_x, const Tensor& in_y, Tensor* out, int start, int limit){

        auto y = ConstTensorToEigenMatrix(in_y);
        for(int i=start; i<limit; i++){
            auto x = ConstTensorSliceToEigenMatrix(in_x, i);
            auto z = TensorSliceToEigenMatrix(out, i);
            z.noalias() = y * x;
        }
    }

    static void Run(const Tensor& in_x, const Tensor& in_y, Tensor* out){
        const int batch_size = in_x.dim_size(0);
        auto y = ConstTensorToEigenMatrix(in_y);
        for(int i=0; i<batch_size; i++){
            auto x = ConstTensorSliceToEigenMatrix(in_x, i);
            auto z = TensorSliceToEigenMatrix(out, i);
            z.noalias() = y * x;
        }
    }
}; // struct Conv2dOpKernel

template <typename Device, typename Scalar>
struct LaunchConv2dOp;

template <typename Scalar>
struct LaunchConv2dOp<CPUDevice, Scalar>{
    static void Launch(OpKernelContext* context, const Tensor& in_x,
                        const Tensor& kernel, Tensor* out){
        OP_REQUIRES(context, in_x.dim_size(1) == kernel.dim_size(1),
                errors::InvalidArgument("input and filter must have the same depth: ",
                                in_x.dim_size(1), " vs ", kernel.dim_size(1)));
        Conv2dOpKernel<Scalar>::Run(in_x, kernel, out);
        /*
        const int64 num_units = in_x.dim_size(0);
        const int64 cost_per_unit =
            kernel.dim_size(0) * kernel.dim_size(1) * in_x.dim_size(2);
        auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
        if(worker_threads.num_threads > num_units){
            Shard(std::max(1, worker_threads.num_threads-1),
                  worker_threads.workers,
                  num_units, cost_per_unit,
                  [context, &in_x, &kernel, out](int start, int limit){
                    Conv2dOpKernel<Scalar>::Run(in_x, kernel, out, start, limit);});
        }else{
            Shard(worker_threads.num_threads,
                  worker_threads.workers,
                  num_units, cost_per_unit,
                  [context, &in_x, &kernel, out](int start, int limit){
                    Conv2dOpKernel<Scalar>::Run(in_x, kernel, out, start, limit);});
        }*/
    }
}; // struct LaunchConv2dOp

template <typename Device, typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Input
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& t_stride = context->input(2);
    auto _t_stride = t_stride.flat<T>();
    const int64 stride = _t_stride(0);
    // Input dim
    const int64 batch_size = input.dim_size(0);
    const int64 in_depth = input.dim_size(1);
    const int64 in_height = input.dim_size(2);
    const int64 in_width = input.dim_size(3);
    const int64 out_depth = filter.dim_size(0);
    OP_REQUIRES(context, in_depth == filter.dim_size(1),
                errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));
    // Kernel size
    const int64 k_height = filter.dim_size(2);
    const int64 k_width = filter.dim_size(3);
    // Stride
    //const int64 stride = 1; //context->input(2);
    // Output height and width
    const int64 out_width = (in_width - k_width + 1) / stride;
    const int64 out_height = (in_height - k_height + 1) / stride;
    // img2col implement, raw major
    // img2col shape
    TensorShape input2col_shape;
    input2col_shape.AddDim(batch_size);
    input2col_shape.AddDim(out_height * out_width);
    input2col_shape.AddDim(in_depth * k_height * k_width);
    // img2col data,
    Tensor input2col;
    OP_REQUIRES_OK(context, context->allocate_temp(tensorflow::DataTypeToEnum<T>::value,
                            TensorShape({batch_size, in_depth*k_height*k_width, out_height*out_width}),
                            &input2col));
    //OP_REQUIRES_OK(context, context->allocate_output(1,
    //                        TensorShape({batch_size, in_depth*k_height*k_width, out_height*out_width}), &input2col));
    auto _input = input.flat<T>();
    auto _input2col = input2col.flat<T>();
    for(int i=0; i<batch_size; i++){
        for(int j=0; j<in_depth; j++){
            for(int h=0; h<in_height-k_height+1; h+=stride){
                for(int w=0; w<in_width-k_width+1; w+=stride){
                    for(int k=0; k<k_height; k++){
                        for(int kk=0; kk<k_width; kk++){
                            const int64 index1 = i*in_depth*k_height*k_width*out_height*out_width
                                           + j*k_height*k_width*out_height*out_width
                                           + (k*k_width+kk)*(out_width*out_height) + h*(in_width-k_width+1)/stride+w;
                            const int64 index2 = i*in_depth*in_height*in_width + j*in_height*in_width + (h+k)*in_width + (w+kk);
                            OP_REQUIRES(context, index1 < batch_size*in_depth*k_height*k_width*out_height*out_width,
                                errors::InvalidArgument("index1: ",
                                    index1, " vs ", batch_size*in_depth*k_height*k_width*out_height*out_width));
                            OP_REQUIRES(context, index2 < batch_size*in_depth*in_height*in_width,
                                errors::InvalidArgument("index2: ",
                                    index2, " vs ", batch_size*in_depth*in_height*in_width));
                            _input2col(index1) = _input(index2);
                            //memcpy(&_input2col(index1), _input(index2), 1 * sizeof(T));
                        }
                    }
                }
            }
        }
    }
    //context->set_output(0, *input2col);
    //Tensor input2col_reshaped;
    //CHECK(input2col_reshaped.CopyFrom(*input2col, TensorShape({batch_size, in_depth*k_height*k_width, out_height*out_width})));

    // kernel2col implement, kernel2col'shape is (out_depth, in_depth*k_height*k_width)
    Tensor kernel2col(filter.dtype());
    CHECK(kernel2col.CopyFrom(filter, TensorShape({out_depth, in_depth*k_height*k_width})));
    //context->set_output(0, kernel2col);
    // Output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, out_depth, out_height, out_width}), &output));
    Tensor output_reshaped;
    CHECK(output_reshaped.CopyFrom(*output, TensorShape({batch_size, out_depth, out_height*out_width})));
    // execute conv2d
    LaunchConv2dOp<Device, T>::Launch(context, input2col, kernel2col, &output_reshaped);
    /*
    // Output reshape
    Tensor output_reshaped;
    CHECK(output_reshaped.CopyFrom(*output, TensorShape({batch_size, out_depth, out_height, out_width})));
    // context->set_output(0, result);
    */
  }
};

REGISTER_OP("ZeroOut")
    .Input("input: T")
    .Input("filter: T")
    .Input("stride: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(Computes a 2-D convolution given 4-D `input` and `filter` tensors.)doc");

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZeroOutOp<CPUDevice, T>);

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.

//TF_CALL_half(REGISTER_CPU);
//TF_CALL_float(REGISTER_CPU);
REGISTER_CPU(float);
REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_KERNEL_BUILDER(Name("SelfConv2d").Device(DEVICE_CPU), SelfConv2dOp);
//REGISTER_KERNEL_BUILDER(
//    Name("SelfConv2dOp").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
//    SelfConv2dOp<CPUDevice, Eigen::half>);
//REGISTER_KERNEL_BUILDER(
//    Name("SelfConv2dOp").Device(DEVICE_CPU).TypeConstraint<float>("T"),
//    SelfConv2dOp<CPUDevice, float>);

//REGISTER_KERNEL_BUILDER(Name("SelfConv2dOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), SelfConv2dOp<CPUDevice, T>);

} // namespace tensorflow