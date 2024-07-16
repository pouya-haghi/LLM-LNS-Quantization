#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/gaussian.h"


#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::GaussianEpilogue<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
>;

// Rest of your GEMM configuration remains the same
template <typename MathOperator>
using Gemm_ = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    kAlignmentA, kAlignmentB, MathOperator
>;


struct Options {

  bool help;
  bool error;
  bool reference_check;
  cutlass::gemm::GemmCoord problem_size;

  int iterations;
  int warmup_iterations;

  bool scale_A;
  bool scale_B;
  bool scale_C;

  float alpha;
  float beta;

  Options():
    help(false),
    error(false),
    reference_check(false),
    iterations(20),
    warmup_iterations(5),
    scale_A(true),
    scale_B(true),
    scale_C(true),
    alpha(1.f),
    beta(0.f)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
    cmd.get_cmd_line_argument("reference-check", reference_check, false);
    cmd.get_cmd_line_argument("scale-A", scale_A, true);
    cmd.get_cmd_line_argument("scale-B", scale_B, true);
    cmd.get_cmd_line_argument("scale-C", scale_C, true);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);

    int m, n, k;
    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("k", k, 1024);

    problem_size = cutlass::gemm::GemmCoord{m, n, k};
  }

  std::ostream & print_usage(std::ostream &out) const {

    out << "58_ada_fp8_gemm\n\n"
      << "  This example executes a GEMM using Ada FP8 Tensor Core operations. In addition to performing\n"
      << "  a normal GEMM, the kernel performs the following operations:\n"
      << "      Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias\n"
      << "        D = activation(Aux)\n\n"
      << "      if Aux is fp8:\n"
      << "         abs_max_output = max( abs(aux) | (for every aux in Aux) )\n"
      << "         Aux = scale_aux * Aux\n\n"
      << "      if D is fp8 type:\n"
      << "         abs_max_output = max( abs(d) | (for every d in D) )\n"
      << "         D = scale_d * D\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement\n\n"
      << "  --m=<int>                        Sets the M dimension of the GEMM\n"
      << "  --n=<int>                        Sets the N dimension of the GEMM\n"
      << "  --k=<int>                        Sets the K dimension of the GEMM\n"
      << "  --scale-A=<bool>                 Whether to apply a scaling factor to operand A (default: true)\n"
      << "  --scale-B=<bool>                 Whether to apply a scaling factor to operand B (default: true)\n"
      << "  --scale-C=<bool>                 Whether to apply a scaling factor to operand C (default: true)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform\n"
      << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
      << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  float gflops(float runtime_s) const {
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};

template <typename Gemm>
struct TestbedRunner {

  using ElementAccumulator = typename Gemm::ElementAccumulator;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Vector;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;

  TestbedRunner(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else {
      std::cerr << "Not implemented";
      return false;
    }

    return true;
  }

  void initialize(const Options& options) {
    tensor_A.resize(options.problem_size.mk());
    tensor_B.resize(options.problem_size.kn());
    tensor_C.resize(options.problem_size.mn());
    tensor_D.resize(options.problem_size.mn());
    tensor_Vector.resize({1, options.problem_size.n()});
    reference_D.resize(options.problem_size.mn(), false);
    tmp_D.resize(options.problem_size.mn(), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);
    initialize_tensor(tensor_Vector.host_view(), init_C, seed + 2020);

    cutlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);
    tensor_Vector.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorFill(tensor_D.host_view());
    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_Vector.sync_device();
  }

  bool compare_reference(const Options& options) {

    tensor_D.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;

      std::string output_file = "testbed_errors.txt";
      std::ofstream file(output_file);

      file
        << "problem: " << options.problem_size
        << ", alpha: " << options.alpha << ", beta: " << options.beta << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nVector =\n" << tensor_Vector.host_view()
        << "\n\nReference D =\n" << reference_D.host_view()
        << "\nComputed D =\n" << tensor_D.host_view();

      std::cerr << "Dumped results to " << output_file << std::endl;

    }

    return passed;
  }

  bool verify(const Options& options) {

    cutlass::Coord<2> origin(0);
    float scaled_alpha = options.alpha;
    if (options.scale_A) {
      scaled_alpha *= 1.0f;
    }
    if (options.scale_B) {
      scaled_alpha *= 1.0f;
    }

    float scaled_beta = options.beta;
    if (options.scale_C) {
      scaled_beta *= 1.0f;
    }

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        float, ElementAccumulator, ElementAccumulator
    >(
      options.problem_size,
      scaled_alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      scaled_beta,
      tensor_C.host_ref(),
      tmp_D.host_ref(),
      ElementAccumulator(0)
    );

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tmp_D.host_view());

    return compare_reference(options);
  }

  bool sufficient() const {

    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)) {
      std::cerr << "This example requires CUDA 12.4 or greater." << std::endl;
      return false;
    }

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDevice() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    if (properties.major < 8 || (properties.major == 8 && properties.minor < 9)) {
      std::cerr << "CUTLASS's Ada FP8 GEMM example requires a device of compute capability 89 or higher.\n" << std::endl;
      return false;
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      std::cerr << "Insufficient shared memory. Need " << smem_size
                << ", but device only has " << properties.sharedMemPerBlockOptin << std::endl;
      return false;
    }

    return true;
  }

  bool run(Options& options) {

    if (!sufficient()) {
      std::cerr << "Insufficient resources to run the kernel." << std::endl;
      return false;
    }

    this->initialize(options);

  //     // Initialize alpha/beta for dot product computation
  // float alpha = float(1);
  // float beta = float(1);
  //   // Split K dimension into 1 partitions
  // int split_k_slices = 1;

  //   typename Gemm::Arguments arguments{
  //     cutlass::gemm::GemmUniversalMode::kGemm,
  //     options.problem_size,
  //     /* batch_count = */ 1,
  //     // {options.alpha, options.beta},
  //     {alpha, beta},    
  //     tensor_A.device_data(),
  //     tensor_B.device_data(),
  //     tensor_C.device_data(),
  //     tensor_D.device_data(),
  //     options.problem_size.m() * options.problem_size.k(),
  //     options.problem_size.n() * options.problem_size.k(),
  //     options.problem_size.m() * options.problem_size.n(),
  //     options.problem_size.m() * options.problem_size.n(),
  //     tensor_A.layout().stride(0),
  //     tensor_B.layout().stride(0),
  //     tensor_C.layout().stride(0),
  //     tensor_D.layout().stride(0),
  //     nullptr, 
  //     nullptr,
  //     nullptr
  //   };

  // Define the epilogue parameters
float p1 = 1.0f;
float p2 = 0.0f;
float p3 = 1.0f;

// Initialize Gaussian epilogue parameters
typename Gemm::EpilogueOutputOp::Params epilogue_params(p1, p2, p3);

// Define the GEMM arguments
typename Gemm::Arguments arguments{
  cutlass::gemm::GemmUniversalMode::kGemm,
  options.problem_size,
  /* batch_count = */ 1,
  epilogue_params,    
  tensor_A.device_data(),
  tensor_B.device_data(),
  tensor_C.device_data(),
  tensor_D.device_data(),
  options.problem_size.m() * options.problem_size.k(),
  options.problem_size.n() * options.problem_size.k(),
  options.problem_size.m() * options.problem_size.n(),
  options.problem_size.m() * options.problem_size.n(),
  tensor_A.layout().stride(0),
  tensor_B.layout().stride(0),
  tensor_C.layout().stride(0),
  tensor_D.layout().stride(0),
  nullptr, 
  nullptr,
  nullptr
};


    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }

    bool passed = true;
    if (options.reference_check) {
      passed &= this->verify(options);
    } else {
      std::cout << "Skipped reference check" << std::endl;
    }

    for (int i = 0; i < options.warmup_iterations; ++i) {
      gemm_op();
    }

    cudaEvent_t events[2];
    cudaError_t error;
    for (auto & event : events) {
      error = cudaEventCreate(&event);
      if (error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(error) << std::endl;
        return false;
      }
    }

    error = cudaEventRecord(events[0]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_op();
    }

    error = cudaEventRecord(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    error = cudaEventSynchronize(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    float runtime_ms = 0;
    error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    runtime_ms = runtime_ms / float(options.iterations);
    float gflops = options.gflops(runtime_ms / 1000.0f);

    std::cout << "Problem size: " << options.problem_size.m() << 'x' << options.problem_size.n() << 'x' << options.problem_size.k() << std::endl;
    std::cout << "Runtime (ms): " << runtime_ms << std::endl;
    std::cout << "GFLOPs/sec:   " << gflops << std::endl;

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    return passed;
  }

};

int main(int argc, char const** argv) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4) ||
      (props.major != 8 && props.minor != 9)) {

    std::cout
      << "CUTLASS's FP8 SM89 example requires a GPU of NVIDIA's Ada architecture "
      << "and CUDA toolkit version 12.4 or later.\n";

    return 0;
  }

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  std::cout << "Running GEMM with staged accumulation (OpMultiplyAdd)" << std::endl;
  std::cout << "=====================================================" << std::endl;
  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAdd>> testbed_staged_accum;
  bool passed = testbed_staged_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  std::cout << "\nRunning GEMM with fast accumulation (OpMultiplyAddFastAccum)" << std::endl;
  std::cout << "============================================================" << std::endl;
  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAddFastAccum>> testbed_fast_accum;
  passed = testbed_fast_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  return 0;
}
