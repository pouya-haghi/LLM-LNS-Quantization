#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/epilogue/thread/linear_combination_params.h>

namespace cutlass {
namespace epilogue {
namespace thread {

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation.
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute the new epilogue
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_
>
class GaussianEpilogue {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute p1;
    ElementCompute p2;
    ElementCompute p3;

    CUTLASS_HOST_DEVICE
    Params():
      p1(ElementCompute(1)),
      p2(ElementCompute(0)),
      p3(ElementCompute(1)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute p1,
      ElementCompute p2,
      ElementCompute p3
    ):
      p1(p1), p2(p2), p3(p3) { }
  };

private:

  //
  // Data members
  //

  ElementCompute p1_;
  ElementCompute p2_;
  ElementCompute p3_;

public:

  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  GaussianEpilogue(Params const &params) 
    : p1_(params.p1), p2_(params.p2), p3_(params.p3) {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return false; // Assuming the source is not used in this epilogue
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    // Implement if needed for k-partitioning, for now, it's a no-op
  }

  /// Computes the new epilogue function: D = p1 * exp(-0.5 * ((p2 - x) / p3)^2)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform the Gaussian function
    FragmentCompute intermediate;

    // PH: start
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Maximum value of threadIdx.x (its size): %d\n", blockDim.x - 1);
        printf("KCount is: %d\n", kCount);
    }
    // PH: end

    for (int i = 0; i < kCount; ++i) {
      ElementCompute x = converted_accumulator[i];
      ElementCompute diff = (p2_ - x) / p3_;
      // intermediate[i] = p1_ * std::exp(-0.5 * std::pow((p2_ - x) / p3_, 2));
      intermediate[i] = p1_ * __expf(-0.5f * diff * diff);
    }

    return destination_converter(intermediate);
  }

  /// Computes the new epilogue function: D = p1 * exp(-0.5 * ((p2 - x) / p3)^2)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {

    // Convert accumulator to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform the Gaussian function
    FragmentCompute intermediate;

    // PH: start
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Maximum value of threadIdx.x (its size): %d\n", blockDim.x - 1);
        printf("KCount is: %d\n", kCount);
    }
    // PH: end

    for (int i = 0; i < kCount; ++i) {
      ElementCompute x = converted_accumulator[i];
      ElementCompute diff = (p2_ - x) / p3_;
      intermediate[i] = p1_ * __expf(-0.5f * diff * diff);
      // intermediate[i] = p1_ * std::exp(-0.5 * std::pow((p2_ - x) / p3_, 2));
    }

    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
    // Convert everything to Compute type, do compute, and then store to output type
    cutlass::NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    cutlass::NumericConverter<ElementCompute, ElementC, Round> source_converter;
    cutlass::NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type
    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    ElementCompute x = converted_accumulator;
    ElementCompute diff = (p2_ - x) / p3_;
    ElementCompute result = p1_ * __expf(-0.5f * diff * diff);
    // ElementCompute result = p1_ * std::exp(-0.5 * std::pow((p2_ - x) / p3_, 2));
    
    return destination_converter(result);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    // Convert everything to Compute type, do compute, and then store to output type
    cutlass::NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    cutlass::NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type
    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    ElementCompute x = converted_accumulator;
    ElementCompute diff = (p2_ - x) / p3_;
    ElementCompute result = p1_ * __expf(-0.5f * diff * diff);
    // ElementCompute result = p1_ * std::exp(-0.5 * std::pow((p2_ - x) / p3_, 2));
    
    return destination_converter(result);
  }
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
