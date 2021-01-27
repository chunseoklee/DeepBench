#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#include "gemmlowp/public/gemmlowp.h"
#include "gemmlowp/test/test.h"
#include "gemm_problems.h"

inline int32_t AccumulateNeonLane(const int32x4_t lane) {
#ifdef __aarch64__
  return vaddvq_s32(lane);
#else
  int64x2_t pairwiseAdded = vpaddlq_s32(lane);
  return vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);
#endif
}

inline void* aligned_alloc(size_t alignment, size_t size,
                           void** freeing_buffer) {
#ifdef TFLITE_USE_STD_ALIGNED_ALLOC
  *freeing_buffer = ::aligned_alloc(
      alignment, (size + alignment - 1) / alignment * alignment);
  return *freeing_buffer;
#else
  *freeing_buffer = malloc(size + alignment);
  const size_t offset = ((uintptr_t)*freeing_buffer) % alignment;  // NOLINT
  return offset == 0
             ? *freeing_buffer
             : ((char*)*freeing_buffer + (alignment - offset));  // NOLINT
#endif
}

void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                             const int m_rows, const int m_cols,
                                             const int8_t* __restrict__ vectors,
                                             const float* scaling_factors,
                                             int n_batch,
                                             float* __restrict__ result) {

  static const int kWeightsPerUint32 = 4;
  static const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
  bool unaligned = false;
  int8_t* aligned_row = nullptr;
  void* aligned_row_free = nullptr;
  if ((m_cols & (kWeightsPerUint32 - 1)) != 0) {
    unaligned = true;
    aligned_row = (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                                         &aligned_row_free);
  }
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                             &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_half_start
  // shows the start index where this should happen. Between postamble_start and
  // postamble_half_start we can still process kWeightsPerNeonLane >> 1 in a
  // vectorized form.
  const int postamble_half_start = m_cols & ~(kWeightsPerNeonLane - 1);
  const int postamble_start = m_cols & ~((kWeightsPerNeonLane >> 1) - 1);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols);
    // Compute dot-product for every column.
    for (int row = 0; row < m_rows; ++row) {
      // Get the address of the first element of the row.
      int8_t* row_ptr = (int8_t*)matrix + row * m_cols;  // NOLINT
      if (unaligned) {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod_32x4 = vmovq_n_s32(0);

      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);

      // For every block of 16 8-bit elements.
      int col = 0;
      for (; col < postamble_half_start; col += kWeightsPerNeonLane) {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned. Otherwise,
        // performance may suffer significantly.
        // TFLITE_DCHECK_EQ(  // NOLINT
        //     (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t*)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 =
            vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the higher 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 =
            vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
      }  // for col

      // Half iteration dealing only 8 elements
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < postamble_start))
      if (col < postamble_start) {
        // Load 8 8-bit values from the row and column each to operate on.
        // Here the assumption is that each buffer is 4-bytes aligned.
        // Otherwise, performance may suffer significantly.
        // TFLITE_DCHECK_EQ(  // NOLINT
        //     (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x8_t s1_8x8 = vld1_s8((const int8_t*)(aligned_vec + col));
        const int8x8_t s2_8x8 = vld1_s8((const int8_t*)(row_ptr + col));
        const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
        col += (kWeightsPerNeonLane >> 1);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int32_t dotprod = AccumulateNeonLane(dotprod_32x4);
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < m_cols))
      for (; col < m_cols; ++col) {
        dotprod += row_ptr[col] * aligned_vec[col];
      }  // for col

      *result += dotprod * batch_scaling_factor;
      ++result;
    }  // for row
  }    // for batch

  if (unaligned) {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

// tflite internal kernel assume that lhd:row-major, rhs:col-major, result: col-major
double time_tflite(int m, int n, int k) {

    int8_t* lhs = (int8_t*)malloc(m*k*sizeof(int8_t));
    int8_t* rhs = (int8_t*)malloc(k*n*sizeof(int8_t));
    float* result = (float*)malloc(m*n*sizeof(float));
    int batch = n;
    float* scale = (float*)malloc(n*sizeof(float));

    // warm up
    NeonMatrixBatchVectorMultiplyAccumulate(lhs, m, k, rhs, scale, batch, result);

    int numRepeats = std::max(std::ceil(1e10 / (m * k * n)), 10.);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
      NeonMatrixBatchVectorMultiplyAccumulate(lhs, m, k, rhs, scale, batch, result);
    }

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / numRepeats;
}



template <bool a_t, bool b_t>
double time_gemm(int m, int n, int k) {
    gemmlowp::GemmContext context;
    context.set_max_num_threads(0);
    
    typedef gemmlowp::MapOrder Order;

    static const Order LhsOrder = a_t ? Order::RowMajor : Order::ColMajor;
    static const Order RhsOrder = b_t ? Order::RowMajor : Order::ColMajor;

    gemmlowp::Matrix<std::uint8_t, LhsOrder> lhs(m, k);
    gemmlowp::Matrix<std::uint8_t, RhsOrder> rhs(k, n);
    gemmlowp::Matrix<std::uint8_t, Order::ColMajor> result(m, n);

    gemmlowp::MakeRandom<typename gemmlowp::OperandRange<0, 255>>(&lhs);
    gemmlowp::MakeRandom<typename gemmlowp::OperandRange<0, 255>>(&rhs);

/** Configuration for low precision shifted matrix multiplication.
 *
 *  The mathematical expression to be computed is the result of the following steps:
 *  1. Cast lhs entries from uint8 to int32 and add lhs_offset to each of them.
 *  2. Cast rhs entries from uint8 to int32 and add rhs_offset to each of them.
 *  3. Compute the int32 matrix product of the resulting lhs times rhs.
 *  4. Add res_offset to each entry of the result.
 *  5. Multiply each entry of the result by the following fraction, and round
 *     to the nearest integer:
 *
 *                         res_mul
 *                       -----------
 *                       2^res_shift
 *
 *  6. Clamp the resulting int32 values to the [0..255] range and cast to uint8.
 *
 *  To summarize:
 *
 *        res_mul
 *  B = ----------- ((A + lhs_offset) * (X + rhs_offset) + res_offset)
 *      2^res_shift
 *
 *  By estimating or observing the range of values of the entries in A, X, and
 *  B matrices, you can determine min_a, max_a, min_x, max_x, min_b, max_b,
 *  which are the minimum/maximum representable float value for your uint8_t
 *  representation for entries in A, X, and B, respectively.
 *
 *  Then the parameters are determined as follows:
 *
 *                  min_a * 256
 *  lhs_offset = ------------------
 *                  max_a - min_a
 *
 *                  min_x * 256
 *  rhs_offset = ------------------
 *                  max_x - min_x
 *
 *                     - min_b * 256 * 256
 *  res_offset = -----------------------------------
 *                (max_a - min_a) * (max_x - min_x)
 *
 *    res_mul     (max_a - min_a) * (max_x - min_x)
 *  ----------- = ---------------------------------
 *  2^res_shift        (max_b - min_b) * 256
 *
 *  The parameters used below correpsonds to:
 *
 *   min_a = -1,  max_a = 1
 *   min_x = 0,   max_x = 16
 *   min_b = -16, max_b = 16
 *
 *  which are tuned to work for our GEMM application.
 *  You should determine your own maximum/minimum that work for your case.
 */

    int lhs_offset = -128;
    int rhs_offset = 0;
    int res_offset = 32768;
    int res_mul = 1;
    int res_shift = 8;

    // warm up
    gemmlowp::Gemm<uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
        &context,
        lhs.const_map(),
        rhs.const_map(),
        &result,
        lhs_offset,
        rhs_offset,
        res_offset,
        res_mul,
        res_shift);

    int numRepeats = std::max(std::ceil(1e10 / (m * k * n)), 10.);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        gemmlowp::Gemm<uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
            &context,
            lhs.const_map(),
            rhs.const_map(),
            &result,
            lhs_offset,
            rhs_offset,
            res_offset,
            res_mul,
            res_shift);
    }
     
    auto end = std::chrono::steady_clock::now();
 
    return std::chrono::duration<double, std::milli>(end - start).count() / numRepeats;
}

double time_gemm_helper(int m, int n, int k, bool a_t, bool b_t) {
#define HANDLE_MATRIX_ORDER(ta, tb)            \
    if (a_t == ta && b_t == tb) {              \
        return time_gemm<ta, tb>(m, n, k);     \
    }

    HANDLE_MATRIX_ORDER(false, false)
    HANDLE_MATRIX_ORDER(false, true)
    HANDLE_MATRIX_ORDER(true, false)
    HANDLE_MATRIX_ORDER(true, true)

#undef HANDLE_MATRIX_ORDER
}

int main(int argc, char** argv) {
    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t    b_t    time (msec)     GOPS " << std::endl;

    std::cout << "====== gemmlowp kernel ======" << std::endl;

    for (const auto &problem : inference_device_set) {
    	int m, n, k;
    	bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        double time = time_gemm_helper(m, n, k, a_t, b_t);
        double mops = 1e-6 * 2 * m * n * k / time; 

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << (a_t ? "t" : "n");
        std::cout << std::setw(7) << (b_t ? "t" : "n");
        std::cout << std::setw(13) << std::setprecision(6) << time;
        std::cout << std::setw(13) << std::setprecision(6) << mops; 
        std::cout << std::endl;
    }

    std::cout << "====== tflite internal kernel ======" << std::endl;
    for (const auto &problem : inference_device_set) {
    	int m, n, k;
    	bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        double time = time_tflite(m, n, k);
        double mops = 1e-6 * 2 * m * n * k / time;

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << "n";
        std::cout << std::setw(7) << "t";
        std::cout << std::setw(13) << std::setprecision(6) << time;
        std::cout << std::setw(13) << std::setprecision(6) << mops;
        std::cout << std::endl;

    }

    return 0;
}
    
		
