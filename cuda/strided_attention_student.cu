#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>
#include <limits>

// One 256-thread block per query. Each warp computes one strided key's
// score at a time; adjacent lanes read adjacent head dimensions.
template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int seq_len,
    const int head_dim,
    const int stride) {
    using Acc = at::acc_type<T, true>;
    __shared__ Acc query[256];
    __shared__ Acc scores[256];
    __shared__ Acc reduction[256];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int key_count = 1 + (seq_len - 1) / stride;
    const int64_t head_offset = static_cast<int64_t>(blockIdx.x) * seq_len * head_dim;
    const int64_t query_offset = head_offset + static_cast<int64_t>(blockIdx.y) * head_dim;
    const Acc scale = Acc(1) / sqrt(Acc(head_dim));

    if (tid < head_dim) {
        query[tid] = static_cast<Acc>(q_ptr[query_offset + tid]);
    }
    __syncthreads();

    // Only calculate scores for keys 0, stride, 2 * stride, ... .
    for (int key = warp; key < key_count; key += 8) {
        const int64_t key_offset = head_offset + static_cast<int64_t>(key) * stride * head_dim;
        Acc dot = Acc(0);
        for (int d = lane; d < head_dim; d += 32) {
            dot += query[d] * static_cast<Acc>(k_ptr[key_offset + d]);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) {
            scores[key] = dot * scale;
        }
    }
    __syncthreads();

    // Stable softmax. Inactive threads contribute -infinity to max and
    // zero to sum, including when the number of keys is not a power of two.
    reduction[tid] = tid < key_count ? scores[tid] : -INFINITY;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) {
            reduction[tid] = reduction[tid] > reduction[tid + offset]
                ? reduction[tid] : reduction[tid + offset];
        }
        __syncthreads();
    }
    const Acc max_score = reduction[0];
    const Acc weight = tid < key_count ? exp(scores[tid] - max_score) : Acc(0);
    // Every thread must finish reading the maximum before reusing scratch.
    __syncthreads();
    reduction[tid] = weight;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    if (tid < key_count) {
        scores[tid] = weight / (reduction[0] + Acc(1e-6));
    }
    __syncthreads();

    // Each thread owns an output dimension, so value loads are coalesced.
    if (tid < head_dim) {
        Acc result = Acc(0);
        for (int key = 0; key < key_count; ++key) {
            const int64_t value_offset = head_offset + static_cast<int64_t>(key) * stride * head_dim;
            result += scores[key] * static_cast<Acc>(v_ptr[value_offset + tid]);
        }
        output_ptr[query_offset + tid] = static_cast<T>(result);
    }
}

torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, int stride) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q, K, V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q, K, V must share a device");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "Q, K, V must share a dtype");
    TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "Q, K, V must have matching [batch, heads, sequence, head_dim] shapes");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Q, K, V must be contiguous");
    TORCH_CHECK(stride > 0, "stride must be positive");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);
    TORCH_CHECK(head_dim > 0 && head_dim <= 256, "head_dim must be between 1 and 256");
    TORCH_CHECK(seq_len <= 65535, "sequence length exceeds the CUDA grid y limit");
    TORCH_CHECK(seq_len == 0 || 1 + (seq_len - 1) / stride <= 256,
                "at most 256 strided keys are supported");
    TORCH_CHECK(batch_size * num_heads <= std::numeric_limits<int>::max(),
                "batch * heads exceeds the CUDA grid x limit");

    const c10::cuda::CUDAGuard device_guard(q.device());
    auto output = torch::empty_like(q);
    if (q.numel() == 0) {
        return output;
    }
    const dim3 grid(batch_size * num_heads, seq_len);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        strided_attention_forward_kernel<scalar_t><<<grid, 256, 0, stream.stream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), seq_len, head_dim, stride);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
