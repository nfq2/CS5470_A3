import torch
import time
import argparse
from model import SimpleModel
from attention_student import *#naive_strided_attention

# (batch_size, seq_len, embed_dim, num_heads, stride)
CORRECTNESS_CONFIGS = [
    (2, 16, 32, 4, 2),
    (2, 17, 48, 4, 3),    # seq_len not divisible by stride, head_dim not a power of 2
    (1, 64, 128, 2, 8),
    (1, 5, 16, 1, 8),     # stride > seq_len: only key 0 is attended
    (1, 1, 1, 1, 1),      # smallest supported head and sequence
    (1, 33, 66, 2, 1),    # multiple warp chunks and no sparsity
    (1, 256, 256, 1, 1),  # maximum head_dim and attended-key count
]

@torch.no_grad()
def test_correctness(device):
    print("--- Running Correctness Test ---")
    all_passed = True
    for config in CORRECTNESS_CONFIGS:
        all_passed &= check_correctness(device, *config)

    if all_passed:
        print("\nCorrectness Test Passed!")
    else:
        print("\nCorrectness Test Failed!")
    print("-" * 30)
    return all_passed


def check_correctness(device, batch_size, seq_len, embed_dim, num_heads, stride):
    print(f"\nbatch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, "
          f"num_heads={num_heads}, stride={stride}")

    torch.manual_seed(0)
    # Use float64 for high precision checking
    model = SimpleModel(embed_dim, num_heads, stride).to(device).double()
    model.eval()

    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.double)
    
    # Get the Q, K, V projections to test the attention module directly
    qkv = model.layer.attention.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, model.layer.attention.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # 1. Run the naive PyTorch implementation
    print("Running naive PyTorch implementation...")
    naive_output = naive_strided_attention(q, k, v, stride)
    
    # 2. Run the custom CUDA kernel implementation
    print("Running custom CUDA implementation...")
    
    # We need to get the output from just the attention calculation, not the final projection
    # So we call the kernel directly via the binding
    from cuda.binding import strided_attention_forward
    custom_output = strided_attention_forward(q.contiguous(), k.contiguous(), v.contiguous(), stride)

    # 3. Compare the results
    print("Comparing results...")
    is_close = torch.allclose(naive_output, custom_output, rtol=1e-5, atol=1e-6)

    if is_close:
        print("Passed")
    else:
        print("Failed! Max difference:", (naive_output - custom_output).abs().max().item())
    # Also exercise reshaping, contiguous copies, and the output projection.
    expected = naive_output.permute(0, 2, 1, 3).contiguous()
    expected = model.layer.attention.out_proj(expected.reshape(batch_size, seq_len, embed_dim))
    actual = model.layer.attention(x)
    integration_close = torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
    print("Attention layer integration:", "Passed" if integration_close else "Failed")
    if not integration_close:
        print("Max integration difference:", (expected - actual).abs().max().item())
    return is_close and integration_close


def test_benchmark(device):
    print("--- Running Benchmark Test ---")
    
    batch_size = 16
    seq_len = 1024
    embed_dim = 512
    num_heads = 8
    stride = 4
    
    warmup_iter = 10
    bench_iter = 100

    model = SimpleModel(embed_dim, num_heads, stride).to(device).float()
    model.eval()

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float)
    
    qkv = model.layer.attention.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, model.layer.attention.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
    
    # --- Benchmark Naive PyTorch ---
    print("Warming up naive implementation...")
    for _ in range(warmup_iter):
        _ = naive_strided_attention(q, k, v, stride)
    torch.cuda.synchronize()
    
    print("Running benchmark for naive implementation...")
    start_time = time.time()
    for _ in range(bench_iter):
        _ = naive_strided_attention(q, k, v, stride)
    torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / bench_iter
    print(f"Naive PyTorch implementation average time: {naive_time * 1000:.4f} ms")
    
    # --- Benchmark Custom CUDA ---
    from cuda.binding import strided_attention_forward

    # A speedup only counts if the kernel is correct
    naive_output = naive_strided_attention(q, k, v, stride)
    custom_output = strided_attention_forward(q, k, v, stride)
    if not torch.allclose(naive_output, custom_output, rtol=1e-3, atol=1e-4):
        print("\nCustom CUDA output does not match the naive implementation. Max difference:",
              (naive_output - custom_output).abs().max().item())
        print("Speedup: N/A (fix correctness first)")
        print("-" * 30)
        return

    print("\nWarming up custom CUDA implementation...")
    for _ in range(warmup_iter):
        _ = strided_attention_forward(q, k, v, stride)
    torch.cuda.synchronize()

    print("Running benchmark for custom CUDA implementation...")
    start_time = time.time()
    for _ in range(bench_iter):
        _ = strided_attention_forward(q, k, v, stride)
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / bench_iter
    print(f"Custom CUDA implementation average time: {custom_time * 1000:.4f} ms")

    # --- Results ---
    speedup = naive_time / custom_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "benchmark"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This assignment requires a CUDA-enabled GPU.")
        exit()
    
    device = torch.device("cuda")
    
    if args.mode == "correctness":
        raise SystemExit(0 if test_correctness(device) else 1)
    elif args.mode == "benchmark":
        test_benchmark(device)
