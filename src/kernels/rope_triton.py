from typing import Any, cast
import torch 
from torch.library import custom_op, triton_op
import triton 
import triton.language as tl
@triton.jit 
def _rope_apply_fused_kernel_triton(x_ptr, freq_ptr, out_ptr, S, BLOCK_M: tl.constexpr, H: tl.constexpr, C: tl.constexpr, is_fwd: tl.constexpr):
    idx_bh = tl.program_id(0)
    idx_s = tl.program_id(1)
    idx_b = idx_bh // H
    idx_h = idx_bh % H
    x_ptr += (idx_b * S * H * C + idx_h * C)
    # freq_ptr += idx_s * (C)
    out_ptr += (idx_b * S * H * C + idx_h * C)
    offs_m = tl.arange(0, BLOCK_M) + idx_s * BLOCK_M
    offs_n_x = tl.arange(0, C)
    x = tl.load(x_ptr + offs_m[:, None] * H * C + offs_n_x[None, :], mask=offs_m[:, None] < S, other=0.0)
    x = x.reshape(BLOCK_M, C // 2, 2).to(tl.float64)
    x0, x1 = tl.split(x)
    freq = tl.load(freq_ptr + offs_m[:, None] * C + offs_n_x[None, :], mask=offs_m[:, None] < S, other=0.0)
    freq = freq.reshape(BLOCK_M, C // 2, 2)
    f0, f1 = tl.split(freq)
    if is_fwd:
        res0 = x0 * f0 - x1 * f1
        res1 = x0 * f1 + x1 * f0
    else:
        res0 = x0 * f0 + x1 * f1
        res1 = -x0 * f1 + x1 * f0
    res = tl.join(res0, res1).to(x.dtype).reshape(BLOCK_M, C)
    tl.store(out_ptr + offs_m[:, None] * H * C + offs_n_x[None, :], res, mask=offs_m[:, None] < S)
@custom_op("creativeailib::rope_apply_fused_triton", mutates_args=(), device_types="cuda")
def rope_apply_fused_kernel_triton(x: torch.Tensor, freq: torch.Tensor, is_fwd: bool) -> torch.Tensor:
    # x: [B, S, H, C], half
    # freq: [S, C // 2], complex128
    assert x.is_contiguous() and freq.is_contiguous(), "Inputs must be contiguous"
    assert freq.dtype == torch.complex128, "freq must be complex128"
    C = x.shape[-1]
    H = x.shape[-2]
    S = x.shape[-3]
    out = torch.empty_like(x)
    BLOCK_M = 32
    #grid = (triton.cdiv(S, BLOCK_M), x.shape[0] * H)
    grid = (x.shape[0] * H, triton.cdiv(S, BLOCK_M))
    ck = _rope_apply_fused_kernel_triton[grid](
        x_ptr=x,
        freq_ptr=freq.view(torch.float64),
        out_ptr=out,
        S=S,
        BLOCK_M=BLOCK_M,
        H=H,
        C=C,
        is_fwd=is_fwd,
    )
    # print(ck.asm["ptx"])
    return out
@rope_apply_fused_kernel_triton.register_fake
def _(x, freq, is_fwd):
    return torch.empty_like(x)
def _rope_setup_context(ctx, inputs, output):
    x, freq, _ = inputs
    ctx.save_for_backward(freq)
def _rope_backward_triton(ctx, grad_output):
    freq, = ctx.saved_tensors
    # dout = torch.empty_like(grad_output)
    dout = rope_apply_fused_kernel_triton(grad_output, freq, False)
    return dout, None, None
rope_apply_fused_kernel_triton.register_autograd(_rope_backward_triton, setup_context=_rope_setup_context)
def rope_apply_fused(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x:          [B, L, N, C].
    freqs:      [L, C // 2].
    """
    return rope_apply_fused_kernel_triton(x, freqs, True)



def rope_apply_ref(x, freqs):
    origin_dtype = x.dtype
    b, s, n, c = x.shape
    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(
        b, s, n, -1, 2))
    # apply rotary embedding
    x = torch.view_as_real(x * freqs.view(s, 1, -1)).flatten(3)
    return x.to(origin_dtype).reshape(b, s, n, c)



@torch.compile
def rope_apply_dev(x, freqs_f64, out_dtype= None):
    if out_dtype is None:
        out_dtype = x.dtype
    b, s, n, c = x.shape
    # precompute multipliers
    x = x.to(freqs_f64.dtype).reshape(
        b, s, n, -1, 2)
    freq = freqs_f64.view(1, s, 1, -1, 2)
    # apply rotary embedding
    x0 = x[..., 0]
    x1 = x[..., 1]
    f0 = freq[..., 0]
    f1 = freq[..., 1]
    res0 = x0 * f0 - x1 * f1
    res1 = x0 * f1 + x1 * f0
    x = torch.stack([res0.to(out_dtype), res1.to(out_dtype)], dim=-1)
    return x.reshape(b, s, n, c)
def _main():
    x = torch.randn(1, 273*1024, 24, 128, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(273*1024, 64, device="cuda", dtype=torch.complex128)
    
    print(x.stride())
    o_triton = rope_apply_fused_kernel_triton(x, freqs, True)
    print(o_triton)
    o_ref = rope_apply_ref(x, freqs)
    print("L2 Error:", torch.linalg.norm((o_ref - o_triton)))
    ms_triton = triton.testing.do_bench(
        lambda: rope_apply_fused_kernel_triton(x, freqs, True))
    # print(f"Ref Time: {ms_ref} ms")
    print(f"Triton Time: {ms_triton} ms")
    
    freqs_compile = torch.randn(273*1024, 64, device="cuda", dtype=torch.complex128).view(torch.float64).to(torch.float32)
    ms_triton = triton.testing.do_bench(
        lambda: rope_apply_dev(x, freqs_compile))
    print(f"Triton Time (freqs as float32): {ms_triton} ms")
if __name__ == "__main__":
    _main()