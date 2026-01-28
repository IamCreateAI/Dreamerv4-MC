import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .pe import rope_params
from src.modules.kvcache import KVCacheFullyStatic
from src.kernels.rope_triton import rope_apply_fused
import torch.amp as amp
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from src.utils.utils import CUDAGraphRunner
def fast_rope_apply(x, freqs):
    res_fast = rope_apply_fused(x, freqs)
    return res_fast

@amp.autocast('cuda', enabled=False)
def get_rope_patch(head_dim, shape, freqs):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    f, h, w = shape
    seq_len = f * h * w
    # precompute multipliers
    
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ],
                        dim=-1).reshape(1, f, h * w, -1)


    return freqs_i

@amp.autocast('cuda', enabled=False)
def get_rope_latent(head_dim, shape, freqs):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # precompute multipliers
    f, s = shape
    seq_len = f * s
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, -1).expand(f, s, -1),
        freqs[1][0].view(1, 1, -1).expand(f, s, -1),
        freqs[2][0].view(1, 1, -1).expand(f, s, -1)
    ],
                        dim=-1).reshape(1, f, s, -1)


    return freqs_i


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, **kwargs))

    # @torch.compiler.disable
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    

    
class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False, **kwargs):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps, **kwargs)

    # @torch.compiler.disable
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)
    
class SwiGLU(nn.Module):
    def __init__(self, 
                 hidden_dim=3072, 
                 ffn_dim=9216, 
                 device='cuda', 
                 dtype=torch.bfloat16):
        params = {
            'dtype': dtype,
            'device': device,
        }
        
        super().__init__()
        # 3 个矩阵，参数量满满
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False, **params) # Gate
        self.w2 = nn.Linear(hidden_dim, ffn_dim, bias=False, **params) # Value
        self.w3 = nn.Linear(ffn_dim, hidden_dim, bias=False, **params) # Output
       
    def forward(self, x):
        # Swish(xW1) * xW2
        x1 = F.silu(self.w1(x))
        x2 = self.w2(x)
        return self.w3(x1 * x2) 

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=-1,
                 qk_norm=True,
                 eps=1e-6,
                 device='cuda',
                 dtype=torch.bfloat16):
        params = {
            'dtype': dtype,
            'device': device,
        }
        
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim, **params)
        self.k = nn.Linear(dim, dim, **params)
        self.v = nn.Linear(dim, dim, **params)
        self.o = nn.Linear(dim, dim, **params)
        self.norm_q = RMSNorm(dim, eps=eps, **params) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, **params) if qk_norm else nn.Identity()

class SpatialAttention(AttentionBase):
    def forward(self, x, freqs, shape):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v


        q, k, v = qkv_fn(x)
        if q.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
        if k.dtype != v.dtype:
            k = k.to(dtype=v.dtype)
        
        b, f, num_latent_token = shape
        q = q.view(b, f, -1, n, d)
        k = k.view(b, f, -1, n, d)
        
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        
        q_roped = fast_rope_apply(q, freqs)
        k_roped = fast_rope_apply(k, freqs)
        
        q_roped = q_roped.view(b, f, -1, n, d)
        k_roped = k_roped.view(b, f, -1, n, d)
        
        q_roped = q_roped.contiguous().view(b * f, -1, n, d)
        k_roped = k_roped.contiguous().view(b * f, -1, n, d)
        
        q_roped_patch = q_roped[:, :-num_latent_token, :, :]
        q_roped_latent = q_roped[:, -num_latent_token:, :, :]
        
        k_roped_patch = k_roped[:, :-num_latent_token, :, :]
        v_patch = v[:, :-num_latent_token, :, :]
        
        x_patch = flash_attn_func(
            q_roped_patch, k_roped_patch, v_patch,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1)
        )
        
        x_latent = flash_attn_func(
            q_roped_latent, k_roped, v,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1)
        )
        x = torch.cat([x_patch, x_latent], dim=1)
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x 

class TemporalAttention(AttentionBase):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=-1,
                 qk_norm=True,
                 eps=1e-6,
                 num_token_per_frame=1472,
                 device='cuda',
                 dtype=torch.bfloat16):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        #if not self.training:
        '''
        self.kvcache = KVCacheStatic(
            head_dim=self.head_dim,
            head_num=num_heads,
            cache_frame_max=window_size,
            num_token_per_frame=num_token_per_frame,
            device=device,
            dtype=dtype,
        )
        '''
        self.kvcache = KVCacheFullyStatic(
            head_dim=self.head_dim,
            head_num=num_heads,
            cache_frame_max=window_size,
            num_token_per_frame=num_token_per_frame,
            device=device,
            dtype=dtype,
        )
        self.use_graph_graph = True
        
    def forward(self, x, freqs, shape,
                current_frame_idx: Optional[Union[int, torch.Tensor]] = None,
                read_rope_indices: Optional[torch.Tensor] = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        bv, f, num_latent_token = shape
        
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if q.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
        if k.dtype != v.dtype:
            k = k.to(dtype=v.dtype)
            
        

        if current_frame_idx is None:
            q = q.view(bv, f, -1, n, d)
            k = k.view(bv, f, -1, n, d)
            
            q = q.view(bv, -1, n, d)
            k = k.view(bv, -1, n, d)
            
            q_roped = fast_rope_apply(q, freqs)
            k_roped = fast_rope_apply(k, freqs)
            
            q_roped = q_roped.view(bv, f, -1, n, d).permute(0, 2, 1, 3, 4)
            k_roped = k_roped.view(bv, f, -1, n, d).permute(0, 2, 1, 3, 4)
            
            q_roped = q_roped.contiguous().view(bv * s, -1, n, d)
            k_roped = k_roped.contiguous().view(bv * s, -1, n, d)
            
            v = v.view(bv, f, -1, n, d).permute(0, 2, 1, 3, 4)
            v = v.contiguous().view(bv * s, -1, n, d)
    
            if self.window_size > 0:
                x = flash_attn_func(
                    q_roped, k_roped, v,
                    dropout_p=0.0,
                    causal=True,
                    window_size=(self.window_size-1, -1)
                )
                
            else:
                x = flash_attn_func(
                    q_roped, k_roped, v,
                    dropout_p=0.0,
                    causal=True,
                    window_size=(-1, -1)
                )
            x = x.view(bv, s, -1, n, d).permute(0, 2, 1, 3, 4)
            x = x.contiguous().view(bv * f, -1, n, d)
        else:
            f_his = self.kvcache.cache_frame_max
            self.kvcache.store_kvcache(k, v, current_frame_idx)
            
            freqs_his = self.kvcache.get_rope(read_rope_indices)
            k_his, v_his = self.kvcache.get_kv()
            
            freqs_last = self.kvcache.get_last_rope(current_frame_idx)
            freqs_last = freqs_last.view(-1, d // 2)
            
            q = q.view(bv, f, -1, n, d)
            q = q.view(bv, -1, n, d)
            q_roped = fast_rope_apply(q, freqs_last)
            q_roped = q_roped.view(bv, f, -1, n, d).permute(0, 2, 1, 3, 4)
            
            k_his = k_his.view(bv, f_his, -1, n, d).view(bv, -1, n, d)
            freqs_his = freqs_his.view(-1, d // 2)
            k_his_roped = fast_rope_apply(k_his, freqs_his)
            k_his_roped = k_his_roped.view(bv, f_his, -1, n, d).permute(0, 2, 1, 3, 4)
            v_his = v_his.view(bv, f_his, -1, n, d).permute(0, 2, 1, 3, 4)

            
            q_roped = q_roped.contiguous().view(bv * s, -1, n, d)
            k_his_roped = k_his_roped.contiguous().view(bv * s, -1, n, d)
            v_his = v_his.contiguous().view(bv * s, -1, n, d)
            

            valid_len_scalar = torch.min(current_frame_idx + 1, self.kvcache.max_len)
            total_batch_size = q_roped.shape[0]  # 即 b * l
            cache_seqlens = valid_len_scalar.expand(total_batch_size).to(torch.int32).contiguous()
            
            x = flash_attn_with_kvcache(
                q_roped, k_his_roped, v_his,
                k=None, v=None, rotary_cos=None,rotary_sin=None,
                cache_seqlens=cache_seqlens,
                causal=False,
            )
            x = x.view(bv, s, -1, n, d).permute(0, 2, 1, 3, 4)
            x = x.contiguous().view(bv * f, -1, n, d)
                
           
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 attn_type: str,
                 window_size=-1,
                 qk_norm=True,
                 eps=1e-6,
                 device='cuda',
                 dtype=torch.bfloat16
    ):
        params = {
            'dtype': dtype,
            'device': device,
        }

        super().__init__()
        self.norm1 = RMSNorm(dim, eps, **params)
        self.attn_temporal = None
        if attn_type == "spatial":
            self.attn = SpatialAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
            )
        elif attn_type == "temporal":
            self.attn = SpatialAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            self.attn_temporal = TemporalAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            self.norm_temporal = RMSNorm(dim, eps, **params)
        
        self.norm2 = RMSNorm(dim, eps, **params)
        self.ffn = SwiGLU(
            hidden_dim=dim,
            ffn_dim=ffn_dim,
            device=device,
            dtype=dtype,
        )

        # modulation
    def forward(
        self,
        x, 
        freqs,
        current_frame_idx: Optional[Union[int, torch.Tensor]] = None,
        read_rope_indices: Optional[torch.Tensor] = None,
        shape: Optional[Any] = None,
    ):
       
        kwargs = {
            'shape': shape,
        }
        if isinstance(self.attn, TemporalAttention):
            kwargs.update(
                {"current_frame_idx": current_frame_idx, 
                 "read_rope_indices": read_rope_indices}
            )
        x = x + self.attn(
            self.norm1(x), 
            freqs,
            **kwargs,
        )
        if self.attn_temporal is not None:
            x = x + self.attn_temporal(
                self.norm_temporal(x),
                freqs,
                current_frame_idx=current_frame_idx,
                read_rope_indices=read_rope_indices,
                shape=shape,
            )
        x = x + self.ffn(self.norm2(x))
        return x
    

    
class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, device='cuda', dtype=torch.bfloat16):
        super().__init__()

        params = {
            'dtype': dtype,
            'device': device,
        }

        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps, **params)
        self.head = nn.Linear(dim, out_dim, **params)
        self.tanh = nn.Tanh()
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
        """
        x = self.head(self.norm(x))
        x = self.tanh(x)
        return x
    
    




class Encoder(nn.Module):
    def __init__(self,
                 patch_size=(16, 16),
                 in_dim=3,
                 dim=1024,
                 ffn_dim=4096,
                 out_dim=16,
                 num_heads=8,
                 num_layers=20,
                 num_learnable_tokens=512,
                 mask_ratio_max=0.5,
                 use_patch_drop=True,
                 window_size=-1,
                 qk_norm=True,
                 eps=1e-6,
                 device='cuda',
                 dtype=torch.bfloat16
    ):
        super().__init__()
        params = {
            'dtype': dtype,
            'device': device
        }
        self.params = params
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.num_layers = num_layers
        # layers
        self.patch_embedding = nn.Conv2d(
            in_channels=in_dim,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            **params,
        )
        '''
        self.causal_block = nn.ModuleList([
            BlockCausalLayer(dim, ffn_dim, num_heads,
                              window_size, qk_norm, eps, **params)
            for _ in range(num_layers // 4)
        ])
        '''
        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                attn_type='temporal' if (i + 1) % 4 == 0 else 'spatial',
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
        ) for i in range(num_layers)])
    
        self.learnable_token = nn.Parameter(
            torch.randn((1, num_learnable_tokens , dim), **params))
        nn.init.normal_(self.learnable_token, std=.02)
        self.mask_token = nn.Parameter(
            torch.randn((1, 1, dim), **params))
        nn.init.normal_(self.mask_token, std=.02)
        self.pos_embedding = nn.Parameter(
            torch.randn((1, num_learnable_tokens, dim), **params))
        nn.init.normal_(self.pos_embedding, std=.02)
        self.post_norm = RMSNorm(dim, eps, **params)
        self.proj = nn.Linear(dim, out_dim, **params)
        self.tanh = nn.Tanh()
        self.mask_ratio_max = mask_ratio_max
        self.use_patch_drop = use_patch_drop
        self.num_learnable_tokens = num_learnable_tokens
        self.out_dim = out_dim  
        d = self.head_dim
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],                            dim=1)
        
        self.register_buffer("base_indices", torch.arange(window_size, device=device), persistent=False)
        self.max_len = torch.tensor([window_size], device=device, dtype=torch.long)
        
    def forward_inference_static(self, x,
                                 current_frame_idx: torch.Tensor, 
                                 read_rope_indices: Optional[torch.Tensor] = None):
        b, f, c, h, w = x.shape
        N = b * f
        # patch embedding
        x = x.view(N, c, h, w)
        x = self.patch_embedding(x)  # [B*F, dim, H',
        h, w = x.shape[2], x.shape[3]
        x = x.view(N, self.dim, -1).permute(0, 2, 1)  # [B*F, L', dim]
        L = x.shape[1]
        
        x = torch.cat([x, self.learnable_token.expand(N, -1, -1) + self.pos_embedding], dim=1)  # [B*F, 1+L', dim]
        frame_idx = torch.min(current_frame_idx, self.max_len-1)
        
        freqs = self.freqs_input[frame_idx][None]
        
        kwargs = dict(
            freqs=freqs,
            shape=(b, f, self.num_learnable_tokens),
            current_frame_idx=current_frame_idx,
            read_rope_indices=read_rope_indices,
        )
        
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
         #x = self.model(x, **kwargs)  # [B*F, 1+L', dim]
        x = self.post_norm(x)
        x = self.proj(x)  # [B*F, 1+L', out_dim]
        x = x.view(b, f, -1, self.out_dim)  # [B, F, 1+L', out_dim]
        x = self.tanh(x)
        return x[:, :, -self.num_learnable_tokens:]
    
    
    def get_read_rope_indices(self, idx: int):
        if idx < self.window_size:
            return self.base_indices
        else:
            ptr = idx % self.window_size
            oldest_ptr = (ptr + 1) % self.window_size
            rope_indices = (self.base_indices - oldest_ptr) % self.window_size # type: ignore
            return rope_indices

    def init_rope_cache(self, shape):
        b, f, c, h, w = shape
        f = self.window_size
        device = self.params['device']  
        freqs_patch = get_rope_patch(self.head_dim, (f, h, w), self.freqs)
        freqs_latent = get_rope_latent(self.head_dim, (f, self.num_learnable_tokens), self.freqs)
            
        freqs = torch.cat([freqs_patch, freqs_latent], dim=2)
        freqs = freqs.reshape(-1, self.head_dim // 2).to(device=device).contiguous()
        freqs = freqs.view(f, -1, self.head_dim // 2)
        self.freqs_input = freqs
        for block in self.blocks:
            if isinstance(block.attn, TemporalAttention):
                block.attn.kvcache.init_rope(freqs)
    
    
    def forward_training(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, F, 3, H, W]
        """
        b, f, c, h, w = x.shape
        N = b * f
        # patch embedding
        x = x.view(N, c, h, w)
        x = self.patch_embedding(x)  # [B*F, dim, H',
        h, w = x.shape[2], x.shape[3]
        x = x.view(N, self.dim, -1).permute(0, 2, 1)  # [B*F, L', dim]
        L = x.shape[1]
    
        x = torch.cat([x, self.learnable_token.expand(N, -1, -1) + self.pos_embedding], dim=1)  # [B*F, 1+L', dim]
        
        freqs_patch = get_rope_patch(self.head_dim, (f, h, w), self.freqs)
        freqs_latent = get_rope_latent(self.head_dim, (f, self.num_learnable_tokens), self.freqs)
            
        freqs = torch.cat([freqs_patch, freqs_latent], dim=2)
        freqs = freqs.reshape(-1, self.head_dim // 2).to(x.device)
        
        kwargs = dict(
            freqs=freqs,
            shape=(b, f, self.num_learnable_tokens)
        )
        for block in self.blocks:
            
            x = block(x, **kwargs)
        #x = self.model(x, **kwargs)  # [B*F, 1+L', dim]
        x = self.post_norm(x)
        x = self.proj(x)  # [B*F, 1+L', out_dim]
        x = x.view(b, f, -1, self.out_dim)  # [B, F, 1+L', out_dim]
        x = self.tanh(x)
        return x[:, :, -self.num_learnable_tokens:]   
        
    def forward(self, x, current_frame_idx=None, 
                read_rope_indices: Optional[torch.Tensor] = None):
        if current_frame_idx is None:
            return self.forward_training(x)
        else:
            return self.forward_inference_static(x, current_frame_idx, read_rope_indices)

class Decoder(nn.Module):
    def __init__(self,
                 patch_size=(16, 16),
                 in_dim=16,
                 dim=1024,
                 ffn_dim=4096,
                 out_dim=3,
                 num_heads=8,
                 num_layers=32,
                 num_learnable_tokens=512,
                 window_size=-1,
                 qk_norm=True,
                 eps=1e-6,
                 device='cuda',
                 dtype=torch.bfloat16,
    ):
        super().__init__()
        params = {
            'dtype': dtype,
            'device': device,
        }
        self.patch_size = patch_size    
        self.head_dim = dim // num_heads
        self.dim = dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.proj = nn.Linear(in_dim, dim, **params)
        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                attn_type='temporal' if (i + 1) % 4 == 0 else 'spatial',
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
        ) for i in range(num_layers)])
        self.learnable_token = nn.Parameter(
            torch.randn((1, 1, dim), **params))
        nn.init.normal_(self.learnable_token, std=.02)
        self.pos_embedding = nn.Parameter(
            torch.randn((1, num_learnable_tokens, dim), **params))
        nn.init.normal_(self.pos_embedding, std=.02)
        self.num_learnable_tokens = num_learnable_tokens
        self.head = Head(
            dim=dim,
            out_dim=out_dim,
            patch_size=patch_size,
            eps=eps,
            **params    
        )
        d = self.head_dim
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],                            dim=1)
        self.params = params
        
        self.register_buffer("base_indices", torch.arange(window_size, device=device), persistent=False)
        self.max_len = torch.tensor([window_size], device=device, dtype=torch.long)

    
    def forward_inference_static(self, z, shape, 
                                 current_frame_idx: torch.Tensor, 
                                 read_rope_indices: Optional[torch.Tensor] = None):
        b, f, c, h, w = shape
        x = self.learnable_token.expand(b * f, h * w, -1)
        z = self.proj(z).view(b * f, -1, self.dim) + self.pos_embedding
        x = torch.cat([z, x], dim=1)
        frame_idx = torch.min(current_frame_idx, self.max_len-1)
        
        freqs = self.freqs_input[frame_idx][None]
        
        kwargs = dict(
            freqs=freqs,
            shape=(b, f, h*w),
            current_frame_idx=current_frame_idx,
            read_rope_indices=read_rope_indices,
        )
        out_hidden_dim = math.prod(self.patch_size) * self.out_dim
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
        #x = self.model(x, **kwargs)  # [B*F, 1+L', dim]
        x = x[:, -h*w:]  # [B*F, L', dim]
        x = self.head(x).view(b, f, -1, out_hidden_dim)
        x = x.view(b, -1, out_hidden_dim)
        

        grid_sizes = [f, h, w]
        x = self.unpatchify(x, grid_sizes=grid_sizes)
        return x
    def get_read_rope_indices(self, idx: int):
        if idx < self.window_size:
            return self.base_indices
        else:
            ptr = idx % self.window_size
            oldest_ptr = (ptr + 1) % self.window_size
            rope_indices = (self.base_indices - oldest_ptr) % self.window_size # type: ignore
            return rope_indices

    def init_rope_cache(self, shape):
        b, f, c, h, w = shape
        f = self.window_size
        device = self.params['device']  
        freqs_patch = get_rope_patch(self.head_dim, (f, h, w), self.freqs)
        freqs_latent = get_rope_latent(self.head_dim, (f, self.num_learnable_tokens), self.freqs)
        
        freqs = torch.cat([freqs_latent, freqs_patch], dim=2)
        freqs = freqs.reshape(-1, self.head_dim // 2).to(device=device).contiguous()
        freqs = freqs.view(f, -1, self.head_dim // 2)
        self.freqs_input = freqs
        for block in self.blocks:
            if isinstance(block.attn_temporal, TemporalAttention):
                block.attn_temporal.kvcache.init_rope(freqs)
    
    def clean_kvcache(self):
        for block in self.blocks:
            if isinstance(block.attn_temporal, TemporalAttention):
                block.attn_temporal.kvcache.clean_cache()
        
    
    def forward(
        self,
        z,
        shape,
        current_frame_idx: Optional[Union[int, torch.Tensor]] = None,
        read_rope_indices: Optional[torch.Tensor] = None, 
    ):

           
        return self.forward_inference_static(z, shape, current_frame_idx, read_rope_indices)
         
    def unpatchify(self, x, grid_sizes):
        """
        x: [Batch, Seq_Len, Patch_Pixel_Dim]
        grid_sizes: [Batch, 3] (假设是 Time, H, W 的 grid) 
        """
        v = grid_sizes
        B = x.shape[0]
        p = (1, *self.patch_size)
        c = x.shape[-1] // math.prod(p)
        u = x.view(B, *v, *p, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        final_shape = [B, c] + [g * patch for g, patch in zip(v, p)]
        out = u.reshape(*final_shape)
        
        return out
        
class CausalTokenizer(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'patch_size', 'qk_norm', 'device', 'dtype'
    ]
    @register_to_config
    def __init__(self,
                 patch_size=(16, 16),
                 in_dim=3,
                 dim=1024,
                 ffn_dim=4096,
                 encode_out_dim=16,
                 out_dim=3,
                 num_heads=8,
                 encoder_layers=16,
                 decoder_layers=16,
                 window_size=16,
                 qk_norm=True,
                 eps=1e-6,
                 device='cuda',
                 dtype=torch.bfloat16,
    ):
        super().__init__() # type: ignore
        params = {
            'dtype': dtype,
            'device': device,
        }
       
        self.patch_size = patch_size
        self.encoder = Encoder(
            patch_size=patch_size,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            out_dim=encode_out_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            **params
        )
        self.decoder = Decoder(
            patch_size=patch_size,
            in_dim=encode_out_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            **params
        )
        self.use_encoder_graph = False
        self.use_decoder_graph = False
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, F, 3, H, W]
        """
        tokens = self.encoder(x)  # [B, F, L', out_dim]
        shape = x.shape
        shape = (shape[0], 
                 shape[1], 
                 shape[2], 
                 shape[3] // self.patch_size[0], 
                 shape[4] // self.patch_size[1])  # [B, F, C, H, W]
        x_recon = self.decoder(tokens, shape=shape).permute(0,2,1,3,4)  # [B, F, 3, H, W]
        return tokens, x_recon
    
    
    def encode(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, F, 3, H, W]
        """
        try:
            tokens = self.encoder(x)  # [B, F, L', out_dim]
        except RuntimeError as e:
            print("Encoder forward failed")
        L = tokens.shape[2]
        return tokens.reshape(x.shape[0], x.shape[1], L//2, -1)
    
    
    def ar_encode(self, x, current_frame_idx):
        read_rope_indices = self.encoder.get_read_rope_indices(current_frame_idx.item())
        if self.use_encoder_graph:
            tokens_input = {
                'x': x,
                'current_frame_idx': current_frame_idx,
                'read_rope_indices': read_rope_indices
            }
            tokens = self.encoder_graph_runner(tokens_input).clone()
            L = tokens.shape[2]
        else:
            tokens = self.encoder(x, 
                                  current_frame_idx=current_frame_idx, 
                                  read_rope_indices=read_rope_indices)  # [B, F, L', out_dim]
            L = tokens.shape[2]
        return tokens.reshape(x.shape[0], x.shape[1], L//2, -1)
    
    def ar_encode_video(self, x):
        
        
        b, f, c, h, w = x.shape
        output = []
        for idx in range(f):
            current_x = x[:, idx:idx+1, :, :, :]
            idx = torch.tensor([idx], device=x.device)
            current_tokens = self.ar_encode(
                current_x,
                current_frame_idx=idx
            ) 
            output.append(current_tokens)
        output = torch.cat(output, dim=1)
        return output
        
    def decode(self, tokens, shape):
        r"""
        Args:
            tokens(Tensor): Shape [B, F, L', out_dim]
        """
        b, f, l2, d = tokens.shape
        tokens = tokens.reshape(b, f, l2 *2, d // 2)
        x_recon = self.decoder(tokens, shape=shape)  # [B, F, 3, H, W]
        return x_recon.permute(0,2,1,3,4)
    
    def ar_decode(self, tokens, shape, current_frame_idx):
        r"""
        Args:
            tokens(Tensor): Shape [B, F, L', out_dim]
        """
        b, f, l2, d = tokens.shape
        tokens = tokens.reshape(b, f, l2 *2, d // 2)
        read_rope_indices = self.decoder.get_read_rope_indices(current_frame_idx.item())
        if self.use_decoder_graph:
            tokens_input = {
                'z': tokens,
                'shape': shape,
                'current_frame_idx': current_frame_idx,
                'read_rope_indices': read_rope_indices
            }
            x_recon = self.decoder_graph_runner(tokens_input).clone()
        else:
            x_recon = self.decoder(
                tokens, 
                shape=shape,
                current_frame_idx=current_frame_idx,
                read_rope_indices=read_rope_indices
            )  # [B, F, 3, H, W]
        return x_recon.permute(0,2,1,3,4)
    
    
    def ar_decode_video(self, tokens, shape):
        b, f, l2, d = tokens.shape
        output = []
        for idx in range(f):
            current_tokens = tokens[:, idx:idx+1, :, :]
            idx = torch.tensor([idx], device=tokens.device)
            x_recon_frame = self.ar_decode(
                current_tokens,
                shape=(
                    b,
                    1,
                    shape[2],
                    shape[3],
                    shape[4]
                ),
                current_frame_idx=idx
            ) 
            output.append(x_recon_frame)
        output = torch.cat(output, dim=1)
        return output

    
    def record_encoder_graph(self, sample_input, autocast_kwargs=None):
        self.encoder_graph_runner = CUDAGraphRunner(
            model=self.encoder,
            sample_input=sample_input,
            autocast_kwargs=autocast_kwargs
        )
        self.encoder_graph_runner.capture()
        for block in self.encoder.blocks:
            if isinstance(block.attn, TemporalAttention):
                block.attn.kvcache.clean_cache()
        self.use_encoder_graph = True
    
   
    def record_decoder_graph(self, sample_input, autocast_kwargs=None):
        self.decoder_graph_runner = CUDAGraphRunner(
            model=self.decoder,
            sample_input=sample_input,
            autocast_kwargs=autocast_kwargs
        )
        self.decoder_graph_runner.capture()
        for block in self.decoder.blocks:
            if isinstance(block.attn, TemporalAttention):
                block.attn.kvcache.clean_cache()
        self.use_decoder_graph = True




