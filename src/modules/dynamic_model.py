import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .pe import rope_params
from src.modules.kvcache import KVCacheFullyStatic
from src.kernels.rope_triton import rope_apply_fused
#from src.kernels.rope import rope_apply_fused
import torch.amp as amp
from flash_attn import flash_attn_func, flash_attn_with_kvcache

def fast_rope_apply(x, freqs):
    res_fast = rope_apply_fused(x, freqs[0, :, 0])
    return res_fast


def modulate(x, shift, scale):
    scale = torch.clamp(scale, -1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@amp.autocast('cuda', enabled=False)
def get_rope_2d(head_dim, shape, freqs):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c//2, c //2], dim=1)
    # precompute multipliers
    f, s = shape
    seq_len = f * s
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, -1).expand(f, s, -1),
        freqs[1][:s].view(1, s, -1).expand(f, s, -1),
    ],
                        dim=-1).reshape(1, f, s, -1)


    return freqs_i

@amp.autocast('cuda', enabled=False)
def get_rope_2d_cond(head_dim, shape, freqs, ids):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c//2, c // 2], dim=1)
    # precompute multipliers
    f, s = shape
    seq_len = f * s
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, -1).expand(f, s, -1),
        freqs[1][ids:ids+s].view(1, s, -1).expand(f, s, -1),
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
    def __init__(self, dim, num_heads, 
                 window_size=-1, 
                 qk_norm=True, 
                 eps=1e-6, 
                 device='cuda', 
                 dtype=torch.bfloat16,
                 cross_attn=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, device, dtype)  
        self.cross_attn = cross_attn
    def forward(self, x, freqs, shape):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        f, l, frame_seq = shape
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
        
        f, s, _ = shape
        
        q_roped = fast_rope_apply(q, freqs)
        k_roped = fast_rope_apply(k, freqs)
        q_roped = q_roped.view(b, f, -1, n, d)
        k_roped = k_roped.view(b, f, -1, n, d)
        v = v.view(b, f, -1, n, d)
        
        

        q_roped = q_roped.view(b * f, -1, n, d)
        k_roped = k_roped.view(b * f, -1, n, d)
        v = v.view(b * f, -1, n, d)
        
        x = flash_attn_func(
            q_roped, k_roped, v,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1)
        )
        x = x.contiguous().view(b, f, -1, n, d)
        x = x.view(b, -1, n, d)
       
        
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
                 device='cuda',
                 dtype=torch.bfloat16,
                 img_token_only=False,
                 use_rope=True):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        num_token_per_frame = 273
        self.img_token_only = img_token_only
        self.kvcache = KVCacheFullyStatic(
            head_dim=self.head_dim,
            head_num=num_heads,
            cache_frame_max=window_size,
            num_token_per_frame=num_token_per_frame,
            device=device,
            dtype=dtype,
        )
        self.use_rope = use_rope

        
    def forward(self, x, freqs, shape,
                current_frame_idx: Optional[int] = None,
                read_rope_indices: Optional[torch.Tensor] = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        f, l, z_s = shape
        
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
            

            
        f_his = self.kvcache.cache_frame_max
        self.kvcache.store_kvcache(k, v, current_frame_idx)
        
        freqs_his = self.kvcache.get_rope(read_rope_indices)
        k_his, v_his = self.kvcache.get_kv()
        
        freqs_last = self.kvcache.get_last_rope(current_frame_idx)
        freqs_last = freqs_last.view(-1, d // 2)[None,:, None,:]
        
        k_his_ = k_his.view(b, f_his, -1, n, d).view(b, -1, n, d)
        freqs_his = freqs_his.view(-1, d // 2)[None, :, None, :]
        
        if self.use_rope:
            q_roped = fast_rope_apply(q, freqs_last)
            k_his_roped = fast_rope_apply(k_his_, freqs_his)
        else:
            q_roped = q
            k_his_roped = k_his_
        
        q_roped = q_roped.view(b, f, -1, n, d)
        k_his_roped = k_his_roped.view(b, f_his, -1, n, d)
        v_his = v_his.view(b, f_his, -1, n, d)
        
        q_roped_img = q_roped[:, :, :z_s].contiguous().permute(0, 2, 1, 3, 4)
        k_his_roped_img = k_his_roped[:, :, :z_s].contiguous().permute(0, 2, 1, 3, 4)
        v_his_img = v_his[:, :, :z_s].contiguous().permute(0, 2, 1, 3, 4)
        
        v_cur = v.view(b, f, -1, n, d)              # [b, f, l, n, d]
        v_cond = v_cur[:, :, z_s:].contiguous()     # [b, f, l-z_s, n, d]
        q_roped_img = q_roped_img.view(b * z_s, f, n, d)
        k_his_roped_img = k_his_roped_img.view(b * z_s, f_his, n, d)
        v_his_img = v_his_img.view(b * z_s, f_his, n, d)
        
        
        valid_len_scalar = torch.min(current_frame_idx + 1, self.kvcache.max_len)
        total_batch_size = q_roped_img.shape[0]  # 即 b * l
        cache_seqlens = valid_len_scalar.expand(total_batch_size).to(torch.int32)
        
        x_img = flash_attn_with_kvcache(
            q_roped_img, k_his_roped_img, v_his_img,
            k=None, v=None, rotary_cos=None,rotary_sin=None,
            cache_seqlens=cache_seqlens,
            causal=False,
        )
        x_img = x_img.view(b, z_s, -1, n, d).permute(0, 2, 1, 3, 4)
        x = torch.cat([x_img, v_cond], dim=2)
        x = x.contiguous().view(b, -1, n, d)
                
        
            
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    


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
                 dtype=torch.bfloat16,
                 use_rope: bool=True
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
                use_rope=use_rope,
            )
            self.norm_temporal = RMSNorm(dim, eps, **params)
            
        
        self.norm2 = RMSNorm(dim, eps, **params)
        self.ffn = SwiGLU(dim, ffn_dim, **params)
        # modulation
    def forward(
        self,
        x,
        freqs,
        current_frame_idx: Optional[int] = None,
        read_rope_indices: Optional[torch.Tensor] = None,   
        shape: Optional[Any] = None,
        train_image_flag: bool=False
    ):
        kwargs = {
            'shape': shape,
        }

        x = x + self.attn(
            self.norm1(x), 
            freqs,
            **kwargs,
        )
        if self.attn_temporal is not None:
            x = x + self.attn_temporal(
                self.norm_temporal(x),
                freqs,
                shape=shape,
                current_frame_idx=current_frame_idx,
                read_rope_indices=read_rope_indices,
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
        self.norm = RMSNorm(dim, eps, **params)
        self.head = nn.Linear(dim, out_dim, **params)


    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        x = self.head(self.norm(x))
        return x
    
    
    
class DynamicModel(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'qk_norm',
    ]
    @register_to_config
    def __init__(
        self,
        in_dim=32,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        out_dim=32,
        num_heads=16,
        num_layers=32,
        window_size=256,
        qk_norm=True,
        eps=1e-6,
        vocab_size=68,
        K_samples_step=128,
        img_token_only: bool=True,
        cross_attn: bool=False,
        use_rope: bool=True,
        device='cuda',
        dtype=torch.bfloat16,
    ):
        super().__init__()
        params = {
            'dtype': dtype,
            'device': device,
        }
        self.params = params
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.head_dim = dim // num_heads
        self.tps = 4
        self.time_token_replicated_num = 1
        self.patch_embedding = nn.Linear(
                in_dim, dim, **params)
  
        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                attn_type='temporal' if (i + 1) % self.tps == 0 else 'spatial',
                window_size=window_size,
                qk_norm=qk_norm,
                eps=eps,
                device=device,
                dtype=dtype,
                use_rope=use_rope
        ) for i in range(num_layers)])
        
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d // 2),
            rope_params(1024, d // 2),
        ],
                               dim=1)
        self.K_samples_step = K_samples_step
        self.head = Head(dim, out_dim, (1, 1), eps, **params)
        self.init_vocab(vocab_size, dim)
        self.init_time_embedding(K_samples_step, dim)
        self.init_registry_token(num_registry_tokens=4, dim=dim)
        self.init_weights()
        self.action_layernorm = LayerNorm(dim, eps, **params)
        self.ts_layernorm = LayerNorm(dim, eps, **params)
        #self.register_rmsnorm = RMSNorm(dim, eps, **params)
        
        self.register_buffer("base_indices", torch.arange(window_size, device=device), persistent=False)
        self.max_len = torch.tensor([window_size], device=device, dtype=torch.long)
        

    def _forward_inference(
        self,
        img,
        timestep,       # 外部传入 discrete index (LongTensor)
        timestep_stride,# 外部传入 discrete index (LongTensor)
        action_ids,
        current_frame_idx: torch.Tensor,
        read_rope_indices: torch.Tensor
    ):
        action_embeds = self.action_embedding(action_ids)[None]
        action_embeds = self.action_in(action_embeds)
        time_embeds = self.time_embed_table(timestep)
        time_embeds = self.time_in(time_embeds)
        stride_embeds = self.stride_embed_table(timestep_stride)
        stride_embeds = self.stride_in(stride_embeds)
        time_stride_embeds = torch.cat([time_embeds, stride_embeds], dim=-1)
        action_slot_len = action_embeds.shape[2]
        frame_seqlen = img.shape[2]
        
        device = img[0].device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        
        b, f, l, c = img.shape
        x = self.patch_embedding(img)  # [B*F, dim, H',
        
        frame_idx = torch.min(current_frame_idx, self.max_len-1)
        time_stride_embeds = time_stride_embeds[None, :, None, :].type_as(x)
        action_embeds = action_embeds + self.learnable_token[:, None, :, :].type_as(x)
        registry_tokens = self.registry_tokens[:, None, :, :].expand(b, f, -1, -1).type_as(x)
        x = torch.cat([x, time_stride_embeds, action_embeds, registry_tokens], dim=2).contiguous()
        x = x.view(b, -1, self.dim)  # [B, L, dim]
        
        freqs = self.freqs_input[frame_idx][:, :, None, :].to(device).contiguous()
        # 传入 kwargs
        kwargs = dict(
            freqs=freqs,
            shape=(f, frame_seqlen + 4 + 1 + action_slot_len, frame_seqlen), # Shape 也要对齐
            train_image_flag=False,
        )
        
        for idx, block in enumerate(self.blocks):
            # 注意：如果 block 内部 TemporalAttention 需要 kvcache，
            # 而这里没传 kvcache 对象，它会走 Branch 2 (self.kvcache)
            # 请确保 __init__ 里 KVCacheStatic 的 num_token_per_frame 足够大 (frame_seqlen + cond_len)
            kwargs.update(
                {
                    'current_frame_idx': current_frame_idx,
                    'read_rope_indices': read_rope_indices,
                }
            )
            x = block(x, **kwargs)
        
        x = self.head(x)
        x = x.view(b, f, -1, self.out_dim)
        return x[:, :, :frame_seqlen, :]#A.clamp(-1, 1)
    
    def get_read_rope_indices(self, idx: int):
        if idx < self.window_size:
            return self.base_indices
        else:
            ptr = idx % self.window_size
            oldest_ptr = (ptr + 1) % self.window_size
            rope_indices = (self.base_indices - oldest_ptr) % self.window_size # type: ignore
            return rope_indices
    
    def init_rope_cache(self, frame_seqlen, action_slot_len):
        f = self.window_size
        device = self.params['device']  
        freqs_img = get_rope_2d(
            self.head_dim,
            (f, frame_seqlen),
            self.freqs,
        )
        freqs_cond = get_rope_2d_cond(
            self.head_dim,
            (f,  self.time_token_replicated_num + action_slot_len + self.registry_tokens.shape[1]),
            self.freqs,
            frame_seqlen,
        )
        freqs = torch.cat([freqs_img, freqs_cond], dim=2)
        freqs = freqs.reshape(-1, self.head_dim // 2).to(device=device).contiguous()
        freqs = freqs.view(f, -1, self.head_dim // 2)
        self.freqs_input = freqs
        for block in self.blocks:
            if isinstance(block.attn_temporal, TemporalAttention):
                block.attn_temporal.kvcache.init_rope(freqs)
    
    
    def forward(self,
                img,
                timestep,
                timestep_stride,
                action_ids,
                current_frame_idx: int=None,
                read_rope_indices: torch.Tensor=None
                ):
    
            output =  self._forward_inference(img, 
                timestep, timestep_stride, action_ids,
                current_frame_idx, read_rope_indices)
            return output
    
    def init_vocab(self, vocab_size, proj_dim):
        r"""
        使用 One-hot 初始化，不需要梯度。
        """
        # 1. 创建 One-hot 矩阵作为权重
        # 形状为 (vocab_size, vocab_size)
        one_hot_weight = torch.eye(vocab_size,**self.params)

        # 2. 使用 nn.Embedding 并加载权重
        self.action_embedding = nn.Embedding.from_pretrained(one_hot_weight, freeze=True)

        # 3. 线性层投影：将 one-hot (vocab_size) 映射到 proj_dim
        # 如果你希望这一层也不需要梯度，可以手动设置
        self.action_in = nn.Sequential(
            nn.Linear(vocab_size, self.dim, **self.params),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim, **self.params)
        )
        
    def init_time_embedding(self, K_samples_step, dim):
        
        d_size = int(math.log2(K_samples_step) + 1)
        t_size = K_samples_step
        
        # 时间步的 One-hot
        time_weight = torch.eye(t_size, **self.params)
        self.time_embed_table = nn.Embedding.from_pretrained(time_weight, freeze=True)
        # 步长的 One-hot
        stride_weight = torch.eye(d_size, **self.params)
        self.stride_embed_table = nn.Embedding.from_pretrained(stride_weight, freeze=True)
        
        # 线性层：将 One-hot 映射到所需的 embedding 维度
        # 注意：这里的输入维度变为了 t_size 和 d_size
        self.time_in = nn.Sequential(
            nn.Linear(t_size, self.dim // 2, **self.params),
            nn.SiLU(),
            nn.Linear(self.dim // 2, self.dim // 2, **self.params)
        )
        self.stride_in = nn.Sequential(
            nn.Linear(d_size, self.dim // 2, **self.params),
            nn.SiLU(),
            nn.Linear(self.dim // 2, self.dim // 2, **self.params)
        )
        
    def init_registry_token(self, num_registry_tokens, dim):
        r"""
        Initialize registry tokens for the model.

        Args:
            num_registry_tokens (int): Number of registry tokens.
            dim (int): Dimension of the tokens.
        """

        self.registry_tokens = nn.Parameter(
            torch.zeros(1, num_registry_tokens, dim, **self.params)
        )
        self.learnable_token = nn.Parameter(
            torch.zeros(1, 1, dim, **self.params)
        )
    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, std=.02)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.action_in.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_in.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.stride_in.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        


        # init output layer
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.head.head.bias)
        #nn.init.normal_(self.head.head.weight, std=.02)
    
    def clean_kvcache(self):
        r"""
        清理 KV Cache。
        """
        for block in self.blocks:
            if isinstance(block.attn_temporal, TemporalAttention):
                block.attn_temporal.kvcache.clean_cache()





from dataclasses import dataclass
from typing import Tuple,Literal
from pydantic import BaseModel, Field, TypeAdapter
@dataclass
class DynamicModelConfig(BaseModel):
    type: Literal['dynamic_model'] = 'dynamic_model'  # <--- 关键标识
    in_dim: int = 32
    dim: int = 2048,
    ffn_dim: int = 8192,
    freq_dim: int = 256,
    out_dim: int = 32,
    num_heads: int = 16,
    num_layers: int = 32,
    window_size: int = -1,
    qk_norm: bool = True,
    eps: float = 1e-6,
    vocab_size: int = 68,
    K_samples_step: int = 128,
    img_token_only: bool = False,
    cross_attn: bool = False,
    use_rope: bool = True,  