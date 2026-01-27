import torch



class KVCacheFullyStatic(torch.nn.Module):
    def __init__(self, 
                 head_dim: int,
                 head_num: int,
                 cache_frame_max: int,
                 num_token_per_frame: int,
                 device: torch.device,
                 dtype: torch.dtype):
        super().__init__()
        self.cache_frame_max = cache_frame_max
        self.num_token_per_frame = num_token_per_frame
        self.head_dim = head_dim
        self.head_num = head_num
        self.cached_frames = 0
        self.device = device
        self.dtype = dtype
        self.max_len = torch.scalar_tensor(cache_frame_max, dtype=torch.long, device=device)
        
        self.register_buffer("k", torch.zeros(
                (cache_frame_max, num_token_per_frame, head_num, head_dim),
                device=device,
                dtype=dtype), persistent=False)
        
        self.register_buffer("v", torch.zeros(
                (cache_frame_max, num_token_per_frame, head_num, head_dim),
                device=device,
                dtype=dtype), persistent=False)
        
        self.register_buffer("rope", torch.zeros(
                (cache_frame_max, num_token_per_frame, head_dim//2),
                device=device,
                dtype=torch.complex128), persistent=False)
        
        

    def store_kvcache(self, k, v, idx: torch.Tensor):
        
        idx = idx % self.max_len
        self.k[idx] = k[0] # type: ignore
        self.v[idx] = v[0] # type: ignore
        
    def init_rope(self, rope):
        self.rope.copy_(rope)  # type: ignore
    
    def get_rope(self, read_rope_indices: torch.Tensor):
        return self.rope[read_rope_indices]  # type: ignore
    
    def get_last_rope(self, idx: torch.Tensor):
        idx = torch.min(idx, self.max_len - 1)
        return self.rope[idx]  # type: ignore
    
    
    def get_kv(self):
        return self.k, self.v  # type: ignore
    
    def clean_cache(self):
        self.k.zero_()
        self.v.zero_()