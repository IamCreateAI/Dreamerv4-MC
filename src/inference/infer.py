import torch
from typing import List, Tuple, Union
from src.utils.utils import CUDAGraphRunner
import math
from src.modules.tokenizer import CausalTokenizer





def generate_one_frame(
    model: torch.nn.Module,
    x: torch.Tensor,
    frame_idx: int,
    steps_size: int,
    action_ids: torch.Tensor,
    device: torch.device,
    K_samples_max = 64,
    refresh_kvcache: bool = False,
    model_runner: Union[None, CUDAGraphRunner] = None,
):
    noise = x.clone()
    

    d_discrete = [1 / 2 ** i for i in range(int(math.log2(K_samples_max)) + 1)]
    d_idx = math.log2(steps_size)
    d = d_discrete[int(d_idx)]
    t = 0
    read_rope_indices = model.get_read_rope_indices(frame_idx)
    frame_idx = torch.tensor([frame_idx], device=device, dtype=torch.long)
    for step in range(steps_size):        
        timestep = torch.tensor([int(t * K_samples_max)], device=device, dtype=torch.long)
        timestep_stride = torch.tensor([int(d_idx)], device=device, dtype=torch.long)
        model_input = {
            'img': x,
            'timestep': timestep,
            'timestep_stride': timestep_stride,
            'action_ids': action_ids,
            "current_frame_idx": frame_idx, 
            "read_rope_indices": read_rope_indices
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.no_grad():
            if model_runner is not None:
                x_hat = model_runner(model_input)
            else:
                x_hat = model(**model_input)
        #v = (x_hat - x) / (1 - t)
        #x = x + v * d
        t = t + d
        x = x_hat * t + noise * (1 - t)
    

    if refresh_kvcache:
        t = 0.875
        x_ = x * t + noise * (1 - t)
        #d_idx = math.log2(K_samples_max)
        timestep = torch.tensor([int(K_samples_max * t)], device=device, dtype=torch.long)
        timestep_stride = torch.tensor([d_idx+1], device=device, dtype=torch.long)
        model_input = {
            'img': x_,
            'timestep': timestep,
            'timestep_stride': timestep_stride,
            'action_ids': action_ids,
            "current_frame_idx": frame_idx, 
            "read_rope_indices": read_rope_indices
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.no_grad():
            if model_runner is not None:
                x = model_runner(model_input)
            else:
                x = model(**model_input)
    
    return x




def decode_one_frame(
    model: CausalTokenizer,
    tokens: torch.Tensor,
    shape: Tuple[int, int, int, int, int],
    frame_idx: int,
    device: torch.device,
):
    model_input = {
        'tokens': tokens,
        "shape": shape,
        "current_frame_idx": torch.tensor([frame_idx], device=device, dtype=torch.long),
    }
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        x_decoded = model.ar_decode(**model_input)
    return x_decoded
    


