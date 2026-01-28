import torch
from src.inference.infer import decode_one_frame
import math
from typing import List, Tuple, Union
from src.utils.utils import CUDAGraphRunner


def prefilling_dynamic(
    tokens: torch.Tensor,
    dynamic_model,
    K_samples_max = 64,
    device: torch.device = torch.device("cuda"),
    action_ids = None,
    frame_idx = 0,
    model_runner: Union[None, CUDAGraphRunner] = None,
):
    read_rope_indices = dynamic_model.get_read_rope_indices(frame_idx)
    frame_idx = torch.tensor([frame_idx], device=device, dtype=torch.long)
    d_discrete = [1 / 2 ** i for i in range(int(math.log2(K_samples_max)) + 1)]
    d_idx = math.log2(4)
    noise = torch.randn_like(tokens)
    t = 0.75
    x_ = tokens * t + noise * (1 - t)
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
            x = dynamic_model(**model_input)
            
    
def prefilling(
    video: torch.Tensor,
    action_ids: torch.Tensor,   
    tokenizer,
    dynamic_model,
    dynamic_runner: Union[None, CUDAGraphRunner] = None,
) -> torch.Tensor:
    """
    Encode video into tokens using the provided tokenizer.

    Args:
        video (torch.Tensor): Input video tensor of shape (F, C, H, W).
        tokenizer: Tokenizer model with an encode method.
        batch_size (int): Batch size for processing.

    Returns:
        torch.Tensor: Encoded tokens of shape (B, F, L).
    """
    
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        token = tokenizer.encode(video)
   
    
    frame_num = video.shape[1]
    
   
    for i in range(frame_num):
        prefilling_dynamic(
            tokens = token[:,i:i+1],
            dynamic_model = dynamic_model,
            device=dynamic_model.device,
            frame_idx=i,
            model_runner=dynamic_runner,
            action_ids=action_ids,
        )

        
    
    
    for i in range(frame_num):
        decode_one_frame(
            model = tokenizer,
            tokens = token[i:i+1],
            shape=(1, 1, 3, 384 // 16, 640 // 16),
            frame_idx=i,
            device=tokenizer.device,
        )



