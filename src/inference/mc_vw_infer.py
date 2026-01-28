import torch
from typing import List
from src.inference.infer import generate_one_frame, decode_one_frame
import os, json
from src.inference.prefill import prefilling
from src.utils.utils import remove_unnecessary_key, CUDAGraphRunner, save_video_tensor
from tqdm import tqdm

    

class MCWorldModelInfer:
    def __init__(
        self,
        dynamic_model_path: str,
        tokenizer_path: str,
        record_video_output_path: str,
        steps_size: int,   
        device: torch.device,
        dtype: torch.dtype,
        random_generator: torch.Generator,
        use_cuda_graph: bool = False,
        refresh_kvcache: bool = False,
    ):  
        
        self.use_safetensor = True
        self.init_tokenizer(
            tokenizer_path=tokenizer_path,
            device=device,
        )
        self.init_dynamic_model(
            dynamic_model_path=dynamic_model_path,
            device=device,
        )
        self.device=device
        self.use_cuda_graph = use_cuda_graph
        self.random_generator = random_generator
        self.steps_size = steps_size      
        self.init_tokenizer_cuda_graph()
        self.init_dynamic_model_cuda_graph()
        self.frame_idx = 0
        self.enable_record_video = False
        self.cached_video = []
        self.max_cached_video = 2048
        self.current_chunk_idx = 0
        self.refresh_kvcache = refresh_kvcache
        self.record_video_output_path = record_video_output_path
        self.defaults_action_ids = torch.tensor([[65, 33, 55, 66,  66,  66,  66,  66, 66, 66, 66, 67]], device=device, dtype= torch.long)
    def init_tokenizer(
        self,
        tokenizer_path: str,
        device: torch.device,
    ):
        from src.modules.tokenizer import CausalTokenizer
        
        

        self.tokenizer = CausalTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = self.tokenizer.to(device)

        self.tokenizer.eval()
    def init_dynamic_model(
        self,
        dynamic_model_path: str,
        device: torch.device,
    ):
        from src.modules.dynamic_model import DynamicModel

        self.model = DynamicModel.from_pretrained(dynamic_model_path)
        self.model = self.model.to(device)
        self.model.eval()
       
        
    
    def init_tokenizer_cuda_graph(self):
        
        frame_num = self.tokenizer.window_size
        patch_size = self.tokenizer.patch_size
        decode_shape = (1, frame_num, 3,384 // patch_size[0], 640 // patch_size[0])
        
        self.tokenizer.decoder.init_rope_cache(decode_shape)
        token = torch.randn(1, 1, 256, 32, device="cuda", dtype=torch.bfloat16)
        index_tensor = torch.tensor([0], dtype=torch.long, device='cuda')
        sample_input = {
            "z": token.reshape(1, 1, 256 *2, 32 // 2),
            "shape": (1, 1, 3, 384 // patch_size[0], 640 // patch_size[1]),
            "current_frame_idx": index_tensor,
            "read_rope_indices": torch.arange(frame_num, device="cuda", dtype=torch.long)
        }
        if self.use_cuda_graph:
            self.tokenizer.record_decoder_graph(sample_input, autocast_kwargs={"dtype": torch.bfloat16, "cache_enabled": False})
        
    def init_dynamic_model_cuda_graph(
        self,
    ):
        self.model.init_rope_cache(frame_seqlen=256, action_slot_len=12)
        if self.use_cuda_graph:
            from src.utils.utils import CUDAGraphRunner
            sample_input = {
                'img': torch.randn((1, 1, 256, 32), device="cuda", dtype=torch.bfloat16),
                'timestep': torch.tensor([0], device="cuda", dtype=torch.long),
                'timestep_stride': torch.tensor([0], device="cuda", dtype=torch.long),
                'action_ids': torch.arange(12, device="cuda", dtype=torch.long)[None],
                "current_frame_idx": torch.tensor([0], device="cuda", dtype=torch.long),
                "read_rope_indices": torch.arange(self.model.window_size, device="cuda", dtype=torch.long)
            }
            self.model_runner = CUDAGraphRunner(self.model, sample_input, 
                                        autocast_kwargs={"dtype": torch.bfloat16, "cache_enabled": False})
            self.model_runner.capture()
            self.model.clean_kvcache()
    
    
    def infer_video(
        self,
        action_ids: torch.Tensor,
        init_frames: torch.Tensor,
    ):

        output_video = []
        if init_frames is not None:
            self.prefilling_kvcache(
                init_frames=init_frames,
                action_ids=action_ids[:init_frames.shape[1]],
            )
            output_video.append(init_frames[0])
        generate_frame_num = action_ids.shape[0]
        output_video = []
        for frame_idx in tqdm(range(generate_frame_num)):
            noise = torch.randn(
                (1, 1, 256, 32), 
                device=self.model.device,
                dtype=torch.bfloat16,
                generator=self.random_generator,
            )
            action_id = action_ids[frame_idx:frame_idx+1]
           
            
            token = generate_one_frame(
                model=self.model,
                x=noise,
                frame_idx=frame_idx,
                steps_size=self.steps_size,
                action_ids=action_id,
                device=self.model.device,
                K_samples_max=self.model.K_samples_step,
                model_runner=self.model_runner if self.use_cuda_graph else None,
            )
            frame = decode_one_frame(
                model=self.tokenizer,
                tokens=token.clone(),
                shape=(1, 1, 3, 384 // 16, 640 // 16),
                frame_idx=frame_idx,
                device=self.model.device,
            )
            output_video.append(frame[0])
        output_video = torch.cat(output_video, dim=0)
        return output_video
    
    def prefilling_kvcache(
        self,
        init_frames: torch.Tensor,
        action_ids: torch.Tensor = None,
    ):
        
        if action_ids is None:
            action_ids = self.defaults_action_ids.repeat(init_frames.shape[1], 1)
        
      
        prefilling(
            video=init_frames,
            action_ids=action_ids,
            tokenizer=self.tokenizer,
            dynamic_model=self.model,
            dynamic_runner=self.model_runner if self.use_cuda_graph else None,
        )
        self.frame_idx = init_frames.shape[1]
        
    
    
    def render_next_frame(
        self,
        action_id: torch.Tensor,
    ):

        
        noise = torch.randn(
            (1, 1, 256, 32), 
            device=self.model.device,
            dtype=torch.bfloat16,
            generator=self.random_generator,
        )
        token = generate_one_frame(
            model=self.model,
            x=noise,
            frame_idx=self.frame_idx,
            steps_size=4,
            action_ids=action_id,
            device=self.model.device,
            K_samples_max=self.model.K_samples_step,
            model_runner=self.model_runner if self.use_cuda_graph else None,
            refresh_kvcache=self.refresh_kvcache
        )
        frame = decode_one_frame(
            model=self.tokenizer,
            tokens=token.clone(),
            shape=(1, 1, 3, 384 // 16, 640 // 16),
            frame_idx=self.frame_idx,
            device=self.model.device,
        )
        self.frame_idx +=1
        
        if self.enable_record_video:
            if len(self.cached_video) >= self.max_cached_video:
                video_tensor = torch.stack(self.cached_video, dim=0)
                save_video_tensor(video_tensor, f"{self.record_video_output_path}/{self.current_chunk_idx}.mp4", fps=20)
                self.current_chunk_idx += 1
                self.cached_video = []
                print("Saved video chunk due to max cache size.")
            self.cached_video.append(frame[0, 0][:, 12:372])
        
        return frame[0, 0][:, 12:372]

    def clean_kvcache(self):
        self.model.clean_kvcache()
        self.tokenizer.decoder.clean_kvcache()
        self.frame_idx = 0
    
    def record_video(self):
        
        if self.enable_record_video:
            self.enable_record_video = False
            print("Stopped recording and saved video.")
            video_tensor = torch.stack(self.cached_video, dim=0)
            try:
                save_video_tensor(video_tensor, f"{self.record_video_output_path}/{self.current_chunk_idx}.mp4", fps=20)
            except Exception as e:
                print(f"{self.record_video_output_path}/{self.current_chunk_idx}.mp4")
                print(f"Error saving video: {e}")
            self.current_chunk_idx += 1
            self.cached_video = []
        else:
            self.enable_record_video = True
            self.cached_video = []
            print("Started recording video.")
    
    
        