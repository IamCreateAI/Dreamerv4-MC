# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
from typing import List

import imageio
import torch
import torchvision
import gc
import time



class CUDAGraphRunner:
    def __init__(self, model, sample_input, autocast_kwargs=None):
        """
        model: 你的 PyTorch 模型
        sample_input: 一个真实的输入样本 (dict)，用于确定形状和分配显存
        autocast_kwargs: 你的 autocast 参数 (例如 {'dtype': torch.bfloat16, 'cache_enabled': False})
        """
        self.model = model
        self.stream = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()
        self.autocast_kwargs = autocast_kwargs if autocast_kwargs else {}
        
        # 1. 创建静态输入 Buffer (Static Input Buffers)
        # 我们深拷贝 sample_input 的结构和形状，作为永久驻留显存的静态输入
        self.static_input = {}
        for k, v in sample_input.items():
            if isinstance(v, torch.Tensor):
                # 必须确保静态 buffer 也是在 GPU 上
                self.static_input[k] = v.clone().detach().requires_grad_(False)
            else:
                # 如果有非 Tensor 的 metadata，直接保留 (前提是模型forward里不依赖它的具体值做控制流)
                self.static_input[k] = v

        self.static_output = None
        self.is_captured = False

    def capture(self):
        """执行 Warmup 并录制 CUDA Graph"""
        self.model.eval()
        
        # 1. Warmup (非常重要，用于初始化 cudnn/cublas 等内部 workspace)
        # 建议至少运行 3-11 次，确保没有任何 lazy initialization 遗留
        s = torch.cuda.current_stream()
        s.wait_stream(self.stream)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                with torch.amp.autocast("cuda", **self.autocast_kwargs), torch.no_grad():
                    self.model(**self.static_input)
        s.wait_stream(self.stream)
        torch.cuda.synchronize()

        # 2. Capture (录制)
        print("Starting CUDA Graph capture...")
        with torch.cuda.graph(self.graph):
            with torch.amp.autocast("cuda", **self.autocast_kwargs), torch.no_grad():
                # 这里必须使用静态输入
                # 假设输出是 tuple/list，取第一个元素 (根据你的代码 [0])
                self.static_output = self.model(**self.static_input)
        self.is_captured = True
        print("CUDA Graph captured successfully.")

    def __call__(self, real_input):
        """
        推理时的调用接口
        real_input: 新的动态数据，必须和 sample_input 形状完全一致
        """
        if not self.is_captured:
            # 如果没录制，先跑一次录制 (Lazy init)
            # 注意：第一次调用会慢，因为要 Warmup + Capture
            self.capture()

        # 1. 将新数据 Copy 到静态 Buffer
        # 这是一个 Device-to-Device 的拷贝，非常快
        for k, v in real_input.items():
            if k in self.static_input and isinstance(v, torch.Tensor):
                self.static_input[k].copy_(v)

        # 2. Replay (回放)
        self.graph.replay()

        # 3. 返回结果
        # 注意：返回的是静态输出的引用，如果外部需要修改它，必须 .clone()
        return self.static_output
        





def remove_unnecessary_key(config: dict, keys_to_remove: List[str]) -> dict:
    for key in keys_to_remove:
        if key in config:
            del config[key]
    return config

def save_video_tensor(video_tensor, save_path, fps=24):
    import torchvision.io as io
    video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2  # to [0,1]
    video_tensor = (video_tensor * 255).to(torch.uint8)  # to uint8
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # (T,H,W,C)
    io.write_video(save_path, video_tensor.cpu(), fps=fps)

