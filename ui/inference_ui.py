import asyncio
import json
import time
import io
import threading
import os
import argparse
import sys
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Dict, Set, Optional, Tuple, Callable, Any

import torch
import numpy as np
import av
from PIL import Image
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 引入你的项目依赖
from src.inference.mc_vw_infer import MCWorldModelInfer
from src.modules.actokenizer import MineCraftActionTokenizer

# ==========================================
# 1. 配置 (支持 环境变量 和 命令行参数)
# ==========================================
@dataclass
class ServerConfig:
    # 优先读取环境变量，如果没有则使用默认路径
    dynamic_model_path: str = os.getenv("DYNAMIC_PATH", "")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "")
    record_video_output_path: str = os.getenv("RECORD_VIDEO_OUTPUT_PATH", "")
    
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    steps_size: int = 4
    seed: int = 42
    target_fps: float = 20.0
    camera_scaler: float = 360.0 / 2400.0 * 2
    
    jpeg_quality: int = 60
    resize_width: int = 640
    resize_height: int = 360

# 初始化全局配置
config = ServerConfig()

# ... [保留 KEYWORD_BUTTON_MAPPING, HOTBAR_KEY_MAPPING 代码不变] ...
KEYWORD_BUTTON_MAPPING = {
    "F1": "key.keyboard.escape", "KeyS": "key.keyboard.s", "KeyW": "key.keyboard.w",
    "KeyA": "key.keyboard.a", "KeyD": "key.keyboard.d", "KeyE": "key.keyboard.e",
    "KeyQ": "key.keyboard.q", "Digit1": "key.keyboard.1", "Digit2": "key.keyboard.2",
    "Digit3": "key.keyboard.3", "Digit4": "key.keyboard.4", "Digit5": "key.keyboard.5",
    "Digit6": "key.keyboard.6", "Digit7": "key.keyboard.7", "Digit8": "key.keyboard.8",
    "Digit9": "key.keyboard.9", "Space": "key.keyboard.space", "ShiftLeft": "key.keyboard.left.shift",
    "ControlLeft": "key.keyboard.left.control", "KeyF": "key.keyboard.f"
}

HOTBAR_KEY_MAPPING = {
    "key.keyboard.1": 0, "key.keyboard.2": 1, "key.keyboard.3": 2,
    "key.keyboard.4": 3, "key.keyboard.5": 4, "key.keyboard.6": 5,
    "key.keyboard.7": 6, "key.keyboard.8": 7, "key.keyboard.9": 8
}

mouse_button_mapping = {
    0: 0,
    1: 2,
    2: 1
}

# ... [保留 InputState, InputStateManager, LatestFrameContainer 代码不变] ...
@dataclass
class InputState:
    keys_down: Set[str] = field(default_factory=set)
    mouse_buttons: Set[str] = field(default_factory=set)
    frame_dx: float = 0.0
    frame_dy: float = 0.0
    seq: int = 0
    frame_id: int = 0
    current_hotbar_idx: int = 0
    wheel: float = 0.0

class InputStateManager:
    def __init__(self):
        self.state = InputState()
        self.lock = threading.Lock()

    def process_event(self, data: dict):
        ev_type = data.get("event")
        if ev_type == "keydown":
            #self.state.keys_down.add(data.get("key", ""))
            raw_key = data.get("key", "")
            self.state.keys_down.add(raw_key)
            mc_key_name = KEYWORD_BUTTON_MAPPING.get(raw_key)
            if mc_key_name and mc_key_name in HOTBAR_KEY_MAPPING:
                self.state.current_hotbar_idx = HOTBAR_KEY_MAPPING[mc_key_name]

            return raw_key
        elif ev_type == "keyup":
            self.state.keys_down.discard(data.get("key", ""))
        elif ev_type == "mousemove":
            self.state.frame_dx += float(data.get("dx", 0.0))
            self.state.frame_dy += float(data.get("dy", 0.0))
        elif ev_type == "mousedown":
            self.state.mouse_buttons.add(str(data.get("button", "")))
        elif ev_type == "mouseup":
            self.state.mouse_buttons.discard(str(data.get("button", "")))
        return None

    def get_snapshot_and_reset(self):
        with self.lock:
            dx = self.state.frame_dx
            dy = self.state.frame_dy
            self.state.frame_dx = 0
            self.state.frame_dy = 0
            self.state.frame_id += 1
            
            state_snapshot = InputState(
                keys_down=self.state.keys_down.copy(),
                mouse_buttons=self.state.mouse_buttons.copy(),
                frame_dx=dx,
                frame_dy=dy,
                seq=self.state.seq,
                frame_id=self.state.frame_id,
                current_hotbar_idx=self.state.current_hotbar_idx
            )
            return state_snapshot, dx, dy

class LatestFrameContainer:
    def __init__(self):
        self.frame_bytes: Optional[bytes] = None
        self.meta: Optional[dict] = None
        self.condition = asyncio.Condition()
        self.updated = False

    async def put(self, frame_bytes, meta):
        async with self.condition:
            self.frame_bytes = frame_bytes
            self.meta = meta
            self.updated = True
            self.condition.notify()

    async def get(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.updated)
            self.updated = False
            return self.frame_bytes, self.meta

# ==========================================
# 3. 推理引擎 (使用全局 config)
# ==========================================
class InferenceEngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = MineCraftActionTokenizer()
        self.tokenizer.camera_scaler = config.camera_scaler
        self.lock = threading.Lock()

    def load_model(self):
        # 这里的 config.dynamic_model_path 已经被 argparse 或 env 更新
        print(f"Loading Model from: {self.config.dynamic_model_path}")
        print(f"Loading Tokenizer from: {self.config.tokenizer_path}")
        
        self.model = MCWorldModelInfer(
            dynamic_model_path=self.config.dynamic_model_path,
            tokenizer_path=self.config.tokenizer_path,
            record_video_output_path=self.config.record_video_output_path,
            steps_size=self.config.steps_size,
            device=torch.device(self.config.device),
            dtype=self.config.dtype,
            random_generator=torch.Generator(device=self.config.device).manual_seed(self.config.seed),
            use_cuda_graph=True,
            refresh_kvcache=False,
        )
        print("Model Loaded.")

    # ... [保留 reset_kv_cache 和 render 代码不变] ...
    def reset_kv_cache(self):
        self.model.clean_kvcache()

    def render(self, state: InputState, dx: float, dy: float) -> bytes:
        if not self.model: return b''
        
        pressed_keys = [KEYWORD_BUTTON_MAPPING[k] for k in state.keys_down if k in KEYWORD_BUTTON_MAPPING]
        pressed_mouse_btns = [mouse_button_mapping[int(k)] for k in state.mouse_buttons]
        
        #pressed_hotbar = [HOTBAR_KEY_MAPPING[k] for k in pressed_keys if k in HOTBAR_KEY_MAPPING]
        #if pressed_hotbar: state.current_hotbar_idx = pressed_hotbar[0]

        action_dict = {
            "mouse": {"dx": dx, "dy": dy, "buttons": pressed_mouse_btns},
            "keyboard": {"keys": pressed_keys},
            "hotbar": state.current_hotbar_idx,
        }
        
        env_action, _ = self.tokenizer.json_action_to_env_action(action_dict, hotbar=True)
        action_idx_seq = self.tokenizer.get_action_index_from_actiondict(env_action, include_gui=True)
        action_tensor = torch.tensor(action_idx_seq).to(self.model.device)

        frame = self.model.render_next_frame(action_id=action_tensor)

        img_tensor = ((frame.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(img_array)
        
        if self.config.resize_width != 640:
            image = image.resize((self.config.resize_width, self.config.resize_height))

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.config.jpeg_quality, optimize=True)
        return buf.getvalue()
    
    def record_video(self):
        if self.model:
            with self.lock:
                self.model.record_video()

# ==========================================
# 4. 初始化
# ==========================================

# 1. 创建引擎实例
engine = InferenceEngine(config)

# 2. 定义生命周期 (启动时加载模型)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 这里会读取最新的 config (包括 argparse 修改后的)
    engine.load_model()
    yield
from fastapi.staticfiles import StaticFiles
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
static_dir_path = BASE_DIR / "static"
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")
#app.mount("static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse(static_dir_path / "index.html")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    
    state_manager = InputStateManager()
    latest_frame = LatestFrameContainer()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def recv_loop():
        try:
            while not stop_event.is_set():
                msg = await ws.receive_text()
                data = json.loads(msg)
                if data.get("type") == "event":
                    with state_manager.lock:
                        key = state_manager.process_event(data)
                    if key == "KeyV":
                        engine.reset_kv_cache()
                        state_manager.state.frame_id = 0
                    if key == "KeyR":
                        engine.record_video()
        except:
            stop_event.set()

    async def render_loop():
        target_interval = 1.0 / config.target_fps
        next_ts = time.perf_counter()
        
        try:
            while not stop_event.is_set():
                now = time.perf_counter()
                if now < next_ts:
                    await asyncio.sleep(next_ts - now)
                next_ts += target_interval

                state_snapshot, dx, dy = state_manager.get_snapshot_and_reset()
                
                # 在线程池中执行
                jpeg_bytes = await loop.run_in_executor(
                    None, 
                    engine.render, 
                    state_snapshot, 
                    dx, 
                    dy
                )

                meta = {
                    "type": "meta",
                    "dx": dx, "dy": dy, 
                    "seq": state_snapshot.seq, 
                    "frame_id": state_snapshot.frame_id
                }
                await latest_frame.put(jpeg_bytes, meta)
        except Exception as e:
            print(f"Render Error: {e}")
            stop_event.set()

    async def send_loop():
        try:
            while not stop_event.is_set():
                frame_bytes, meta = await latest_frame.get()
                await ws.send_text(json.dumps(meta))
                await ws.send_bytes(frame_bytes)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Send Error: {e}")
            stop_event.set()

    t1 = asyncio.create_task(recv_loop())
    t2 = asyncio.create_task(render_loop())
    t3 = asyncio.create_task(send_loop())
    
    await stop_event.wait()
    t1.cancel()
    t2.cancel()
    t3.cancel()

# ==========================================
# 5. 命令行启动入口 (新增)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="MC World Model Inference Server")
    parser.add_argument("--dynamic_path", type=str, help="Path to dynamic model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer checkpoint")
    parser.add_argument("--record_video_output_path", type=str, help="Path to save recorded videos")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    # 覆盖默认配置
    if args.dynamic_path:
        config.dynamic_model_path = args.dynamic_path
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
        
    if args.record_video_output_path:
        config.record_video_output_path = args.record_video_output_path
        
    print(f"Starting server on {args.host}:{args.port}")
    
    # 直接运行 uvicorn
    uvicorn.run(app, host=args.host, port=args.port)