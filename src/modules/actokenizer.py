from typing import Any, Optional, List, Tuple, Union, Sequence, Dict
from dataclasses import dataclass, field
import collections, os, json
import numpy as np
from typing import Dict
import attr

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),  # [y, x]
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}
# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0



class Buttons:
    # 14 in total without hotbar and camera
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    INVENTORY = "inventory"
    # added by Yang
    ESC = "ESC"
    SWAPHANDS = "swapHands"
    PICKITEM = "pickItem"

    ALL = [
        USE,
        ATTACK,
        FORWARD,
        BACK,
        LEFT,
        RIGHT,
        JUMP,
        SNEAK,
        SPRINT,
        DROP,
        SWAPHANDS,
        PICKITEM,
        INVENTORY,
        ESC,
    ] + [f"hotbar.{i}" for i in range(1, 10)]



class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L49
@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy

@dataclass(slots=True)
class MineCraftActionTokenizer:
    # --- tokenizer形参 ---
    action_length: int = 12                   # 含 bos/eos
    camera_binsize: int = 9                    # vpt=2（这里保留你的默认）
    camera_maxval: int = 90                    # vpt=10（这里保留你的默认）
    camera_mu: float = 11.4887                 # vpt=10（这里保留你的默认）
    quantization_scheme: str = "mu_law"
    camera_scaler: float = 360.0 / 2400.0      # 替代全局 CAMERA_SCALER

    # --- 依赖可注入，避免硬编码全局 ---
    buttons_all: Optional[Sequence[str]] = None
    keyboard_mapping: Optional[Dict[str, str]] = None
    noop_action_template: Optional[Dict[str, Union[int, np.ndarray]]] = None

    # --- 运行期对象 ---
    action_vocab: Dict[str, int] = field(default_factory=dict, init=False)
    camera_quantizer: Any = field(default=None, init=False)

    def __post_init__(self):
        # 初始化相机量化器
        self.camera_quantizer = CameraQuantizer(
            camera_binsize=self.camera_binsize,
            camera_maxval=self.camera_maxval,
            mu=self.camera_mu,
            quantization_scheme=self.quantization_scheme,
        )
        # 尝试从全局拉取缺省依赖
        if self.buttons_all is None:
            try:
                self.buttons_all = list(Buttons.ALL)  # type: ignore[name-defined]
            except NameError:
                raise ValueError("buttons_all 未提供，且未找到全局 Buttons.ALL")
        if self.keyboard_mapping is None:
            try:
                self.keyboard_mapping = dict(KEYBOARD_BUTTON_MAPPING)  # type: ignore[name-defined]
            except NameError:
                raise ValueError("keyboard_mapping 未提供，且未找到全局 KEYBOARD_BUTTON_MAPPING")
        if self.noop_action_template is None:
            try:
                self.noop_action_template = dict(NOOP_ACTION)  # type: ignore[name-defined]
            except NameError:
                raise ValueError("noop_action_template 未提供，且未找到全局 NOOP_ACTION")

    # ----------------- 基础工具 -----------------
    def _noop(self) -> Dict[str, Union[int, np.ndarray]]:
        """返回一个可写的 NOOP 动作副本，并确保 camera 字段存在。"""
        env_action = {k: (v.copy() if isinstance(v, np.ndarray) else v) 
                      for k, v in self.noop_action_template.items()}
        env_action["camera"] = np.array([0.0, 0.0], dtype=np.float32)
        return env_action

    @staticmethod
    def _both_zero_out(env_action: Dict[str, Union[int, np.ndarray]], k1: str, k2: str):
        if env_action.get(k1, 0) == 1 and env_action.get(k2, 0) == 1:
            env_action[k1] = 0
            env_action[k2] = 0

    def _pair_index(self,
                    action_dict: Dict[str, Union[int, np.ndarray]],
                    k1: str,
                    k2: str,
                    null_key: str) -> int:
        """返回冲突二元动作在 vocab 中的索引。"""
        v1 = int(action_dict.get(k1, 0))
        v2 = int(action_dict.get(k2, 0))
        if v1 and v2:
            return self.action_vocab[null_key]
        if v1:
            return self.action_vocab[k1]
        if v2:
            return self.action_vocab[k2]
        return self.action_vocab[null_key]

    # ----------------- 词表 -----------------
    def make_action_vocab(self,
                          num_cam_bins: int = 21,
                          action_vocab_offset: int = 0,
                          verbose: bool = False):
        """构建并缓存 token 词表（只构建一次）。"""
        if self.action_vocab:
            return  # 已构建

        action_vocab = collections.OrderedDict()
        # 基础按键（14个+ hotbar.1~9，随 Buttons.ALL）
        for i, action in enumerate(self.buttons_all):
            action_vocab[action] = i

        base = len(self.buttons_all)
        for i in range(num_cam_bins):
            action_vocab[f"cam_0_{i}"] = base + i
        for i in range(num_cam_bins):
            action_vocab[f"cam_1_{i}"] = base + num_cam_bins + i

        action_vocab["<act_bos>"]  = base + 2 * num_cam_bins
        action_vocab["<null_act>"] = base + 2 * num_cam_bins + 1
        action_vocab["<act_eos>"]  = base + 2 * num_cam_bins + 2

        if action_vocab_offset:
            action_vocab = {k: v + action_vocab_offset for k, v in action_vocab.items()}

        if verbose:
            print(f"[MCDataset] Action Vocab size={len(action_vocab)}")

        self.action_vocab = action_vocab

    # ----------------- JSON动作 -> 环境动作 -----------------
    def json_action_to_env_action(self, json_action: Dict[str, Any], hotbar=False) -> tuple[Dict[str, Union[int, np.ndarray]], bool]:
        """将日志JSON动作转换为 MineRL 环境动作，并返回 (env_action, is_null_action)。"""
        env_action = self._noop()
        is_null_action = True

        if hotbar:
            current_hotbar_idx = json_action.get("hotbar", 0) 
            env_action["hotbar_state"] = current_hotbar_idx # 存一个临时字段
        
        # 键盘
        for key in json_action.get("keyboard", {}).get("keys", []):
            mapped = self.keyboard_mapping.get(key)
            if mapped is not None:
                env_action[mapped] = 1
                is_null_action = False

        # 鼠标相机（dy->pitch, dx->yaw）
        mouse = json_action.get("mouse", {})
        dx = float(mouse.get("dx", 0.0))
        dy = float(mouse.get("dy", 0.0))
        cam = env_action["camera"]  # ndarray
        cam[0] = dy * self.camera_scaler
        cam[1] = dx * self.camera_scaler

        if dx != 0.0 or dy != 0.0:
            is_null_action = False
        else:
            # 安全剪裁（与原实现一致）
            if abs(cam[0]) > 180:
                cam[0] = 0.0
            if abs(cam[1]) > 180:
                cam[1] = 0.0

        # 鼠标按键 -> attack/use/pickItem
        btns = mouse.get("buttons", [])
        if 0 in btns:
            env_action["attack"] = 1; is_null_action = False
        if 1 in btns:
            env_action["use"] = 1;    is_null_action = False
        if 2 in btns:
            env_action["pickItem"] = 1; is_null_action = False

        # 互斥冲突消解：同按则都置0（保持与你原逻辑一致）
        self._both_zero_out(env_action, "forward", "back")
        self._both_zero_out(env_action, "left", "right")
        self._both_zero_out(env_action, "jump", "sneak")
        self._both_zero_out(env_action, "sprint", "sneak")
        self._both_zero_out(env_action, "attack", "use")

        # inventory / ESC 视为 null
        #if env_action.get("inventory", 0) == 1: is_null_action = True
        #if env_action.get("ESC", 0) == 1:       is_null_action = True

        return env_action, False

    # ----------------- 环境动作 -> 索引序列 -----------------
    def get_action_index_from_actiondict(self,
                                         action_dict: Dict[str, Union[int, np.ndarray]],
                                         num_cam_bins: int = 21,
                                         action_vocab_offset: int = 0,
                                         verbose: bool = False,
                                         include_gui: bool = False
                                         ) -> List[int]:
        """将 env_action 字典打包为长度固定的 action token 序列。"""
        if not self.action_vocab:
            self.make_action_vocab(num_cam_bins=num_cam_bins,
                                   action_vocab_offset=action_vocab_offset,
                                   verbose=verbose)
        if not include_gui:
            self.action_length = 11
            
        
        null_id = self.action_vocab["<null_act>"]
        seq = [null_id] * self.action_length
        seq[0]  = self.action_vocab["<act_bos>"]
        seq[-1] = self.action_vocab["<act_eos>"]

        # 相机离散化
        camera_action = action_dict.get("camera", np.array([0, 0]))
        if not isinstance(camera_action, np.ndarray):
            camera_action = np.array(camera_action, dtype=np.float32)
        cam_idx = self.camera_quantizer.discretize(camera_action)  # -> [idx0, idx1]
        seq[1] = self.action_vocab[f"cam_0_{int(cam_idx[0])}"]
        seq[2] = self.action_vocab[f"cam_1_{int(cam_idx[1])}"]

        # 热键栏
        if "hotbar_state" in action_dict:
            hotbar_state_idx = int(action_dict.get("hotbar_state", 0)) # 默认为0
        
            # 映射：0 -> hotbar.1, 1 -> hotbar.2 ... 8 -> hotbar.9
            # 确保它不越界 (0-8)
            hotbar_state_idx = max(0, min(8, hotbar_state_idx))
            target_token_str = f"hotbar.{hotbar_state_idx + 1}"
            
            if target_token_str in self.action_vocab:
                seq[3] = self.action_vocab[target_token_str]
            else:
                # 理论上不应该进这里，除非 vocab 没构建好
                seq[3] = null_id
        else:
            seq[3] = null_id
            for i in range(1, 10):
                if int(action_dict.get(f"hotbar.{i}", 0)) == 1:
                    seq[3] = self.action_vocab[f"hotbar.{i}"]
                    break

        # 二元冲突类
        seq[4] = self._pair_index(action_dict, "forward", "back", "<null_act>")
        seq[5] = self._pair_index(action_dict, "left", "right", "<null_act>")
        seq[6] = self._pair_index(action_dict, "sprint", "sneak", "<null_act>")
        seq[7] = self._pair_index(action_dict, "use", "attack", "<null_act>")
        # 单键
        seq[8] = self.action_vocab["jump"] if int(action_dict.get("jump", 0)) == 1 else null_id
        seq[9] = self._pair_index(action_dict, "drop", "pickItem", "<null_act>")

        if include_gui:
            seq[10] = self._pair_index(action_dict, "inventory", "ESC", "<null_act>")
        if verbose:
            print(f"[MCDataset] action idx seq: {seq}")

        return seq

    # ----------------- 读取 JSONL -----------------
    def read_jsonl(self, jsonl_path: str) -> Optional[List[Dict[str, Any]]]:
        """更简单、健壮的 JSONL 读取。失败返回 None。"""
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"[MCDataset] {jsonl_path} does not exist")

        try:
            with open(jsonl_path, "r", encoding="cp1252") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"[MCDataset] {jsonl_path} cannot be read: {e}")
            return None
        
        
    def read_and_encode(self, action_file: str, include_gui: bool = False, hotbar: bool = False) -> List[int]:
        action_json = self.read_jsonl(action_file)
        if action_json is None:
            return []
        action_idx_all = []
        for act in action_json:
            env_action, is_null = self.json_action_to_env_action(act, hotbar=hotbar)
            if not is_null:
                action_idx_seq = self.get_action_index_from_actiondict(env_action, include_gui=include_gui)
                action_idx_all.append(action_idx_seq)
        return action_idx_all
