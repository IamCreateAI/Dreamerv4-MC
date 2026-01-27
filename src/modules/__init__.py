

from src.modules.tokenizer import CausalTokenizer
from src.modules.dynamic_model import DynamicModel
import torch

MODEL_REGISTRY = {
    "CAUSAL_TOKENIZER": CausalTokenizer,
    "CAUSAL_DYMAMIC_MODEL": DynamicModel,
}

def register_model(name, model_class):
    MODEL_REGISTRY[name] = model_class

def get_model(name, device="cuda", dtype=torch.float32, **kwargs):
    if name in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY[name]
        return model_class(**kwargs, device=device, dtype=dtype)
    else:
        raise ValueError(f"Model {name} not found in registry!")

def get_model_class(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        raise ValueError(f"Model {name} not found in registry!")