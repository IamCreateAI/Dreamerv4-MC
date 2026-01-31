# Dreamer-MC: A Real-Time Autoregressive World Model for Infinite Video Generation

<div align="center">

<br>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-orange)](https://pytorch.org/)
[![Blog](https://img.shields.io/badge/Project-Blog-blue)](https://findlamp.github.io/dreamer-mc.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Zoo-yellow)](https://huggingface.co/IamCreateAI/Dreamerv4-MC)

[**📖 Introduction**](#-introduction) | [**🏰 Model Zoo**](#-model-zoo) | [**🛠️ Installation**](#-installation) | [**💻 Quick Start**](#-quick-start)

</div>

---

## 📖 Introduction

This repository contains the **Inference Code** for the Minecraft Autoregressive World Model.
This project serves as an **open-source reproduction of the [DreamerV4](https://danijar.com/project/dreamerv4/) architecture**, tailored specifically for high-fidelity simulation in the Minecraft environment. Our model utilizes a **MAE (Masked Autoencoder)** for efficient video compression and a **DiT (Diffusion Transformer)** architecture to autoregressively predict future game frames based on history and action inputs in the latent space.
 This codebase is streamlined for **deployment and generation**, supporting long-context inference and real-time interaction.

### Key Features
* **Inference Only**: Lightweight codebase focused on generation, stripped of complex training logic.
* **Long Context Support**: Capable of loading Long-Context models to recall events from 12 seconds prior.
* **Fast Inference Backend**: Built-in optimized inference pipeline designed for high-performance, real-time next-frame prediction.
* **Infinite Generation**: Supports infinite generation without image quality degradation during long-term rollouts.
* **Complex Interaction**: Supports a variety of interactions within the Minecraft world, such as eating food, collecting water, using weapons, etc.

## 🏰 Model Zoo

Please download the pre-trained weights and place them in the `checkpoints/` directory before running the code.

| Model Name | Params | VRAM Req | Description |
| :--- | :---: | :---: | :--- |
| **MAE-Tokenizer** | 430M | >2GB | Handles video encoding and decoding. |
| **Dynamic Model** | 1.7B | 9GB | Generates the next frame based on history and action. |

> 🔗 **Download**: [HuggingFace Collection](https://huggingface.co/IamCreateAI/Dreamerv4-MC)

## 🛠️ Installation

We recommend using Python 3.10+ and CUDA 12.1+.

```bash
# 1. Clone the repository
git clone https://github.com/IamCreateAI/Dreamerv4-MC.git
cd Dreamerv4-MC

# 2. Create a virtual environment
conda create -n dreamer python=3.12 -y
conda activate dreamer

# 3. Install PyTorch (Adjust index-url for your CUDA version)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install dependencies
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install -e .
```
## 💻 Quick-Start
```bash
python ui/inference_ui.py --dynamic_path=/path/to/dynamic_model \
 --tokenizer_path=/path/to/tokenizer/ \
 --record_video_output_path=output/
```
## 🏋️ Training Code
Coming Soon


## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **W / A / S / D** | Move |
| **Space** | Jump |
| **Left Click** | Attack / Destroy |
| **Right Click** | Place / Use Item |
| **E** | Open/Close Inventory (Simulation) |
| **1 - 9** | Select Hotbar Slot |
| **R** | start/stop record the video|
| **V** | refresh into new scene |
| **left Shift** |Sneak|
| **left ctrl** |Sprint|


## 📜 Citation
If you use this codebase in your research, please consider citing us as:
```bash
@article{hafner2025dreamerv4,
    title   = {Dreamer-MC: A Real-Time Autoregressive World Model for Infinite Video Generation},
    author  = {Ming Gao, Yan Yan, ShengQu Xi, Yu Duan, ShengQian Li, Feng Wang},
    year    = {2026},
    url     = {https://findlamp.github.io/dreamer-mc.github.io/}
}
```
as well as the original Dreamer 4 paper:
```bash
   @misc{Hafner2025TrainingAgents,
    title={Training Agents Inside of Scalable World Models}, 
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2509.24527}, 
}
```


## 📚 References
This project is built upon the following foundational works:
* **MaeTok**: [Masked Autoencoders Are Effective Tokenizers for Diffusion Models](https://arxiv.org/abs/2502.03444) (Chen et al., ICML 2025)
* **DreamerV4**: [Training Agents Inside of Scalable World Models](https://danijar.com/project/dreamerv4/) (Hafner et al., 2025)
* **CausVid**: [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772) (Yin et al., CVPR 2025)