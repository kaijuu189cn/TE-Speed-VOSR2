# TE-Speed-VOSR2

VOSR2 图片与视频超分辨率放大 ComfyUI 节点。

- **Windows**: 使用编译的 .pyd（原始 C++ 实现，多加速策略）
- **Linux**: 使用纯 Python `nodes_linux.py`（等效算法，自动回退）

两种实现注册相同的节点 ID（TESpeedVOSR2Loader、TESpeedVOSR2Settings、TESpeedVOSR2Image、TESpeedVOSR2Video），工作流无需改线即可跨平台使用。

模型下载：
https://pan.quark.cn/s/88fc2eeced6f

## 功能

- 支持 VOSR 2.0 1.4B one-step 模型。
- 支持图片原尺寸修复与整数倍放大。
- 支持视频 IMAGE 帧批次放大。
- 支持 `manual` 与 `speed` 两种推理配置。
- 支持空间分块和VAE分块。
- 支持可选的视频 DINO 时序缓存。
- 自动使用 ComfyUI 的 SageAttention，未安装时回退到 SDPA。
- 支持可选 `torch.compile`，运行失败时自动回退到 eager 模式。

## 跨平台说明

| 文件 | 平台 | 说明 |
|---|---|---|
| `nodes.pyd` + `settings.pyd` + `inference.pyd` + `model_store.pyd` | Windows | 原始编译实现 |
| `nodes_linux.py` | Linux | 纯 Python 兼容壳，复用 `backend/` 中的模型代码 |
| `__init__.py` | 自动 | 自动检测平台，优先尝试 .pyd，失败则回退到 .py |

## 环境要求

- Windows x64 或 Linux x64 (ROCm/CUDA)
- ComfyUI
- GPU（NVIDIA / AMD）
- Python 3.12 或 Python 3.13 环境
- PyTorch 与 CUDA 版本需要和当前 ComfyUI 环境兼容
- 可选：SageAttention；ComfyUI 能正常识别即可，无需在节点中单独选择
- 可选：Triton；只在启用 `torch_compile` 时需要



## 模型目录

模型放在：

```text
ComfyUI/models/vosr2/VOSR2/
```

目录结构：

```text
ComfyUI/models/vosr2/VOSR2/
|-- args.json
|-- checkpoints/
|   `-- ema_model.safetensors
|-- Qwen-Image-vae-2d/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
`-- dinov2_vitl14.safetensors
```

DINOv2 权重也兼容以下文件名：

```text
dinov2_vitl14_pretrain.pth
```



## 显存不足或速度异常

按以下顺序尝试：

1. 保持 `tile_size=512`，不要先提高到 1024。
2. 保持 `vae_tile_size=1024`；图片节点会自动限制到 1024。
3. 将 `image_batch` 设为 1，或将视频 `frame_batch` 降到 1。
4. 将 `quality_profile` 改为 `manual`。
5. 将 `memory_policy` 改为 `staged`。
6. 关闭 `torch_compile`，尤其是出现 CUDA Graph 或 `cudaMallocAsync` 警告时。
7. 降低输出倍率或先缩小输入尺寸。


如果 `speed` 比 `manual` 慢，通常是更大的 tile batch 或视频 VAE tile 提高了显存峰值，导致模型换载或 Windows WDDM 使用共享显存。此时 `manual + tile_size 512 + vae_tile_size 1024` 通常更稳定。

