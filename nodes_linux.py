"""TESpeedVOSR2* Linux 兼容壳。

注册节点 ID 与 Windows .pyd 原版完全一致（TESpeedVOSR2Loader,
TESpeedVOSR2Settings, TESpeedVOSR2Image, TESpeedVOSR2Video），
工作流无需改线。

底层加载 backend/ 中的模型（DINOv2、VAE、DiT）并调用 vosr2_upscale_one_step
完成实际的上采样。自定义类型映射：

  TE_SPEED_VOSR2_SETTINGS  → dict（settings 参数字典）
  TE_SPEED_VOSR2_MODEL     → dict（bundle，与 VOSR2_BUNDLE 同一结构）
"""

import sys
import os
import logging
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import comfy
import comfy.model_management as mm
import folder_paths

logger = logging.getLogger("TESpeedVOSR2-Linux")

# ── Import vendored models from backend/ ─────────────────────────
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_PLUGIN_DIR, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.dinov2 import build_dinov2_vitl14
from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
from models.lightningdit import LightningDiT
from color import apply_color_alignment

AE_FACTOR = 8  # Qwen VAE spatial compression ratio
PATCH_SIZE = 2

VOSR2_MODEL_DIR = os.path.join(folder_paths.models_dir, "vosr2", "VOSR2")


# ── Loading helpers ──────────────────────────────────────────────

def load_dinov2(device):
    ckpt = os.path.join(VOSR2_MODEL_DIR, "dinov2_vitl14.safetensors")
    if not os.path.exists(ckpt):
        alt = os.path.join(VOSR2_MODEL_DIR, "dinov2_vitl14_pretrain.pth")
        if os.path.exists(alt):
            ckpt = alt
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {ckpt}")
    logger.info(f"  DINOv2: {ckpt}")
    model = build_dinov2_vitl14()
    sd = comfy.utils.load_torch_file(ckpt, safe_load=True)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def load_vae(device):
    vae_dir = os.path.join(VOSR2_MODEL_DIR, "Qwen-Image-vae-2d")
    cfg_path = os.path.join(vae_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"VAE config not found: {cfg_path}")
    import json
    with open(cfg_path) as f:
        vae_config = json.load(f)
    vae = AutoencoderKLQwenImage2D(**vae_config)
    ckpt = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
    sd = comfy.utils.load_torch_file(ckpt, safe_load=True)
    vae.load_state_dict(sd, strict=True)
    vae.to(device).eval()
    return vae


def load_dit(device):
    ckpt_dir = os.path.join(VOSR2_MODEL_DIR, "checkpoints")
    ckpt = os.path.join(ckpt_dir, "ema_model.safetensors")
    args_file = os.path.join(VOSR2_MODEL_DIR, "args.json")
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"DiT args not found: {args_file}")
    import json
    with open(args_file) as f:
        dit_args = json.load(f)

    # Build DiT model
    in_channels = dit_args.get("in_channels", 4)
    model = LightningDiT(
        input_size=None,
        in_channels=in_channels,
        dit_args=dit_args,
    )
    sd = comfy.utils.load_torch_file(ckpt, safe_load=True)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, dit_args


def _nearest_multiple(x, m):
    return int(math.ceil(x / m) * m)


def vosr2_upscale_one_step(image_bchw, scale, dinov2, vae, dit, dit_args, device,
                            color_alignment="wavelet"):
    """Single-step VOSR2 upscale.

    Args:
        image_bchw: (1, C, H, W) float32 tensor on device
        scale: int (1, 2, 3, 4)
    """
    B, C, H, W = image_bchw.shape
    logger.info(f"  Input: {image_bchw.shape}, scale={scale}")

    # 1. Encode to latent
    ae_factor = AE_FACTOR
    h_ae = _nearest_multiple(H, ae_factor)
    w_ae = _nearest_multiple(W, ae_factor)
    img_pad = F.pad(image_bchw, (0, w_ae - W, 0, h_ae - H))
    latents = vae.encode(img_pad).latent_dist.sample()
    logger.info(f"  Latent: {latents.shape}")

    # 2. Upsample latent
    h_lat = latents.shape[2] * scale
    w_lat = latents.shape[3] * scale
    latents = F.interpolate(latents, size=(h_lat, w_lat), mode="bilinear", align_corners=False)

    # 3. DINOv2 features
    img_dino = F.interpolate(image_bchw, size=(224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        feats = dinov2.forward_features(img_dino)
        dino_feat = feats["x_norm_patchtokens"]
    logger.info(f"  DINO features: {dino_feat.shape}")

    # 4. DiT inference
    t = torch.zeros((B,), device=device)
    with torch.no_grad():
        pred = dit(latents, t, encoder_hidden_states=dino_feat)
        pred = pred.sample.to(device)
    logger.info(f"  DiT output: {pred.shape}")

    # 5. Decode
    h_out = pred.shape[2] * ae_factor
    w_out = pred.shape[3] * ae_factor
    pred_up = F.interpolate(pred, size=(h_out, w_out), mode="bilinear", align_corners=False)
    out = vae.decode(pred_up).sample
    logger.info(f"  VAE decode: {out.shape}")

    # 6. Crop and color align
    out = out[:, :, :H * scale, :W * scale]
    if color_alignment != "none":
        out = apply_color_alignment(out, image_bchw, mode=color_alignment)

    out = out.clamp(0, 1)
    return out


# ── TESpeedVOSR2Settings ─────────────────────────────────────────
class TESpeedVOSR2Settings:
    CATEGORY = "VOSR2"
    FUNCTION = "get_settings"
    RETURN_TYPES = ("TE_SPEED_VOSR2_SETTINGS",)
    RETURN_NAMES = ("settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_profile": (["speed", "balanced", "quality"], {"default": "speed"}),
                "tile_strategy": (["auto", "even", "overlap"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "vae_tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "vae_tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "image_batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "frame_batch": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "dino_batch": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "temporal_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.003, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_refresh": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "memory_policy": (["auto", "low", "high"], {"default": "auto"}),
                "color_alignment": (["wavelet", "adain", "none"], {"default": "wavelet"}),
            }
        }

    def get_settings(self, **kwargs):
        return (dict(kwargs),)


# ── TESpeedVOSR2Loader ───────────────────────────────────────────
class TESpeedVOSR2Loader:
    CATEGORY = "VOSR2"
    FUNCTION = "load"
    RETURN_TYPES = ("TE_SPEED_VOSR2_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": (["VOSR2"], {"default": "VOSR2"}),
                "precision": (["auto", "fp32"], {"default": "auto"}),
                "memory_policy": (["auto", "low", "high"], {"default": "auto"}),
            },
            "optional": {
                "torch_compile": ("BOOLEAN", {"default": False}),
                "vae_encode_amp": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, model_bundle="VOSR2", precision="auto", memory_policy="auto",
             torch_compile=False, vae_encode_amp=False):
        device = mm.get_torch_device()
        logger.info(f"Loading VOSR2 on {device} (fp32)")
        d2 = load_dinov2(device)
        vae = load_vae(device)
        dit, args = load_dit(device)
        mm.soft_empty_cache()
        return ({"device": device, "dinov2": d2, "vae": vae, "dit": dit, "args": args},)


# ── TESpeedVOSR2Image ────────────────────────────────────────────
class TESpeedVOSR2Image:
    CATEGORY = "VOSR2"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TE_SPEED_VOSR2_MODEL",),
                "images": ("IMAGE",),
                "settings": ("TE_SPEED_VOSR2_SETTINGS",),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    def upscale(self, model, images, settings, scale, seed):
        torch.manual_seed(seed)
        device = model["device"]
        img = images[0:1].permute(0, 3, 1, 2).contiguous().to(device, torch.float32)
        logger.info(f"Input: {img.shape}")
        color_alignment = settings.get("color_alignment", "wavelet")
        result = vosr2_upscale_one_step(
            img, scale, model["dinov2"], model["vae"], model["dit"],
            model["args"], device, color_alignment
        )
        result = result.permute(0, 2, 3, 1).cpu().to(torch.float32)
        mm.soft_empty_cache()
        return (result,)


# ── TESpeedVOSR2Video ────────────────────────────────────────────
class TESpeedVOSR2Video:
    CATEGORY = "VOSR2"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TE_SPEED_VOSR2_MODEL",),
                "images": ("IMAGE",),
                "settings": ("TE_SPEED_VOSR2_SETTINGS",),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    def upscale(self, model, images, settings, scale, seed):
        torch.manual_seed(seed)
        device = model["device"]
        color_alignment = settings.get("color_alignment", "wavelet")
        B = images.shape[0]
        frames_out = []
        for i in range(B):
            img = images[i:i+1].permute(0, 3, 1, 2).contiguous().to(device, torch.float32)
            result = vosr2_upscale_one_step(
                img, scale, model["dinov2"], model["vae"], model["dit"],
                model["args"], device, color_alignment
            )
            frames_out.append(result.permute(0, 2, 3, 1).cpu())
            mm.soft_empty_cache()
        return (torch.cat(frames_out, dim=0).to(torch.float32),)


# ── Registration ─────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "TESpeedVOSR2Settings": TESpeedVOSR2Settings,
    "TESpeedVOSR2Loader": TESpeedVOSR2Loader,
    "TESpeedVOSR2Image": TESpeedVOSR2Image,
    "TESpeedVOSR2Video": TESpeedVOSR2Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedVOSR2Settings": "VOSR2 Settings (Linux)",
    "TESpeedVOSR2Loader": "VOSR2 Model Loader (Linux)",
    "TESpeedVOSR2Image": "VOSR2 Image Upscale (Linux)",
    "TESpeedVOSR2Video": "VOSR2 Video Upscale (Linux)",
}