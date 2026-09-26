"""ComfyUI nodes for TE-Speed-VOSR2.

Reconstructed from the Cython-compiled `nodes.pyd` via static analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from . import model_store, inference
from .model_store import VOSR2LoadError, bundle_names
from .settings import VOSR2Settings

log = logging.getLogger("TE-Speed-VOSR2")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_BUNDLE: str = "VOSR2"


def _get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_bundle_path(bundle: str) -> str:
    import folder_paths
    models_dir = folder_paths.get_folder_paths("vosr2")[0] \
        if hasattr(folder_paths, "get_folder_paths") else \
        folder_paths.models_dir
    return str(model_store._bundle_path(models_dir, bundle))


# ---------------------------------------------------------------------------
# NODE CLASS MAPPINGS
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# TESpeedVOSR2Loader
# ---------------------------------------------------------------------------


class TESpeedVOSR2Loader:
    """Load VOSR2 model bundle from disk."""

    CATEGORY = "TE-Speed/VOSR2"
    RETURN_TYPES = ("TE_SPEED_VOSR2_MODEL",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_bundle": (["VOSR2"], {"default": "VOSR2"}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "vae_encode_amp": ("BOOLEAN", {"default": True}),
                "precision": (["fp16", "bf16", "float32"], {"default": "fp16"}),
            }
        }

    def load(self, model_bundle: str, torch_compile: bool = False,
             vae_encode_amp: bool = True, precision: str = "fp16",
             **kwargs) -> tuple[model_store.TESpeedVOSR2Model]:
        """Load the model bundle and return it."""
        import folder_paths

        models_root = folder_paths.get_folder_paths("vosr2")[0] \
            if folder_paths.folder_names_and_paths.get("vosr2") else \
            str(folder_paths.models_dir)

        bundle_dir = model_store._bundle_path(models_root, model_bundle)

        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        cache_root = None
        try:
            cache_root = folder_paths.get_folder_paths("vosr2_cache")[0]
        except Exception:
            pass

        m = model_store.TESpeedVOSR2Model(
            bundle_dir=str(bundle_dir),
            device=torch.device(_get_device_type()),
            dtype=dtype,
            vae_encode_amp=vae_encode_amp,
            cache_root=cache_root,
        )

        if torch_compile:
            m.set_torch_compile(True)

        return (m,)


NODE_CLASS_MAPPINGS["TESpeedVOSR2Loader"] = TESpeedVOSR2Loader
NODE_DISPLAY_NAME_MAPPINGS["TESpeedVOSR2Loader"] = "TE-Speed VOSR2 Loader"

# ---------------------------------------------------------------------------
# TESpeedVOSR2Settings
# ---------------------------------------------------------------------------


class TESpeedVOSR2Settings:
    """Create VOSR2 inference settings."""

    CATEGORY = "TE-Speed/VOSR2"
    RETURN_TYPES = ("TE_SPEED_VOSR2_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "quality_profile": (["speed", "manual"], {"default": "speed"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 8, "max": 1024, "step": 8}),
                "tile_strategy": (["auto", "tiled", "balanced"], {"default": "auto"}),
                "vae_tile_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "vae_tile_overlap": ("INT", {"default": 128, "min": 8, "max": 1024, "step": 8}),
                "memory_policy": (["resident", "staged", "auto"], {"default": "resident"}),
                "image_batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "frame_batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "dino_batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "color_alignment": (["none", "adain", "wavelet"], {"default": "wavelet"}),
                "full_frame": ("BOOLEAN", {"default": False}),
                "temporal_cache": ("BOOLEAN", {"default": True}),
                "cache_refresh": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "cache_threshold": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 0.95, "step": 0.05}),
            }
        }

    def make(self, **kwargs) -> tuple[VOSR2Settings]:
        """Create and return a VOSR2Settings dataclass."""
        # Map param names to VOSR2Settings field names
        s = VOSR2Settings(
            quality_profile=kwargs.get("quality_profile", "speed"),
            tile_size=kwargs.get("tile_size", 512),
            tile_overlap=kwargs.get("tile_overlap", 64),
            tile_strategy=kwargs.get("tile_strategy", "auto"),
            vae_tile_size=kwargs.get("vae_tile_size", 1024),
            vae_tile_overlap=kwargs.get("vae_tile_overlap", 128),
            memory_policy=kwargs.get("memory_policy", "resident"),
            image_batch=kwargs.get("image_batch", 1),
            frame_batch=kwargs.get("frame_batch", 1),
            dino_batch=kwargs.get("dino_batch", 1),
            color_alignment=kwargs.get("color_alignment", "wavelet"),
            full_frame=kwargs.get("full_frame", False),
            temporal_cache=kwargs.get("temporal_cache", True),
            cache_refresh=kwargs.get("cache_refresh", 16),
            cache_threshold=kwargs.get("cache_threshold", 0.7),
        )
        return (s,)


NODE_CLASS_MAPPINGS["TESpeedVOSR2Settings"] = TESpeedVOSR2Settings
NODE_DISPLAY_NAME_MAPPINGS["TESpeedVOSR2Settings"] = "TE-Speed VOSR2 Settings"

# ---------------------------------------------------------------------------
# _VOSR2Base
# ---------------------------------------------------------------------------


class _VOSR2Base:
    """Base class for VOSR2 inference nodes."""

    CATEGORY = "TE-Speed/VOSR2"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("TE_SPEED_VOSR2_MODEL",),
                "settings": ("TE_SPEED_VOSR2_SETTINGS",),
                "images": ("IMAGE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0, "step": 0.25}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    def upscale(self, model: model_store.TESpeedVOSR2Model,
                settings: VOSR2Settings, images: torch.Tensor,
                scale: float = 1.0, seed: int = 0,
                noise_scale: float = 1.0,
                **kwargs) -> tuple[torch.Tensor]:
        """Run VOSR2 inference and return upscaled images."""
        n = settings.normalized()

        # Apply memory policy
        model.set_memory_policy(n.memory_policy)

        # Set torch compile if requested
        if model._compile_requested:
            model.set_torch_compile(True)

        output, _ = inference.run_vosr2(
            model=model,
            images=images,
            settings=settings,
            scale=scale,
            seed=seed,
            noise_scale=noise_scale,
            dino_features=None,
            enable_timing=False,
        )
        return (output,)


# ---------------------------------------------------------------------------
# TESpeedVOSR2Image
# ---------------------------------------------------------------------------


class TESpeedVOSR2Image(_VOSR2Base):
    """Upscale a single image or image batch with VOSR2."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        parent = super().INPUT_TYPES()
        parent["optional"]["auto_expand_vae_tile"] = ("BOOLEAN", {"default": True})
        parent["optional"]["batch_override"] = ("INT", {"default": 0, "min": 0, "max": 64, "step": 1})
        return parent

    def upscale(self, model, settings, images, scale=1.0, seed=0,
                noise_scale=1.0, auto_expand_vae_tile=True,
                batch_override=0, **kwargs):
        return super().upscale(
            model, settings, images, scale, seed, noise_scale,
        )


NODE_CLASS_MAPPINGS["TESpeedVOSR2Image"] = TESpeedVOSR2Image
NODE_DISPLAY_NAME_MAPPINGS["TESpeedVOSR2Image"] = "TE-Speed VOSR2 Image"

# ---------------------------------------------------------------------------
# TESpeedVOSR2Video
# ---------------------------------------------------------------------------


class TESpeedVOSR2Video(_VOSR2Base):
    """Upscale video frames with VOSR2 and temporal DINO caching."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        parent = super().INPUT_TYPES()
        parent["optional"]["effective_frame_batch"] = ("INT", {"default": 1, "min": 1, "max": 64, "step": 1})
        parent["optional"]["auto_expand_vae_tile"] = ("BOOLEAN", {"default": True})
        return parent

    def upscale(self, model, settings, images, scale=1.0, seed=0,
                noise_scale=1.0, effective_frame_batch=1,
                auto_expand_vae_tile=True, **kwargs):
        n = settings.normalized()

        # Override frame batch if specified
        if effective_frame_batch > 1:
            n = VOSR2Settings(
                quality_profile=n.quality_profile,
                tile_size=n.tile_size,
                tile_overlap=n.tile_overlap,
                tile_strategy=n.tile_strategy,
                vae_tile_size=n.vae_tile_size,
                vae_tile_overlap=n.vae_tile_overlap,
                memory_policy=n.memory_policy,
                image_batch=n.image_batch,
                frame_batch=effective_frame_batch,
                dino_batch=n.dino_batch,
                color_alignment=n.color_alignment,
                full_frame=n.full_frame,
                temporal_cache=n.temporal_cache,
                cache_refresh=n.cache_refresh,
                cache_threshold=n.cache_threshold,
            )

        model.set_memory_policy(n.memory_policy)

        output, _ = inference.run_vosr2(
            model=model,
            images=images,
            settings=n,
            scale=scale,
            seed=seed,
            noise_scale=noise_scale,
            enable_timing=False,
        )
        return (output,)


NODE_CLASS_MAPPINGS["TESpeedVOSR2Video"] = TESpeedVOSR2Video
NODE_DISPLAY_NAME_MAPPINGS["TESpeedVOSR2Video"] = "TE-Speed VOSR2 Video Frames"