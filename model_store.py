"""VOSR2 model store — bundle loading, memory management, VAE/DiT/DINO orchestration.

Reconstructed from the Cython-compiled `model_store.pyd` via static analysis.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
import comfy.model_management as mm
import comfy.utils
from comfy.model_patcher import ModelPatcher
from safetensors.torch import load_file

from .backend.tiled_vae import decode_dispatch, encode_dispatch
from .backend.models.dinov2 import build_dinov2_vitl14
from .backend.models.lightningdit import LightningDiT, vosr_attention_backend
from .backend.models.qwenimage_vae2d import AutoencoderKLQwenImage2D

log = logging.getLogger("TE-Speed-VOSR2")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ROOT = "vosr2/VOSR2"

_DINO_FILES = ("dinov2_vitl14.safetensors", "dinov2_vitl14_pretrain.pth")

VAE_DECODE_RESERVE = 2.0  # GiB reserved before VAE decode

_TORCHINDENT_CACHE_DIR = "TORCHINDUCTOR_CACHE_DIR"
_TRITON_CACHE_DIR = "TRITON_CACHE_DIR"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class VOSR2LoadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Bundle structure helper
# ---------------------------------------------------------------------------


def bundle_names(models_dir: str | Path) -> list[str]:
    """Return sorted list of available bundle directories under ``models_dir``."""
    p = Path(models_dir)
    if not p.is_dir():
        return []
    return sorted(
        d.name for d in p.iterdir() if d.is_dir() and (d / "args.json").is_file()
    )


def _bundle_path(models_dir: str | Path, bundle: str) -> Path:
    return Path(models_dir) / bundle


# ---------------------------------------------------------------------------
# TESpeedVOSR2Model
# ---------------------------------------------------------------------------


class TESpeedVOSR2Model:
    """Container for the full VOSR2 pipeline: DiT backbone, DINOv2 encoder, VAE."""

    def __init__(
        self,
        bundle_dir: str | Path,
        device: Optional[torch.device] = None,
        offload_device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        vae_encode_amp: bool = True,
        cache_root: Optional[str] = None,
    ):
        self.bundle_dir = Path(bundle_dir)
        self.device = device or mm.get_torch_device()
        self.offload_device = offload_device or mm.unet_offload_device()
        self.dtype = dtype or mm.unet_dtype()
        self.vae_encode_amp = vae_encode_amp

        if cache_root is not None:
            import os
            os.environ[_TORCHINDENT_CACHE_DIR] = str(Path(cache_root) / "inductor")
            os.environ[_TRITON_CACHE_DIR] = str(Path(cache_root) / "triton")

        # ── model handles (loaded lazily) ────────────────────────────────
        self.dit: Optional[LightningDiT] = None
        self.dino: Optional[torch.nn.Module] = None
        self.vae: Optional[AutoencoderKLQwenImage2D] = None
        self.dino_patcher: Optional[ModelPatcher] = None
        self.dit_patcher: Optional[ModelPatcher] = None
        self.vae_patcher: Optional[ModelPatcher] = None

        # state
        self._args: Optional[dict] = None
        self._compile_enabled: bool = False
        self._compile_fallback: bool = False  # True after first compile failure
        self._compile_first_call: bool = True
        self._compile_requested: bool = False
        self._resident_ready: bool = False
        self._dino_norm_cache: Optional[dict] = None

        # memory policy
        self._policy: str = "resident"
        self._active_patcher: Optional[ModelPatcher] = None

        self.load()

    # ── public helpers ───────────────────────────────────────────────────

    @property
    def _patchers(self):
        return [p for p in (self.dit_patcher, self.dino_patcher) if p is not None]

    def set_torch_compile(self, enabled: bool):
        self._compile_enabled = enabled
        self._compile_requested = enabled
        if enabled and not hasattr(torch, "compile"):
            log.warning("torch.compile requested but not available in this PyTorch version")
            self._compile_enabled = False

    def set_memory_policy(self, policy: str):
        self._policy = policy
        if policy == "staged":
            self._unload_transformers()
        elif policy == "resident":
            self.pin_resident()

    def pin_resident(self):
        """Pin all models to GPU."""
        for p in self._patchers:
            p.pin_resident()
        self._resident_ready = True

    def clear_staged(self):
        """Release staged (non-pinned) memory."""
        for p in self._patchers:
            p.unpin_resident()
        self._resident_ready = False

    @property
    def _transformer_bytes(self) -> int:
        total = 0
        if self.dit is not None:
            for p in self.dit.parameters():
                total += p.numel() * p.element_size()
        if self.dino is not None:
            for p in self.dino.parameters():
                total += p.numel() * p.element_size()
        return total

    def _unload_transformers(self):
        for p in self._patchers:
            mm.unload_model_and_clones(p)
        mm.soft_empty_cache()

    # ── VAE encode / decode ──────────────────────────────────────────────

    def _vae_autocast(self):
        """VAE is full fp32 internally, but we keep the context uniform."""
        if self.vae_encode_amp and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=self.device.type, dtype=torch.float32)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _autocast_if(self, condition: bool):
        if condition and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def encode(self, x: torch.Tensor, tile_size: int = 0, tile_overlap: int = 128):
        """VAE encode → normalized latent."""
        with self._vae_autocast():
            return encode_dispatch(self.vae, x, tile_size, tile_overlap)

    def decode(self, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
               tile_size: int = 0, tile_overlap: int = 128) -> torch.Tensor:
        """Denormalized latent → VAE decode."""
        with self._vae_autocast():
            return decode_dispatch(self.vae, z, mean, std, tile_size, tile_overlap)

    def prepare_vae_decode(self, tile_size: int, tile_overlap: int):
        """Release DiT/DINO residency before VAE decode (staged policy)."""
        # If we are in staged mode and VAE decode is large, free transformers first
        if self._policy == "staged":
            tile_count = math.ceil(4096 / (tile_size or 1024)) ** 2  # estimate
            free, total = mm.get_torch_device().mem_get_info()
            needed = tile_count * 256 * 1024 * 1024  # rough
            if free < needed:
                self._unload_transformers()
                mm.soft_empty_cache()
        elif self._policy == "resident":
            # Keep transformers resident but ensure VAE has room
            free, total = mm.get_torch_device().mem_get_info()
            reserve_bytes = int(VAE_DECODE_RESERVE * (1024 ** 3))
            if free < reserve_bytes:
                log.info(
                    "[TE-Speed-VOSR2] releasing DiT/DINO residency for VAE decode "
                    "(free %.2f GiB, reserve %.2f GiB)",
                    free / (1024**3), VAE_DECODE_RESERVE,
                )
                for p in self._patchers:
                    p.unpin_resident()
                mm.soft_empty_cache()

    # ── DINO features ────────────────────────────────────────────────────

    def _dino_norm(self, feats: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the feature dimension."""
        return torch.nn.functional.normalize(feats, dim=-1)

    def dino_features(self, x: torch.Tensor, layer: int = 17,
                      dino_size: int = 518) -> torch.Tensor:
        """Extract DINOv2 patch tokens at ``layer``."""
        with self._autocast_if(False):  # DINO runs in full precision
            # Resize to DINO input size
            x_resized = torch.nn.functional.interpolate(
                x, size=(dino_size, dino_size), mode="bicubic", align_corners=False
            )
            feats = self.dino.forward_intermediate_layer(x_resized, layer)
            return self._dino_norm(feats)

    # ── DiT forward ──────────────────────────────────────────────────────

    def velocity(self, z: torch.Tensor, t: torch.Tensor,
                 r: Optional[torch.Tensor] = None,
                 z_ca: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DiT forward_flexible — one-step velocity prediction."""
        with self._autocast_if(True):
            if self._compile_enabled and self._compile_first_call:
                self._compile_first_call = False
                try:
                    self.dit.forward_flexible = torch.compile(
                        self.dit.forward_flexible, mode="max-autotune"
                    )
                except Exception as e:
                    log.warning("[TE-Speed-VOSR2] torch.compile failed: %s", e)
                    self._compile_fallback = True
            return self.dit.forward_flexible(z, t, r=r, z=z_ca)

    def one_step(self, z: torch.Tensor, z_ca: Optional[torch.Tensor] = None,
                 seed: int = 0, noise_scale: float = 1.0) -> torch.Tensor:
        """One-step flow matching: sample random noise and denoise."""
        b, c, h, w = z.shape
        device = z.device
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(b, c, h, w, device=device, dtype=z.dtype, generator=generator)
        z_noisy = z + noise_scale * noise
        t = torch.ones(b, device=device, dtype=z.dtype)
        r = torch.zeros(b, device=device, dtype=z.dtype)
        return self.velocity(z_noisy, t, r=r, z_ca=z_ca)

    # ── Load logic ───────────────────────────────────────────────────────

    def load(self):
        """Load full bundle from ``self.bundle_dir``."""
        args_path = self.bundle_dir / "args.json"
        if not args_path.is_file():
            raise VOSR2LoadError(f"VOSR2 bundle not found: {self.bundle_dir}")

        with open(args_path, "r") as f:
            self._args = json.load(f)

        self._load()

    def _load(self):
        """Internal load: DiT, DINO, VAE."""
        self._load_dit()
        self._load_dino()
        self._load_vae()
        self.pin_resident()

    def _load_dit(self):
        """Load LightningDiT backbone."""
        dit_path = self.bundle_dir / "checkpoints" / "ema_model.safetensors"
        if not dit_path.is_file():
            raise VOSR2LoadError(f"Missing VOSR2 DiT weight under {dit_path}")

        args = self._args
        dit = LightningDiT(
            input_size=args.get("input_size", 32),
            patch_size=args.get("patch_size", 2),
            in_channels=args.get("in_channels", 32),
            out_channels=args.get("out_channels", 16),
            hidden_size=args.get("hidden_size", 1536),
            depth=args.get("depth", 36),
            num_heads=args.get("num_heads", 24),
            mlp_ratio=args.get("mlp_ratio", 4.0),
            use_qknorm=args.get("use_qknorm", False),
            use_swiglu=args.get("use_swiglu", False),
            use_rope=args.get("use_rope", False),
            use_rmsnorm=args.get("use_rmsnorm", False),
            z_dims=args.get("z_dims", None),
            encdim_ratio=args.get("encdim_ratio", 2),
            num_fused_layers=args.get("num_fused_layers", 1),
            auxiliary_time_cond=args.get("auxiliary_time_cond", False),
        )
        dit.eval().to(self.device, dtype=self.dtype)

        sd = load_file(str(dit_path))
        # Strip "ema_model." prefix if present
        clean = {}
        for k, v in sd.items():
            key = k
            if key.startswith("ema_model."):
                key = key[len("ema_model."):]
            clean[key] = v

        missing, unexpected = dit.load_state_dict(clean, strict=False)
        if missing:
            log.warning("DiT missing keys: %s", missing)
        if unexpected:
            log.warning("DiT unexpected keys: %s", unexpected)

        self.dit = dit
        self.dit_patcher = ModelPatcher(
            dit, load_device=self.device, offload_device=self.offload_device,
            dtype=self.dtype,
        )
        log.info("DiT loaded: %s parameters", sum(p.numel() for p in dit.parameters()))

    def _load_dino(self):
        """Load DINOv2 ViT-L/14 encoder."""
        dino_path = None
        for fname in _DINO_FILES:
            candidate = self.bundle_dir / fname
            if candidate.is_file():
                dino_path = candidate
                break
        if dino_path is None:
            # Try models directory
            dino_dir = self.bundle_dir.parent / "dinov2_vitl14.safetensors"
            if dino_dir.is_file():
                dino_path = dino_dir
        if dino_path is None:
            raise VOSR2LoadError(f"Missing DINOv2 weight under {self.bundle_dir}")

        dino = build_dinov2_vitl14()
        sd = load_file(str(dino_path))
        missing, unexpected = dino.load_state_dict(sd, strict=False)
        if missing:
            log.warning("DINO missing keys: %s", missing)
        if unexpected:
            log.warning("DINO unexpected keys: %s", unexpected)

        dino.eval().to(self.device, dtype=torch.float32)
        self.dino = dino
        self.dino_patcher = ModelPatcher(
            dino, load_device=self.device, offload_device=self.offload_device,
        )

    def _load_vae(self):
        """Load Qwen-Image 2D VAE."""
        vae_dir = self.bundle_dir / "Qwen-Image-vae-2d"
        if not vae_dir.is_dir():
            vae_dir = self.bundle_dir.parent / "Qwen-Image-vae-2d"
        if not vae_dir.is_dir():
            raise VOSR2LoadError(f"Missing VAE directory: Qwen-Image-vae-2d under {self.bundle_dir}")

        self.vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
        self.vae.eval().to(self.device, dtype=torch.float32)
        self.vae_patcher = ModelPatcher(
            self.vae, load_device=self.device, offload_device=self.offload_device,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def load_model(bundle_dir: str | Path, **kwargs) -> TESpeedVOSR2Model:
    """Convenience entry point matching ``model_store.load_model``."""
    return TESpeedVOSR2Model(bundle_dir, **kwargs)