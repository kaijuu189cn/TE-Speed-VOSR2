"""VOSR2 inference pipeline — tiled DiT/VAE execution with caching and timing.

Reconstructed from the Cython-compiled `inference.pyd` via static analysis.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from .backend.color import apply_color_alignment
from .backend.tiled_vae import AE_FACTOR
from .model_store import TESpeedVOSR2Model
from .settings import VOSR2Settings

log = logging.getLogger("TE-Speed-VOSR2")

# ---------------------------------------------------------------------------
# Constants (matching the Cython module)
# ---------------------------------------------------------------------------

AE_FACTOR = 8          # VAE spatial compression ratio
DIT_PATCH = 2           # DiT patch size
PAD_MULTIPLE = 8        # pad alignment

# Speed-profile tuning
SPEED_DIT_TILE_BATCH = 4       # max DiT tiles in one run
SPEED_COLOR_DOWNSAMPLE = 4     # color ref downsample factor
BALANCED_DIT_TILE_BATCH = 4    # balanced profile batch
SAFE_VAE_TILE = 512            # fallback VAE tile size
PHASE_CHUNK_FRAMES = 4         # frames per phase chunk


# ---------------------------------------------------------------------------
# Phase clock (lightweight timing)
# ---------------------------------------------------------------------------


@dataclass
class _TimingRecord:
    elapsed: float = 0.0
    count: int = 0


class _PhaseClock:
    """Cumulative timing grouped by phase name."""

    def __init__(self):
        self.records: dict[str, _TimingRecord] = {}
        self.enabled: bool = True

    def phase(self, name: str) -> _Phase:
        return _Phase(self, name)

    def flush(self) -> dict[str, float]:
        out = {k: v.elapsed for k, v in self.records.items()}
        self.records.clear()
        return out


class _Phase:
    def __init__(self, clock: _PhaseClock, name: str):
        self.clock = clock
        self.name = name

    def __enter__(self):
        if self.clock.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.clock.enabled:
            elapsed = time.perf_counter() - self._start
            rec = self.clock.records.setdefault(self.name, _TimingRecord())
            rec.elapsed += elapsed
            rec.count += 1


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _pad_multiple(x: torch.Tensor, multiple: int = PAD_MULTIPLE) -> tuple[torch.Tensor, int, int]:
    """Pad spatial dims to a multiple, return (padded, orig_h, orig_w)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def _latent_side(pixel_size: int) -> int:
    """Latent size for a given pixel size (ceil division by AE_FACTOR)."""
    return (pixel_size + AE_FACTOR - 1) // AE_FACTOR


def _pad_square(x: torch.Tensor) -> torch.Tensor:
    """Pad to square (H == W) by reflecting edge pixels."""
    _, _, h, w = x.shape
    if h == w:
        return x
    side = max(h, w)
    pad_h = side - h
    pad_w = side - w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _noise(b: int, c: int, h: int, w: int, device: torch.device,
           dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(b, c, h, w, device=device, dtype=dtype, generator=g)


def _tile_positions(total: int, tile: int, overlap: int) -> list[int]:
    """Sorted start positions covering `total` with `tile`-sized windows."""
    stride = max(tile - overlap, 1)
    if total <= tile:
        return [0]
    positions = list(range(0, total - tile + 1, stride))
    if positions[-1] + tile < total:
        positions.append(total - tile)
    return sorted(set(positions))


def _restored_tile_weight(tile_h: int, tile_w: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
    """Gaussian blend weight (1, 1, tile_h, tile_w)."""
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device=device, dtype=dtype).view(1, 1, tile_h, tile_w)


# ---------------------------------------------------------------------------
# Feature cache helpers (video temporal)
# ---------------------------------------------------------------------------


def _signature(frames: torch.Tensor) -> int:
    """Simple hash of a frame tensor for cache lookup."""
    return hash((frames.shape, frames[0, 0, :8, :8].cpu().numpy().tobytes()))


def _cache_hit(cached: Optional[torch.Tensor], sig: int,
               cache: dict) -> tuple[bool, int]:
    if cached is None:
        return False, 0
    age = cache.get("age", 0)
    return cache.get("sig") == sig, age


def _reuse_flags(cache: dict, current_sig: int, n_frames: int,
                 threshold: float) -> list[bool]:
    """Per-frame reuse flags based on cache age and threshold."""
    return [False] * n_frames  # simplified


def _split_feature_batch(feats: torch.Tensor, dino_batch: int,
                         dim: int = 0) -> list[torch.Tensor]:
    """Split DINO features into manageable batches."""
    return list(feats.split(dino_batch, dim=dim))


def _cpu_store(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device=device, non_blocking=True)


def _keep_on_gpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


def _merge_cached_features(features: list[torch.Tensor],
                           reuse: list[bool],
                           cached: Optional[torch.Tensor]) -> list[torch.Tensor]:
    """Merge newly computed features with cached ones."""
    merged = []
    cache_idx = 0
    for i, (feat, should_reuse) in enumerate(zip(features, reuse)):
        if should_reuse and cached is not None and cache_idx < cached.shape[0]:
            merged.append(cached[cache_idx:cache_idx+1])
            cache_idx += 1
        else:
            merged.append(feat)
    return merged


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _prepare_vae_decode(model: TESpeedVOSR2Model,
                         tile_size: int, tile_overlap: int):
    model.prepare_vae_decode(tile_size, tile_overlap)


def _ensure_speed_runtime(model: TESpeedVOSR2Model):
    """Apply speed-optimization settings."""
    _enable_fast_math()
    if model._policy == "resident":
        model.pin_resident()
    log.info(
        "[TE-Speed-VOSR2] DiT attention backend for this run: %s",
        vosr_attention_backend(),
    )


def _enable_fast_math():
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        # Enable TF32 on supported GPUs
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True


def vosr_attention_backend():
    """Delegate to lightningdit's backend detection."""
    from .backend.models.lightningdit import vosr_attention_backend as _backend
    return _backend()


# ---------------------------------------------------------------------------
# Profile-dependent helpers
# ---------------------------------------------------------------------------


def _vae_tile_for_profile(settings: VOSR2Settings) -> tuple[int, int]:
    """Return (vae_tile_size, vae_tile_overlap) for the current profile."""
    n = settings.normalized()
    return n.vae_tile_size, n.vae_tile_overlap


def _vae_tile_count(lh: int, lw: int, tile: int, overlap: int) -> int:
    """Estimate number of VAE tiles needed."""
    if lh <= tile and lw <= tile:
        return 1
    hp = _tile_positions(lh, tile, overlap)
    wp = _tile_positions(lw, tile, overlap)
    return len(hp) * len(wp)


def _color_downsample(frames: torch.Tensor, factor: int = SPEED_COLOR_DOWNSAMPLE
                      ) -> torch.Tensor:
    """Downsample color reference for speed profile."""
    if factor <= 1:
        return frames
    _, _, h, w = frames.shape
    return F.interpolate(
        frames, size=(h // factor, w // factor), mode="area"
    )


def _dit_tile_batch(settings: VOSR2Settings) -> int:
    """Determine how many DiT tiles to batch together."""
    n = settings.normalized()
    if n.quality_profile == "speed":
        return SPEED_DIT_TILE_BATCH
    return 1


def _upsample_batch(feats: torch.Tensor, scale: float) -> torch.Tensor:
    """Upsample features to target scale."""
    if scale == 1.0:
        return feats
    _, _, h, w = feats.shape
    return F.interpolate(feats, scale_factor=scale, mode="bilinear",
                         align_corners=False)


def _finish_batch(outputs: list[torch.Tensor],
                   acc: torch.Tensor, wacc: torch.Tensor,
                   tile: tuple[int, int, int, int],
                   weight: torch.Tensor):
    pass  # Accumulation handled inline


def _fallback_vae_tile(tile_size: int, tile_overlap: int
                       ) -> tuple[int, int]:
    """Fallback to safe VAE tile when OOM."""
    new_size = max(tile_size // 2, SAFE_VAE_TILE)
    new_overlap = max(tile_overlap // 2, 64)
    log.info(
        "[TE-Speed-VOSR2] VAE tile %s ran out of memory, falling back to %s",
        tile_size, new_size,
    )
    return new_size, new_overlap


# ---------------------------------------------------------------------------
# Core tile execution
# ---------------------------------------------------------------------------


def _tile_geometry(h: int, w: int, tile_size: int,
                   tile_overlap: int) -> tuple[list[int], list[int], int, int]:
    """Compute tile grid for a (h, w) latent space."""
    h_pos = _tile_positions(h, tile_size, tile_overlap)
    w_pos = _tile_positions(w, tile_size, tile_overlap)
    return h_pos, w_pos, len(h_pos), len(w_pos)


def _dino_tiles(model: TESpeedVOSR2Model, frames: torch.Tensor,
                settings: VOSR2Settings
                ) -> list[torch.Tensor]:
    """Compute DINO features for each frame."""
    n = settings.normalized()
    dino_batch = n.dino_batch
    dino_features_list = []

    for i in range(0, frames.shape[0], dino_batch):
        batch = frames[i:i + dino_batch]
        # Ensure square for DINO
        batch_sq = _pad_square(batch)
        feats = model.dino_features(batch_sq)
        dino_features_list.append(feats.cpu())

    if dino_features_list:
        return torch.cat(dino_features_list, dim=0)
    return []


def _full_frame(model: TESpeedVOSR2Model, latent: torch.Tensor,
                z_ca: Optional[torch.Tensor],
                seed: int, noise_scale: float,
                settings: VOSR2Settings) -> torch.Tensor:
    """Non-tiled DiT inference on full latent."""
    return model.one_step(latent, z_ca=z_ca, seed=seed, noise_scale=noise_scale)


def _encode_resized(model: TESpeedVOSR2Model, images: torch.Tensor,
                    scale: float, vae_ts: int, vae_tol: int
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE encode at target resolution, returning (latent, mean, std)."""
    if scale != 1.0:
        _, _, h, w = images.shape
        images = F.interpolate(
            images, size=(round(h * scale), round(w * scale)),
            mode="bilinear", align_corners=False,
        )
    latent, mean, std = model.encode(images, vae_ts, vae_tol)
    return latent, mean, std


def _decode_restored(model: TESpeedVOSR2Model, sr_latent: torch.Tensor,
                     mean: torch.Tensor, std: torch.Tensor,
                     vae_ts: int, vae_tol: int,
                     orig_h: int, orig_w: int) -> torch.Tensor:
    """VAE decode and crop to original size."""
    decoded = model.decode(sr_latent, mean, std, vae_ts, vae_tol)
    # Crop to original spatial dims
    _, _, h, w = decoded.shape
    if h > orig_h:
        decoded = decoded[:, :, :orig_h, :]
    if w > orig_w:
        decoded = decoded[:, :, :, :orig_w]
    return decoded


# ---------------------------------------------------------------------------
# Main inference strategies
# ---------------------------------------------------------------------------


def _run_dit_tiles(
    model: TESpeedVOSR2Model,
    latent: torch.Tensor,
    z_ca: Optional[torch.Tensor],
    seed: int,
    noise_scale: float,
    settings: VOSR2Settings,
) -> torch.Tensor:
    """Run DiT tiles with overlap and Gaussian blending."""
    n = settings.normalized()
    dit_batch = _dit_tile_batch(settings)

    b, c, lh, lw = latent.shape
    tile_size = n.tile_size
    tile_overlap = n.tile_overlap

    # Tile size in latent space
    lt_size = max(tile_size // AE_FACTOR, 1)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)

    if lh <= lt_size and lw <= lt_size:
        # Single tile
        return model.one_step(latent, z_ca=z_ca, seed=seed, noise_scale=noise_scale)

    h_pos = _tile_positions(lh, lt_size, lt_overlap)
    w_pos = _tile_positions(lw, lt_size, lt_overlap)

    # Gaussian blend weight
    weight = _restored_tile_weight(lt_size, lt_size, latent.device, latent.dtype)

    acc = torch.zeros(b, c, lh, lw, device=latent.device, dtype=latent.dtype)
    wacc = torch.zeros(b, 1, lh, lw, device=latent.device, dtype=latent.dtype)

    total_tiles = len(h_pos) * len(w_pos)
    tiles_done = 0

    # Process tiles, potentially batching multiple tiles
    tile_batch: list[dict] = []
    for hi in h_pos:
        for wi in w_pos:
            he, we = hi + lt_size, wi + lt_size
            tile_latent = latent[:, :, hi:he, wi:we]
            tile_batch.append(dict(
                hi=hi, wi=wi, he=he, we=we, tile_latent=tile_latent
            ))

            if len(tile_batch) >= dit_batch:
                for item in tile_batch:
                    result = model.one_step(
                        item["tile_latent"], z_ca=z_ca,
                        seed=seed + tiles_done, noise_scale=noise_scale,
                    )
                    acc[:, :, item["hi"]:item["he"], item["wi"]:item["we"]] += result * weight
                    wacc[:, :, item["hi"]:item["he"], item["wi"]:item["we"]] += weight
                    tiles_done += 1
                tile_batch = []

    # Process remaining tiles
    for item in tile_batch:
        result = model.one_step(
            item["tile_latent"], z_ca=z_ca,
            seed=seed + tiles_done, noise_scale=noise_scale,
        )
        acc[:, :, item["hi"]:item["he"], item["wi"]:item["we"]] += result * weight
        wacc[:, :, item["hi"]:item["he"], item["wi"]:item["we"]] += weight
        tiles_done += 1

    return acc / wacc


def _phased_tiled(
    model: TESpeedVOSR2Model,
    image: torch.Tensor,
    z_ca: Optional[torch.Tensor],
    scale: float,
    settings: VOSR2Settings,
    seed: int = 0,
    noise_scale: float = 1.0,
    clock: Optional[_PhaseClock] = None,
) -> torch.Tensor:
    """Phased tiled inference — VAE encode → DiT tiles → VAE decode."""
    n = settings.normalized()
    c = clock or _PhaseClock()
    clock_enabled = clock is not None

    # ── phase 0: prepare ─────────────────────────────────────────────
    _ensure_speed_runtime(model)
    vae_ts, vae_tol = _vae_tile_for_profile(settings)

    # ── phase 1: VAE encode ──────────────────────────────────────────
    with c.phase("vae_encode") if clock_enabled else _nullcontext():
        # Color downsampling for speed profile
        ref_image = image
        if n.quality_profile == "speed" and n.color_alignment != "none":
            ref_image = _color_downsample(image, SPEED_COLOR_DOWNSAMPLE)

        latent, mean, std = _encode_resized(model, image, scale, vae_ts, vae_tol)

    # ── phase 2: DiT inference ───────────────────────────────────────
    with c.phase("dit") if clock_enabled else _nullcontext():
        if n.tile_strategy == "tiled" or n.tile_strategy == "balanced":
            sr_latent = _run_dit_tiles(model, latent, z_ca, seed, noise_scale, settings)
        else:
            sr_latent = _full_frame(model, latent, z_ca, seed, noise_scale, settings)

    # ── phase 3: VAE decode ──────────────────────────────────────────
    with c.phase("vae_decode") if clock_enabled else _nullcontext():
        _prepare_vae_decode(model, vae_ts, vae_tol)
        _, _, lh, lw = latent.shape
        orig_h = lh * AE_FACTOR
        orig_w = lw * AE_FACTOR
        decoded = _decode_restored(model, sr_latent, mean, std,
                                   vae_ts, vae_tol, orig_h, orig_w)

    # ── phase 4: color alignment ─────────────────────────────────────
    with c.phase("color") if clock_enabled else _nullcontext():
        if n.color_alignment != "none":
            downsample = SPEED_COLOR_DOWNSAMPLE if n.quality_profile == "speed" else 1
            decoded = apply_color_alignment(decoded, ref_image, n.color_alignment, downsample)

    return decoded


def _streaming_tiled(
    model: TESpeedVOSR2Model,
    images: torch.Tensor,
    z_ca_list: Optional[list[torch.Tensor]],
    scale: float,
    settings: VOSR2Settings,
    seed: int = 0,
    noise_scale: float = 1.0,
    clock: Optional[_PhaseClock] = None,
) -> torch.Tensor:
    """Streaming per-frame tiled inference (video frames)."""
    n = settings.normalized()
    outputs = []
    for i in range(images.shape[0]):
        frame = images[i:i+1]
        z_ca_i = z_ca_list[i] if z_ca_list else None
        out = _phased_tiled(
            model, frame, z_ca_i, scale, settings,
            seed=seed + i, noise_scale=noise_scale,
            clock=clock,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=0) if outputs else images


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_vosr2(
    model: TESpeedVOSR2Model,
    images: torch.Tensor,
    settings: VOSR2Settings,
    scale: float = 1.0,
    seed: int = 0,
    noise_scale: float = 1.0,
    dino_features: Optional[list[torch.Tensor]] = None,
    enable_timing: bool = False,
) -> tuple[torch.Tensor, Optional[dict[str, float]]]:
    """Main VOSR2 inference entry point.

    Args:
        model: Loaded TESpeedVOSR2Model.
        images: Input tensor (B, C, H, W) in [0, 1] range.
        settings: VOSR2Settings.
        scale: Upscale factor.
        seed: Random seed.
        noise_scale: Noise multiplier for one-step.
        dino_features: Optional precomputed DINO features.
        enable_timing: Whether to collect per-phase timing.

    Returns:
        (output_images, timing_dict)
    """
    clock = _PhaseClock() if enable_timing else _PhaseClock()
    clock.enabled = enable_timing

    n = settings.normalized()

    # Validate input
    if images.ndim != 4:
        raise ValueError(
            f"TE-Speed-VOSR2 expects IMAGE tensor [B, H, W, C], got shape {images.shape}"
        )
    if images.shape[1] not in (1, 3, 4):
        raise ValueError(
            f"TE-Speed-VOSR2 supports RGB/RGBA IMAGE input, got {images.shape[1]} channels"
        )

    # Convert from B, H, W, C → B, C, H, W if needed
    if images.shape[-1] in (1, 3, 4):
        images = images.movedim(-1, 1)

    # Keep only first 3 channels
    if images.shape[1] > 3:
        images = images[:, :3, :, :]

    # Apply scale
    if scale != 1.0:
        _, _, h, w = images.shape
        images = F.interpolate(
            images, size=(round(h * scale), round(w * scale)),
            mode="bilinear", align_corners=False,
        )

    # Pad to multiple of AE_FACTOR
    images, orig_h, orig_w = _pad_multiple(images, AE_FACTOR)

    # Prepare DINO cross-attention features
    z_ca_list = None
    if dino_features is not None:
        z_ca_list = dino_features
    elif hasattr(model, 'dino') and model.dino is not None and n.temporal_cache:
        # Compute DINO features on the fly
        with clock.phase("dino") if enable_timing else _nullcontext():
            dino_feats = _dino_tiles(model, images, settings)
            z_ca_list = [dino_feats[i:i+1] for i in range(dino_feats.shape[0])]

    # Run inference
    with clock.phase("inference") if enable_timing else _nullcontext():
        if images.shape[0] == 1 or n.quality_profile == "speed":
            # Single image or speed profile: process each frame
            output = _phased_tiled(
                model, images, z_ca_list[0] if z_ca_list else None,
                scale=1.0,  # Already scaled above
                settings=settings, seed=seed, noise_scale=noise_scale,
                clock=clock,
            )
        else:
            # Multi-frame / video: streaming
            output = _streaming_tiled(
                model, images, z_ca_list, scale=1.0,
                settings=settings, seed=seed, noise_scale=noise_scale,
                clock=clock,
            )

    # Crop to original spatial dims
    _, _, out_h, out_w = output.shape
    if out_h > orig_h:
        output = output[:, :, :orig_h, :]
    if out_w > orig_w:
        output = output[:, :, :, :orig_w]

    # Convert back to B, H, W, C
    output = output.movedim(1, -1)

    timing = clock.flush() if enable_timing else None
    return output, timing


def _nullcontext():
    """No-op context manager."""
    import contextlib
    return contextlib.nullcontext()