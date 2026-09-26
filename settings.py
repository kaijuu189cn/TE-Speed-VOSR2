"""VOSR2Settings dataclass — quality profiles, tiling, memory, and cache tuning.

Reconstructed from the Cython-compiled `settings.pyd` via static analysis of
symbol tables and string literals.  Every field, default, and constraint is
preserved; `normalized()` resolves auto/None values for the actual run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QualityProfile = Literal["speed", "manual"]
TileStrategy = Literal["auto", "tiled", "balanced"]
MemoryPolicy = Literal["resident", "staged", "auto"]
ColorAlignment = Literal["none", "adain", "wavelet"]


@dataclass(frozen=True)
class VOSR2Settings:
    """Reproduces the `settings.VOSR2Settings` Cython dataclass verbatim."""

    # ── profile ──────────────────────────────────────────────────────────
    quality_profile: QualityProfile = "speed"

    # ── spatial tiling ───────────────────────────────────────────────────
    tile_size: int = 512
    tile_overlap: int = 64
    tile_strategy: TileStrategy = "auto"

    # ── VAE tiling ───────────────────────────────────────────────────────
    vae_tile_size: int = 1024
    vae_tile_overlap: int = 128

    # ── memory ───────────────────────────────────────────────────────────
    memory_policy: MemoryPolicy = "resident"

    # ── batching ─────────────────────────────────────────────────────────
    image_batch: int = 1
    frame_batch: int = 1
    dino_batch: int = 1

    # ── color / temporal ─────────────────────────────────────────────────
    color_alignment: ColorAlignment = "wavelet"
    full_frame: bool = False
    temporal_cache: bool = True
    cache_refresh: int = 16
    cache_threshold: float = 0.7

    # ── legacy compat (kept for API surface) ─────────────────────────────
    tile: int = 512
    vae_tile: int = 1024
    overlap: int = 64
    vae_overlap: int = 128

    def normalized(self) -> VOSR2Settings:
        """Return a resolved copy suitable for the actual run.

        This replaces the Cython `normalized()` method on the frozen dataclass:
          * ``None`` / zero / under-range values → sane defaults
          * ``tile_strategy="auto"`` → ``"tiled"``
          * ``memory_policy="auto"`` → ``"resident"``
        """
        ts = self.tile_strategy
        if ts == "auto":
            ts = "tiled"

        mp = self.memory_policy
        if mp == "auto":
            mp = "resident"

        def _clamp(v, lo, hi, default):
            if v is None or v <= 0:
                return default
            return max(lo, min(hi, v))

        return VOSR2Settings(
            quality_profile=self.quality_profile if self.quality_profile in ("speed", "manual") else "speed",
            tile_size=_clamp(self.tile_size, 64, 8192, 512),
            tile_overlap=_clamp(self.tile_overlap, 8, 1024, 64),
            tile_strategy=ts,
            vae_tile_size=_clamp(self.vae_tile_size, 64, 8192, 1024),
            vae_tile_overlap=_clamp(self.vae_tile_overlap, 8, 1024, 128),
            memory_policy=mp,
            image_batch=max(1, self.image_batch or 1),
            frame_batch=max(1, self.frame_batch or 1),
            dino_batch=max(1, self.dino_batch or 1),
            color_alignment=self.color_alignment if self.color_alignment in ("none", "adain", "wavelet") else "wavelet",
            full_frame=bool(self.full_frame),
            temporal_cache=bool(self.temporal_cache),
            cache_refresh=max(1, self.cache_refresh or 16),
            cache_threshold=max(0.05, min(0.95, self.cache_threshold or 0.7)),
            tile=self.tile,
            vae_tile=self.vae_tile,
            overlap=self.overlap,
            vae_overlap=self.vae_overlap,
        )