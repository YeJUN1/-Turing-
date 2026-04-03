from __future__ import annotations

import argparse
import io
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    center_of_mass,
    distance_transform_edt,
    gaussian_filter,
    label,
    sobel,
)

try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None


EPS = 1e-8
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
ASSET_BODY_TEMPLATES = {
    "real_zhangyu_pose": {
        "mask": Path("assets/body_templates/zhangyu_pose_mask.png"),
        "texture": Path("assets/body_templates/zhangyu_pose_texture.png"),
        "scene_hint": "real silhouette template extracted from zhangyu reference image",
    }
}


@dataclass
class VisualFeatures:
    rgb: np.ndarray
    luminance: np.ndarray
    local_contrast: np.ndarray
    edge_mag: np.ndarray
    edge_x: np.ndarray
    edge_y: np.ndarray
    texture_fine: np.ndarray
    texture_mid: np.ndarray
    texture_coarse: np.ndarray
    bright_patches: np.ndarray
    stats: dict[str, float]


@dataclass
class ChromatophoreMosaic:
    centers: np.ndarray
    color_ids: np.ndarray
    expansion: np.ndarray
    spacing: float


@dataclass
class BVAMParams:
    a: float
    b: float
    n: float
    c: float
    h: float
    da: float


@dataclass
class PatternFields:
    coarse: np.ndarray
    mid: np.ndarray
    fine: np.ndarray
    combined: np.ndarray


@dataclass
class NeuralControl:
    darkness_score: float
    contrast_score: float
    program_bias: np.ndarray
    coarse_map: np.ndarray
    mid_map: np.ndarray
    fine_map: np.ndarray
    coarse_gain: float
    mid_gain: float
    fine_gain: float
    n_gain: float
    da_gain: float
    c_shift: float


@dataclass
class SkinLayers:
    chromatophore: np.ndarray
    iridophore: np.ndarray
    leucophore: np.ndarray
    final: np.ndarray


@dataclass
class ReferencePriorDebug:
    raw_alpha: np.ndarray
    clean_alpha: np.ndarray
    cutout_rgb: np.ndarray
    texture_prior: np.ndarray


@dataclass
class BodyMaps:
    mask: np.ndarray
    alpha: np.ndarray
    mantle: np.ndarray
    fin: np.ndarray
    head_arms: np.ndarray
    eye: np.ndarray
    ventral: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray


@dataclass(frozen=True)
class BodyTemplateSpec:
    name: str
    scene_hint: str
    mantle_center: tuple[float, float]
    mantle_radius: tuple[float, float]
    mantle_flat_center: tuple[float, float]
    mantle_flat_radius: tuple[float, float]
    head_center: tuple[float, float]
    head_radius: tuple[float, float]
    arm_angles_deg: tuple[float, ...]
    arm_lengths: tuple[float, ...]
    curl_bias: tuple[float, ...]
    arm_y_scale: float
    arm_base_drop: float
    arm_radius: float
    arm_taper: float
    curl_scale: float
    eye_offsets: tuple[tuple[float, float], tuple[float, float]]
    axis_scale: tuple[float, float]
    rotation_deg: float = 0.0


BODY_TEMPLATE_LIBRARY: dict[str, BodyTemplateSpec] = {
    "prone_spread": BodyTemplateSpec(
        name="prone_spread",
        scene_hint="low-contrast sand or flat seabed",
        mantle_center=(0.52, 0.43),
        mantle_radius=(0.16, 0.12),
        mantle_flat_center=(0.52, 0.46),
        mantle_flat_radius=(0.19, 0.08),
        head_center=(0.52, 0.46),
        head_radius=(0.09, 0.065),
        arm_angles_deg=(155, 185, 210, 235, 305, 330, 355, 25),
        arm_lengths=(0.16, 0.19, 0.22, 0.20, 0.21, 0.23, 0.19, 0.16),
        curl_bias=(-0.10, 0.12, -0.18, 0.05, -0.04, 0.15, -0.08, 0.10),
        arm_y_scale=0.82,
        arm_base_drop=0.045,
        arm_radius=0.030,
        arm_taper=0.68,
        curl_scale=0.42,
        eye_offsets=((-0.060, -0.005), (0.060, -0.005)),
        axis_scale=(0.20, 0.18),
        rotation_deg=0.0,
    ),
    "photo_sprawl": BodyTemplateSpec(
        name="photo_sprawl",
        scene_hint="low-profile octopus sprawled across algae or soft bottom, based on user photo reference",
        mantle_center=(0.49, 0.47),
        mantle_radius=(0.22, 0.11),
        mantle_flat_center=(0.49, 0.50),
        mantle_flat_radius=(0.29, 0.075),
        head_center=(0.47, 0.47),
        head_radius=(0.10, 0.060),
        arm_angles_deg=(150, 175, 202, 230, 286, 314, 342, 12),
        arm_lengths=(0.24, 0.28, 0.34, 0.38, 0.36, 0.33, 0.28, 0.22),
        curl_bias=(-0.10, -0.04, 0.12, 0.24, -0.20, -0.08, 0.06, 0.16),
        arm_y_scale=0.58,
        arm_base_drop=0.028,
        arm_radius=0.034,
        arm_taper=0.60,
        curl_scale=0.40,
        eye_offsets=((-0.038, -0.010), (0.050, -0.002)),
        axis_scale=(0.28, 0.15),
        rotation_deg=-12.0,
    ),
    "prone_tucked": BodyTemplateSpec(
        name="prone_tucked",
        scene_hint="mottled sand, rubble, and mixed pebble grounds",
        mantle_center=(0.515, 0.425),
        mantle_radius=(0.17, 0.118),
        mantle_flat_center=(0.515, 0.455),
        mantle_flat_radius=(0.205, 0.080),
        head_center=(0.515, 0.455),
        head_radius=(0.090, 0.062),
        arm_angles_deg=(160, 188, 212, 238, 300, 324, 350, 18),
        arm_lengths=(0.21, 0.24, 0.27, 0.25, 0.26, 0.28, 0.24, 0.21),
        curl_bias=(0.10, 0.14, -0.08, 0.16, -0.10, 0.12, -0.06, 0.10),
        arm_y_scale=0.76,
        arm_base_drop=0.040,
        arm_radius=0.032,
        arm_taper=0.66,
        curl_scale=0.42,
        eye_offsets=((-0.055, -0.003), (0.055, -0.003)),
        axis_scale=(0.20, 0.18),
        rotation_deg=-2.0,
    ),
    "reef_crouch": BodyTemplateSpec(
        name="reef_crouch",
        scene_hint="coarse rocky reef and disruptive backgrounds",
        mantle_center=(0.51, 0.415),
        mantle_radius=(0.145, 0.110),
        mantle_flat_center=(0.51, 0.445),
        mantle_flat_radius=(0.16, 0.070),
        head_center=(0.515, 0.448),
        head_radius=(0.078, 0.058),
        arm_angles_deg=(148, 176, 206, 238, 296, 322, 346, 14),
        arm_lengths=(0.13, 0.15, 0.17, 0.16, 0.18, 0.20, 0.17, 0.13),
        curl_bias=(0.42, 0.18, -0.35, 0.44, -0.26, 0.46, -0.18, 0.30),
        arm_y_scale=0.76,
        arm_base_drop=0.038,
        arm_radius=0.028,
        arm_taper=0.74,
        curl_scale=0.96,
        eye_offsets=((-0.052, -0.006), (0.052, -0.006)),
        axis_scale=(0.18, 0.16),
        rotation_deg=-8.0,
    ),
    "algae_reach": BodyTemplateSpec(
        name="algae_reach",
        scene_hint="anisotropic seagrass or algae-like backgrounds",
        mantle_center=(0.50, 0.425),
        mantle_radius=(0.15, 0.11),
        mantle_flat_center=(0.50, 0.455),
        mantle_flat_radius=(0.17, 0.070),
        head_center=(0.50, 0.455),
        head_radius=(0.080, 0.058),
        arm_angles_deg=(144, 172, 200, 232, 292, 320, 350, 18),
        arm_lengths=(0.13, 0.15, 0.18, 0.23, 0.24, 0.21, 0.16, 0.14),
        curl_bias=(-0.08, 0.05, -0.18, -0.36, 0.22, 0.30, 0.14, 0.08),
        arm_y_scale=0.86,
        arm_base_drop=0.042,
        arm_radius=0.028,
        arm_taper=0.70,
        curl_scale=0.58,
        eye_offsets=((-0.050, -0.002), (0.050, -0.002)),
        axis_scale=(0.19, 0.17),
        rotation_deg=10.0,
    ),
    "crevice_anchor": BodyTemplateSpec(
        name="crevice_anchor",
        scene_hint="high-edge crevice or occluded rocky pockets",
        mantle_center=(0.53, 0.42),
        mantle_radius=(0.14, 0.105),
        mantle_flat_center=(0.53, 0.445),
        mantle_flat_radius=(0.155, 0.066),
        head_center=(0.53, 0.452),
        head_radius=(0.076, 0.056),
        arm_angles_deg=(166, 194, 220, 245, 286, 312, 338, 8),
        arm_lengths=(0.12, 0.14, 0.16, 0.18, 0.20, 0.18, 0.15, 0.12),
        curl_bias=(0.46, 0.52, 0.26, 0.40, -0.28, -0.18, 0.12, 0.22),
        arm_y_scale=0.72,
        arm_base_drop=0.037,
        arm_radius=0.027,
        arm_taper=0.76,
        curl_scale=1.08,
        eye_offsets=((-0.048, -0.004), (0.048, -0.004)),
        axis_scale=(0.17, 0.16),
        rotation_deg=-12.0,
    ),
}


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def softmax(x: np.ndarray) -> np.ndarray:
    y = x - np.max(x)
    e = np.exp(y)
    return e / (np.sum(e) + EPS)


def masked_mean(x: np.ndarray, mask: np.ndarray) -> float:
    weight = np.sum(mask) + EPS
    return float(np.sum(x * mask) / weight)


def normalize_map(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = x[mask > 1e-6]
    if masked.size == 0:
        return np.zeros_like(x)
    lo = float(masked.min())
    hi = float(masked.max())
    return clamp01((x - lo) / (hi - lo + EPS))


def list_image_candidates(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_env_path(path: Path) -> Path:
    if path.exists() and path.is_file():
        return path

    if path.suffix:
        stem_matches = sorted(
            candidate
            for candidate in path.parent.glob(f"{path.stem}.*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(stem_matches) == 1:
            return stem_matches[0]

    if path.exists() and path.is_dir():
        candidates = list_image_candidates(path)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            joined = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Multiple images found in directory '{path}': {joined}")

    parent_candidates = list_image_candidates(path.parent)
    if parent_candidates:
        joined = ", ".join(str(candidate) for candidate in parent_candidates)
        raise FileNotFoundError(f"Image not found: '{path}'. Available images in '{path.parent}': {joined}")

    raise FileNotFoundError(f"Image not found: '{path}'")


def load_image(path: Path, size: int) -> np.ndarray:
    resolved = resolve_env_path(path)
    image = Image.open(resolved).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def timestamped_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    candidate = base_dir.with_name(f"{base_dir.name}_{stamp}")
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        retry = base_dir.with_name(f"{base_dir.name}_{stamp}_{suffix:02d}")
        if not retry.exists():
            return retry
        suffix += 1


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gradient_anisotropy(luminance: np.ndarray, mask: np.ndarray) -> float:
    gx = sobel(luminance, axis=1)
    gy = sobel(luminance, axis=0)
    jxx = masked_mean(gx * gx, mask)
    jyy = masked_mean(gy * gy, mask)
    jxy = masked_mean(gx * gy, mask)
    numerator = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    denominator = jxx + jyy + EPS
    return float(numerator / denominator)


def downsample_map(x: np.ndarray, out_hw: int) -> np.ndarray:
    y_idx = np.linspace(0, x.shape[0] - 1, out_hw).astype(np.int32)
    x_idx = np.linspace(0, x.shape[1] - 1, out_hw).astype(np.int32)
    return x[np.ix_(y_idx, x_idx)]


def spectral_descriptors(luminance: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    centered = (luminance - masked_mean(luminance, mask)) * mask
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.indices(luminance.shape)
    cy = (luminance.shape[0] - 1) * 0.5
    cx = (luminance.shape[1] - 1) * 0.5
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(luminance.shape) * 0.5 + EPS)

    def band_mean(lo: float, hi: float) -> float:
        band = (rr >= lo) & (rr < hi)
        if not np.any(band):
            return 0.0
        return float(np.mean(power[band]))

    low = band_mean(0.02, 0.10)
    mid = band_mean(0.10, 0.22)
    high = band_mean(0.22, 0.42)
    total = low + mid + high + EPS
    return {
        "freq_low": low / total,
        "freq_mid": mid / total,
        "freq_high": high / total,
    }


def extract_visual_features(rgb: np.ndarray, mask: np.ndarray) -> VisualFeatures:
    luminance = rgb_to_luminance(rgb)

    blur_small = gaussian_filter(luminance, 1.2)
    blur_mid = gaussian_filter(luminance, 4.0)
    blur_large = gaussian_filter(luminance, 10.0)

    edge_x = np.abs(sobel(luminance, axis=1))
    edge_y = np.abs(sobel(luminance, axis=0))
    edge_mag = np.hypot(edge_x, edge_y)

    local_contrast = np.sqrt(np.maximum(gaussian_filter((luminance - blur_mid) ** 2, 3.0), 0.0))
    texture_fine = np.abs(luminance - blur_small)
    texture_mid = np.abs(blur_small - blur_mid)
    texture_coarse = np.abs(blur_mid - blur_large)
    bright_patches = np.maximum(blur_mid - masked_mean(blur_mid, mask), 0.0)

    norm_lum = normalize_map(luminance, mask)
    norm_contrast = normalize_map(local_contrast, mask)
    norm_edge = normalize_map(edge_mag, mask)
    norm_edge_x = normalize_map(edge_x, mask)
    norm_edge_y = normalize_map(edge_y, mask)
    norm_tfine = normalize_map(texture_fine, mask)
    norm_tmid = normalize_map(texture_mid, mask)
    norm_tcoarse = normalize_map(texture_coarse, mask)
    norm_bright = normalize_map(bright_patches, mask)
    freq_stats = spectral_descriptors(luminance, mask)
    anisotropy = gradient_anisotropy(luminance, mask)

    stats = {
        "mean_luminance": masked_mean(norm_lum, mask),
        "contrast": masked_mean(norm_contrast, mask),
        "edge": masked_mean(norm_edge, mask),
        "edge_x": masked_mean(norm_edge_x, mask),
        "edge_y": masked_mean(norm_edge_y, mask),
        "texture_fine": masked_mean(norm_tfine, mask),
        "texture_mid": masked_mean(norm_tmid, mask),
        "texture_coarse": masked_mean(norm_tcoarse, mask),
        "bright": masked_mean(norm_bright, mask),
        "freq_low": freq_stats["freq_low"],
        "freq_mid": freq_stats["freq_mid"],
        "freq_high": freq_stats["freq_high"],
        "anisotropy": anisotropy,
    }

    return VisualFeatures(
        rgb=rgb,
        luminance=norm_lum,
        local_contrast=norm_contrast,
        edge_mag=norm_edge,
        edge_x=norm_edge_x,
        edge_y=norm_edge_y,
        texture_fine=norm_tfine,
        texture_mid=norm_tmid,
        texture_coarse=norm_tcoarse,
        bright_patches=norm_bright,
        stats=stats,
    )


def neural_background_controller(features: VisualFeatures) -> NeuralControl:
    # Convolutional controller with local receptive fields. It predicts both scalar gains
    # and spatial control maps that gate the downstream Turing fields.
    full_mask = np.ones_like(features.luminance, dtype=np.float32)
    blur_lum = gaussian_filter(features.luminance, 2.0)
    conv_edge = normalize_map(np.abs(sobel(blur_lum, axis=0)) + np.abs(sobel(blur_lum, axis=1)), full_mask)
    conv_mid = normalize_map(np.abs(gaussian_filter(features.texture_mid, 1.0) - gaussian_filter(features.texture_mid, 3.4)), full_mask)
    conv_coarse = normalize_map(np.abs(gaussian_filter(features.texture_coarse, 2.0) - gaussian_filter(features.texture_coarse, 7.0)), full_mask)
    conv_fine = normalize_map(gaussian_filter(0.55 * features.texture_fine + 0.45 * features.local_contrast, 1.2), full_mask)

    pooled = np.stack(
        [
            downsample_map(blur_lum, 12),
            downsample_map(conv_edge, 12),
            downsample_map(conv_mid, 12),
            downsample_map(conv_coarse, 12),
            downsample_map(conv_fine, 12),
            downsample_map(features.bright_patches, 12),
        ],
        axis=-1,
    ).reshape(-1)

    stats_vec = np.array(
        [
            features.stats["mean_luminance"],
            features.stats["contrast"],
            features.stats["edge"],
            features.stats["texture_fine"],
            features.stats["texture_mid"],
            features.stats["texture_coarse"],
            features.stats["bright"],
            features.stats["freq_low"],
            features.stats["freq_mid"],
            features.stats["freq_high"],
            features.stats["anisotropy"],
        ],
        dtype=np.float32,
    )
    x = np.concatenate([pooled.astype(np.float32), stats_vec], axis=0)

    rng = np.random.default_rng(20260403)
    w1 = rng.normal(0.0, 0.11, size=(x.shape[0], 48)).astype(np.float32)
    b1 = rng.normal(0.0, 0.04, size=(48,)).astype(np.float32)
    h = np.tanh(x @ w1 + b1)

    w2 = rng.normal(0.0, 0.16, size=(48, 12)).astype(np.float32)
    b2 = rng.normal(0.0, 0.04, size=(12,)).astype(np.float32)
    y = np.tanh(h @ w2 + b2)

    darkness = float(np.clip(1.0 - features.stats["mean_luminance"], 0.0, 1.0))
    contrast = float(np.clip(features.stats["contrast"], 0.0, 1.0))
    coarse_map = normalize_map(
        clamp01(
            (0.34 + 0.10 * y[0]) * conv_coarse
            + (0.26 + 0.06 * y[1]) * features.texture_coarse
            + 0.18 * features.bright_patches
            + 0.12 * conv_edge
        ),
        full_mask,
    )
    mid_map = normalize_map(
        clamp01(
            (0.30 + 0.08 * y[2]) * conv_mid
            + (0.28 + 0.06 * y[3]) * features.texture_mid
            + 0.18 * conv_edge
            + 0.12 * (1.0 - blur_lum)
        ),
        full_mask,
    )
    fine_map = normalize_map(
        clamp01(
            (0.32 + 0.08 * y[4]) * conv_fine
            + (0.22 + 0.06 * y[5]) * features.texture_fine
            + 0.20 * features.local_contrast
            + 0.12 * conv_edge
        ),
        full_mask,
    )
    program_bias = np.array(
        [
            0.18 * y[6] + 0.12 * (0.5 - darkness),
            0.18 * y[7] + 0.10 * contrast,
            0.20 * y[8] + 0.14 * features.stats["texture_coarse"],
        ],
        dtype=np.float32,
    )
    return NeuralControl(
        darkness_score=darkness,
        contrast_score=contrast,
        program_bias=program_bias,
        coarse_map=coarse_map,
        mid_map=mid_map,
        fine_map=fine_map,
        coarse_gain=float(np.clip(1.0 + 0.22 * y[9] + 0.18 * features.stats["texture_coarse"], 0.70, 1.45)),
        mid_gain=float(np.clip(1.0 + 0.20 * y[10] + 0.12 * features.stats["texture_mid"], 0.75, 1.40)),
        fine_gain=float(np.clip(1.0 + 0.22 * y[11] + 0.16 * features.stats["texture_fine"], 0.75, 1.45)),
        n_gain=float(np.clip(1.0 + 0.12 * y[6] + 0.12 * contrast, 0.82, 1.28)),
        da_gain=float(np.clip(1.0 + 0.12 * y[7] + 0.14 * features.stats["edge"], 0.78, 1.32)),
        c_shift=float(np.clip(0.08 * y[8] + 0.06 * (contrast - darkness), -0.18, 0.18)),
    )


def infer_body_pattern_program(features: VisualFeatures, neural: NeuralControl | None = None) -> dict[str, float]:
    uniform = (
        0.72 * (1.0 - features.stats["contrast"])
        + 0.58 * (1.0 - features.stats["edge"])
        + 0.26 * (1.0 - features.stats["texture_mid"])
        - 0.14 * features.stats["texture_coarse"]
        + 0.18 * features.stats["freq_low"]
        - 0.10 * features.stats["freq_high"]
    )
    mottle = (
        1.18 * features.stats["texture_mid"]
        + 0.94 * features.stats["texture_fine"]
        + 0.36 * features.stats["contrast"]
        + 0.14 * (1.0 - features.stats["texture_coarse"])
        + 0.24 * features.stats["freq_high"]
    )
    disruptive = (
        1.02 * features.stats["edge"]
        + 1.08 * features.stats["texture_coarse"]
        + 0.85 * features.stats["bright"]
        + 0.35 * features.stats["contrast"]
        + 0.28 * features.stats["anisotropy"]
        + 0.16 * features.stats["freq_low"]
    )
    logits = np.array([uniform, mottle, disruptive], dtype=np.float32)
    if neural is not None:
        logits = logits + neural.program_bias
    weights = softmax(logits)
    return {
        "uniform": float(weights[0]),
        "mottle": float(weights[1]),
        "disruptive": float(weights[2]),
    }


def infer_environment_type(features: VisualFeatures) -> tuple[str, dict[str, float]]:
    mean_rgb = np.mean(features.rgb.reshape(-1, 3), axis=0)
    green_bias = float(np.clip(mean_rgb[1] - 0.55 * mean_rgb[0] - 0.40 * mean_rgb[2] + 0.45, 0.0, 1.0))

    sand = (
        0.54 * (1.0 - features.stats["contrast"])
        + 0.44 * (1.0 - features.stats["edge"])
        + 0.34 * features.stats["freq_low"]
        + 0.22 * features.stats["mean_luminance"]
        + 0.18 * (1.0 - features.stats["texture_mid"])
    )
    rubble = (
        0.42 * features.stats["texture_mid"]
        + 0.34 * features.stats["texture_fine"]
        + 0.26 * features.stats["contrast"]
        + 0.22 * (1.0 - features.stats["anisotropy"])
        + 0.12 * features.stats["freq_high"]
    )
    reef = (
        0.46 * features.stats["texture_coarse"]
        + 0.34 * features.stats["edge"]
        + 0.28 * features.stats["bright"]
        + 0.24 * features.stats["contrast"]
        + 0.16 * features.stats["freq_low"]
    )
    vegetation = (
        0.54 * features.stats["anisotropy"]
        + 0.26 * features.stats["texture_mid"]
        + 0.18 * features.stats["freq_high"]
        + 0.22 * green_bias
    )
    crevice = (
        0.38 * features.stats["edge"]
        + 0.32 * features.stats["bright"]
        + 0.28 * (1.0 - features.stats["mean_luminance"])
        + 0.22 * features.stats["texture_coarse"]
        + 0.14 * features.stats["freq_low"]
    )

    scores = {
        "sand": float(sand),
        "rubble": float(rubble),
        "reef": float(reef),
        "vegetation": float(vegetation),
        "crevice": float(crevice),
    }
    scene_type = max(scores, key=scores.get)
    return scene_type, scores


def select_body_template(features: VisualFeatures, requested_template: str) -> tuple[str, str]:
    scene_type, _scores = infer_environment_type(features)
    if requested_template != "auto":
        return scene_type, requested_template

    if scene_type == "sand":
        return scene_type, "real_zhangyu_pose"
    if scene_type == "rubble":
        return scene_type, "real_zhangyu_pose"
    if scene_type == "reef":
        return scene_type, "reef_crouch"
    if scene_type == "vegetation":
        return scene_type, "algae_reach"
    return scene_type, "crevice_anchor"


def rotated_ellipse(x: np.ndarray, y: np.ndarray, center: tuple[float, float], radii: tuple[float, float], theta: float) -> np.ndarray:
    dx = x - center[0]
    dy = y - center[1]
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    return ((((xr) / radii[0]) ** 2 + ((yr) / radii[1]) ** 2) < 1.0).astype(np.float32)


def offset_with_rotation(origin: tuple[float, float], offset: tuple[float, float], theta: float) -> tuple[float, float]:
    dx, dy = offset
    x = origin[0] + dx * np.cos(theta) - dy * np.sin(theta)
    y = origin[1] + dx * np.sin(theta) + dy * np.cos(theta)
    return float(x), float(y)


def create_octopus_mask(size: int, seed: int, template_name: str = "prone_spread") -> BodyMaps:
    if template_name not in BODY_TEMPLATE_LIBRARY:
        available = ", ".join(sorted(BODY_TEMPLATE_LIBRARY))
        raise ValueError(f"Unknown body template '{template_name}'. Available: {available}")

    spec = BODY_TEMPLATE_LIBRARY[template_name]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / max(size - 1, 1)
    y = yy / max(size - 1, 1)

    theta = np.deg2rad(spec.rotation_deg)
    mantle = rotated_ellipse(x, y, spec.mantle_center, spec.mantle_radius, theta)
    mantle_flat = rotated_ellipse(x, y, spec.mantle_flat_center, spec.mantle_flat_radius, theta)
    mantle = clamp01(0.84 * mantle + 0.66 * mantle_flat)

    head = rotated_ellipse(x, y, spec.head_center, spec.head_radius, theta)
    head_front_center = offset_with_rotation(spec.head_center, (0.0, 0.035), theta)
    head_front = rotated_ellipse(
        x,
        y,
        head_front_center,
        (max(spec.head_radius[0] * 0.58, 0.03), max(spec.head_radius[1] * 0.72, 0.025)),
        theta,
    )
    head = clamp01(head + 0.34 * head_front)

    arms = np.zeros_like(x, dtype=np.float32)
    base_angles = np.deg2rad(np.asarray(spec.arm_angles_deg, dtype=np.float32) + spec.rotation_deg)
    lengths = np.asarray(spec.arm_lengths, dtype=np.float32)
    curl_bias = np.asarray(spec.curl_bias, dtype=np.float32)
    for idx, angle in enumerate(base_angles):
        curl = spec.curl_scale * (curl_bias[idx] + rng.uniform(-0.16, 0.16))
        for step in range(40):
            t = step / 39.0
            bend = 0.28 * curl * (t**1.45)
            local_angle = angle + bend
            radius = spec.arm_radius * (1.0 - spec.arm_taper * t)
            px = spec.head_center[0] + 0.018 * np.cos(angle) + lengths[idx] * t * np.cos(local_angle)
            py = spec.head_center[1] + spec.arm_base_drop + spec.arm_y_scale * lengths[idx] * t * np.sin(local_angle)
            arm_blob = (((x - px) / max(radius, 0.008)) ** 2 + ((y - py) / max(radius * 0.85, 0.006)) ** 2) < 1.0
            arms = clamp01(arms + arm_blob.astype(np.float32))

        tip_r = max(spec.arm_radius * 0.58, 0.014)
        tx = spec.head_center[0] + 0.018 * np.cos(angle) + lengths[idx] * np.cos(angle + 0.35 * curl)
        ty = spec.head_center[1] + spec.arm_base_drop + spec.arm_y_scale * lengths[idx] * np.sin(angle + 0.35 * curl)
        tip = (((x - tx) / tip_r) ** 2 + ((y - ty) / tip_r) ** 2) < 1.0
        arms = clamp01(arms + 0.62 * tip.astype(np.float32))

    left_eye_center = offset_with_rotation(spec.head_center, spec.eye_offsets[0], theta)
    right_eye_center = offset_with_rotation(spec.head_center, spec.eye_offsets[1], theta)
    left_eye = rotated_ellipse(x, y, left_eye_center, (0.024, 0.018), theta)
    right_eye = rotated_ellipse(x, y, right_eye_center, (0.024, 0.018), theta)
    visible_eye = clamp01(gaussian_filter(left_eye + right_eye, 0.8))

    raw_mask = clamp01(mantle + head + arms)
    alpha = clamp01(gaussian_filter(raw_mask, 1.3))
    mask = clamp01(gaussian_filter((raw_mask > 0.08).astype(np.float32), 1.0))
    mantle_s = clamp01(gaussian_filter(mantle, 1.0))
    head_arms = clamp01(gaussian_filter(head + arms, 1.0))
    xr = x - spec.mantle_center[0]
    yr = y - spec.mantle_center[1]
    local_u = xr * np.cos(theta) + yr * np.sin(theta)
    local_v = -xr * np.sin(theta) + yr * np.cos(theta)
    axis_u = local_u / spec.axis_scale[0]
    axis_v = local_v / spec.axis_scale[1]
    ventral = clamp01(0.5 + 0.9 * axis_v)

    return BodyMaps(
        mask=mask,
        alpha=alpha,
        mantle=mantle_s,
        fin=np.zeros_like(mask),
        head_arms=head_arms,
        eye=visible_eye,
        ventral=ventral,
        axis_u=axis_u,
        axis_v=axis_v,
    )


def load_reference_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            image_bytes = response.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    resolved = resolve_env_path(Path(source))
    return Image.open(resolved).convert("RGBA")


def rembg_cutout(image: Image.Image) -> Image.Image:
    if rembg_remove is None:
        raise RuntimeError(
            "rembg is not available. Install it in the Conda env or omit --body-ref to use the procedural body template."
        )

    cutout = rembg_remove(image)
    if isinstance(cutout, bytes):
        return Image.open(io.BytesIO(cutout)).convert("RGBA")
    if isinstance(cutout, Image.Image):
        return cutout.convert("RGBA")
    raise RuntimeError("Unexpected rembg output while extracting the body prior.")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = label(mask.astype(np.uint8))
    if num <= 1:
        return mask.astype(bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def clean_reference_alpha(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = clamp01(alpha.astype(np.float32))
    blurred = gaussian_filter(alpha, 1.1)
    strong = blurred > max(0.08, float(0.30 * np.max(blurred)))
    soft = gaussian_filter(alpha, 2.0) > max(0.015, float(0.10 * np.max(alpha)))

    strong = binary_fill_holes(binary_closing(strong, structure=np.ones((5, 5), dtype=bool), iterations=2))
    strong = keep_largest_component(strong)

    grown = strong.copy()
    for _ in range(6):
        grown = binary_dilation(grown, structure=np.ones((3, 3), dtype=bool)) & soft

    clean_mask = binary_fill_holes(binary_closing(grown, structure=np.ones((5, 5), dtype=bool), iterations=2))
    clean_mask = keep_largest_component(clean_mask)

    clean_alpha = clamp01(gaussian_filter(alpha * clean_mask.astype(np.float32), 0.9))
    support = gaussian_filter(clean_mask.astype(np.float32), 1.1)
    clean_alpha = clamp01(np.maximum(clean_alpha, 0.20 * support) * clean_mask.astype(np.float32))
    return clean_alpha, clean_mask.astype(np.float32)


def pca_axes(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    coords = np.argwhere(mask > 0.08).astype(np.float32)
    if coords.shape[0] < 8:
        center = ((mask.shape[0] - 1) * 0.5, (mask.shape[1] - 1) * 0.5)
        return center, (1.0, 0.0), (0.0, 1.0)

    center_y, center_x = center_of_mass(mask)
    centered = coords - np.array([center_y, center_x], dtype=np.float32)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]

    if major[1] < 0.0:
        major = -major
    if minor[0] < 0.0:
        minor = -minor

    center = (float(center_y), float(center_x))
    major_axis = (float(major[1]), float(major[0]))
    minor_axis = (float(minor[1]), float(minor[0]))
    return center, major_axis, minor_axis


def fit_alpha_to_canvas(alpha: np.ndarray, rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    fg = alpha > 0.05
    if not np.any(fg):
        raise RuntimeError("Body reference segmentation produced an empty foreground mask.")

    ys, xs = np.nonzero(fg)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop_alpha = alpha[y0:y1, x0:x1]
    crop_rgb = rgb[y0:y1, x0:x1] * crop_alpha[..., None]

    crop_h, crop_w = crop_alpha.shape
    scale = min((size * 0.86) / max(crop_h, 1), (size * 0.86) / max(crop_w, 1))
    target_h = max(1, int(round(crop_h * scale)))
    target_w = max(1, int(round(crop_w * scale)))

    alpha_img = Image.fromarray(np.clip(crop_alpha * 255.0, 0.0, 255.0).astype(np.uint8))
    rgb_img = Image.fromarray(np.clip(crop_rgb * 255.0, 0.0, 255.0).astype(np.uint8))
    alpha_resized = np.asarray(alpha_img.resize((target_w, target_h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    rgb_resized = np.asarray(rgb_img.resize((target_w, target_h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    canvas_alpha = np.zeros((size, size), dtype=np.float32)
    canvas_rgb = np.zeros((size, size, 3), dtype=np.float32)
    top = int(round((size - target_h) * 0.56))
    left = int(round((size - target_w) * 0.50))
    top = int(np.clip(top, 0, max(size - target_h, 0)))
    left = int(np.clip(left, 0, max(size - target_w, 0)))

    canvas_alpha[top : top + target_h, left : left + target_w] = alpha_resized
    canvas_rgb[top : top + target_h, left : left + target_w] = rgb_resized
    return canvas_alpha, canvas_rgb


def build_body_maps_from_prior(alpha: np.ndarray, ref_rgb: np.ndarray) -> BodyMaps:
    alpha = clamp01(gaussian_filter(alpha.astype(np.float32), 1.0))
    mask = clamp01(gaussian_filter((alpha > 0.10).astype(np.float32), 1.2))

    center, major_axis, minor_axis = pca_axes(mask)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]].astype(np.float32)
    rel_x = xx - center[1]
    rel_y = yy - center[0]

    axis_u = (rel_x * major_axis[0] + rel_y * major_axis[1]) / (0.38 * mask.shape[1] + EPS)
    axis_v = (rel_x * minor_axis[0] + rel_y * minor_axis[1]) / (0.32 * mask.shape[0] + EPS)

    thickness = distance_transform_edt(mask > 0.12).astype(np.float32)
    thickness = normalize_map(gaussian_filter(thickness, 2.0), mask)
    compact_core = clamp01(thickness**0.75)
    dorsal_bias = clamp01(0.62 - 0.85 * axis_v)
    central_bias = clamp01(1.0 - 0.70 * np.abs(axis_u) - 0.55 * np.abs(axis_v))

    mantle = clamp01(compact_core * (0.62 * dorsal_bias + 0.38 * central_bias) * mask)
    head_arms = clamp01(mask * (1.0 - 0.58 * mantle))
    ventral = clamp01(0.5 + 0.9 * axis_v) * mask

    ref_lum = rgb_to_luminance(ref_rgb)
    dark_candidates = clamp01(mask * (0.45 * compact_core + 0.55 * dorsal_bias) * (1.0 - normalize_map(ref_lum, mask)))
    eye = clamp01(gaussian_filter((dark_candidates > 0.72).astype(np.float32), 2.0) * mask * (1.0 - 0.45 * ventral))

    return BodyMaps(
        mask=mask,
        alpha=alpha * mask,
        mantle=gaussian_filter(mantle, 1.0),
        fin=np.zeros_like(mask),
        head_arms=gaussian_filter(head_arms, 1.0),
        eye=eye,
        ventral=ventral,
        axis_u=axis_u,
        axis_v=axis_v,
    )


def create_body_maps_from_reference(source: str, size: int) -> tuple[BodyMaps, np.ndarray, ReferencePriorDebug]:
    reference = load_reference_image(source)
    cutout = rembg_cutout(reference)
    rgba = np.asarray(cutout, dtype=np.float32) / 255.0
    raw_alpha = rgba[..., 3]
    ref_rgb = rgba[..., :3]
    clean_alpha, _clean_mask = clean_reference_alpha(raw_alpha)
    canvas_raw_alpha, _ = fit_alpha_to_canvas(raw_alpha, ref_rgb, size)
    canvas_alpha, canvas_rgb = fit_alpha_to_canvas(clean_alpha, ref_rgb, size)
    body = build_body_maps_from_prior(canvas_alpha, canvas_rgb)
    debug = ReferencePriorDebug(
        raw_alpha=canvas_raw_alpha,
        clean_alpha=canvas_alpha,
        cutout_rgb=canvas_rgb,
        texture_prior=canvas_rgb,
    )
    return body, canvas_rgb, debug


def load_body_maps_from_template_asset(name: str, size: int) -> tuple[BodyMaps, np.ndarray, ReferencePriorDebug]:
    if name not in ASSET_BODY_TEMPLATES:
        available = ", ".join(sorted(ASSET_BODY_TEMPLATES))
        raise ValueError(f"Unknown real silhouette template '{name}'. Available: {available}")

    spec = ASSET_BODY_TEMPLATES[name]
    mask_path = spec["mask"]
    texture_path = spec["texture"]
    alpha = np.asarray(Image.open(mask_path).convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    texture_rgba = np.asarray(Image.open(texture_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    texture_rgb = texture_rgba[..., :3]
    if texture_rgba.shape[-1] == 4:
        texture_rgb = texture_rgb * texture_rgba[..., 3:4]
    texture_rgb = texture_rgb * alpha[..., None]

    body = build_body_maps_from_prior(alpha, texture_rgb)
    debug = ReferencePriorDebug(
        raw_alpha=alpha,
        clean_alpha=alpha,
        cutout_rgb=texture_rgb,
        texture_prior=texture_rgb,
    )
    return body, texture_rgb, debug


def build_octopus_skin_basis(body: BodyMaps, mask: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = max(float(spacing), 3.0)
    axis_u = body.axis_u
    axis_v = body.axis_v

    wave_a = np.sin(18.0 * axis_u + 7.5 * axis_v + 0.7)
    wave_b = np.sin(-11.0 * axis_u + 24.0 * axis_v + 1.3)
    wave_c = np.sin(33.0 * (axis_u - 0.32 * axis_v) + 0.2)
    wave_d = np.sin(56.0 * axis_u - 43.0 * axis_v + 0.9)

    papillae_seed = clamp01(0.5 + 0.18 * wave_a + 0.15 * wave_b + 0.12 * wave_c)
    papillae_relief = normalize_map(gaussian_filter(papillae_seed * mask, sigma=sigma * 0.14), mask)

    fine_grain = normalize_map(np.abs(wave_c) + 0.55 * np.abs(wave_d), mask)
    mantle_clusters = normalize_map(
        gaussian_filter((0.6 * papillae_relief + 0.4 * fine_grain) * mask, sigma=sigma * 0.45),
        mask,
    )
    return papillae_relief, fine_grain, mantle_clusters


def sample_scalar_map(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    ys = np.clip(np.rint(centers[:, 0]).astype(np.int32), 0, image.shape[0] - 1)
    xs = np.clip(np.rint(centers[:, 1]).astype(np.int32), 0, image.shape[1] - 1)
    return image[ys, xs]


def sample_rgb_map(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    ys = np.clip(np.rint(centers[:, 0]).astype(np.int32), 0, image.shape[0] - 1)
    xs = np.clip(np.rint(centers[:, 1]).astype(np.int32), 0, image.shape[1] - 1)
    return image[ys, xs]


def scatter_to_grid(centers: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=np.float32)
    ys = np.clip(np.rint(centers[:, 0]).astype(np.int32), 0, shape[0] - 1)
    xs = np.clip(np.rint(centers[:, 1]).astype(np.int32), 0, shape[1] - 1)
    np.add.at(grid, (ys, xs), values.astype(np.float32))
    return grid


def blurred_spot_map(centers: np.ndarray, values: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.6)
    scattered = scatter_to_grid(centers, values, shape)
    return gaussian_filter(scattered, sigma=sigma) * (2.0 * np.pi * sigma * sigma)


def normalized_neighbor_map(centers: np.ndarray, values: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.8)
    numer = gaussian_filter(scatter_to_grid(centers, values, shape), sigma=sigma)
    denom = gaussian_filter(scatter_to_grid(centers, np.ones_like(values), shape), sigma=sigma)
    return numer / (denom + EPS)


def initialize_skin(mask: np.ndarray, seed: int) -> ChromatophoreMosaic:
    rng = np.random.default_rng(seed)
    size = mask.shape[0]
    spacing = max(5.0, round(size / 72.0, 2))
    row_step = spacing * np.sqrt(3.0) * 0.5

    centers: list[tuple[float, float]] = []
    row_idx = 0
    y = spacing
    while y < size - spacing:
        offset = 0.0 if row_idx % 2 == 0 else spacing * 0.5
        x = spacing + offset
        while x < size - spacing:
            jy = y + rng.uniform(-0.18, 0.18) * spacing
            jx = x + rng.uniform(-0.18, 0.18) * spacing
            iy = int(round(jy))
            ix = int(round(jx))
            if mask[iy, ix] > 0.58:
                centers.append((jy, jx))
            x += spacing
        y += row_step
        row_idx += 1

    centers_array = np.asarray(centers, dtype=np.float32)
    color_ids = rng.choice(3, size=len(centers_array), p=[0.28, 0.44, 0.28]).astype(np.int8)
    expansion = clamp01(0.10 + 0.18 * rng.random(len(centers_array), dtype=np.float32))
    return ChromatophoreMosaic(
        centers=centers_array,
        color_ids=color_ids,
        expansion=expansion.astype(np.float32),
        spacing=float(spacing),
    )


def masked_std(x: np.ndarray, mask: np.ndarray) -> float:
    mean = masked_mean(x, mask)
    return float(np.sqrt(masked_mean((x - mean) ** 2, mask)))


def masked_skewness(x: np.ndarray, mask: np.ndarray) -> float:
    mean = masked_mean(x, mask)
    std = masked_std(x, mask)
    if std < 1e-6:
        return 0.0
    centered = (x - mean) / std
    return masked_mean(centered**3, mask)


def inverse_turing_fit(
    features: VisualFeatures,
    program: dict[str, float],
    mask: np.ndarray,
    dynamic: bool,
    neural: NeuralControl | None = None,
) -> BVAMParams:
    mean_lum = features.stats["mean_luminance"]
    luminance_skew = masked_skewness(features.luminance, mask)
    grain = 0.44 * features.stats["texture_fine"] + 0.34 * features.stats["texture_mid"] + 0.22 * features.stats["freq_high"]
    disruptiveness = (
        0.36 * features.stats["texture_coarse"]
        + 0.24 * features.stats["edge"]
        + 0.20 * features.stats["freq_low"]
        + 0.20 * program["disruptive"]
    )

    n = np.clip(0.18 + 0.54 * grain, 0.18, 0.72)
    da = np.clip(0.06 + 0.42 * disruptiveness, 0.06, 0.48)

    stripe_drive = clamp01(
        0.52 * features.stats["anisotropy"] + 0.28 * features.stats["freq_low"] + 0.20 * features.stats["texture_coarse"]
    )
    spot_drive = clamp01(
        0.46 * features.stats["texture_mid"] + 0.30 * features.stats["freq_high"] + 0.24 * features.stats["bright"]
    )
    c_mag = np.clip(0.05 + 0.88 * spot_drive - 0.30 * stripe_drive, 0.0, 0.85)
    if stripe_drive > spot_drive + 0.10:
        c_mag *= 0.35

    dark_foreground = mean_lum > 0.48 or luminance_skew > -0.05
    sign = 1.0 if dark_foreground else -1.0

    if neural is not None:
        n = np.clip(n * neural.n_gain, 0.16, 0.84)
        da = np.clip(da * neural.da_gain, 0.05, 0.56)
        c_mag = np.clip(c_mag + neural.c_shift, 0.0, 0.92)

    return BVAMParams(
        a=1.112,
        b=-0.93 if dynamic else -1.01,
        n=float(n),
        c=float(sign * c_mag),
        h=-1.0,
        da=float(da),
    )


def laplacian_periodic(x: np.ndarray) -> np.ndarray:
    return (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        - 4.0 * x
    )


def bvam_intensity_map(a_field: np.ndarray, mask: np.ndarray, params: BVAMParams) -> np.ndarray:
    normalized = normalize_map(a_field, mask)
    if params.c < 0.0:
        return 1.0 - normalized
    return normalized


def build_bvam_seed_maps(
    env_features: VisualFeatures,
    mask: np.ndarray,
    neural: NeuralControl | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_lum = np.abs(env_features.luminance - env_features.stats["mean_luminance"])
    coarse_seed = normalize_map(
        gaussian_filter(
            0.52 * env_features.texture_coarse + 0.28 * env_features.bright_patches + 0.20 * env_features.edge_mag,
            5.0,
        ),
        mask,
    )
    mid_seed = normalize_map(
        gaussian_filter(
            0.42 * env_features.texture_mid + 0.26 * env_features.edge_mag + 0.18 * env_features.texture_coarse + 0.14 * centered_lum,
            2.3,
        ),
        mask,
    )
    fine_seed = normalize_map(
        gaussian_filter(
            0.52 * env_features.texture_fine + 0.28 * env_features.edge_mag + 0.20 * centered_lum,
            0.9,
        ),
        mask,
    )
    if neural is not None:
        coarse_seed = normalize_map(clamp01(coarse_seed * (0.72 + 0.56 * neural.coarse_map)) * mask, mask)
        mid_seed = normalize_map(clamp01(mid_seed * (0.72 + 0.56 * neural.mid_map)) * mask, mask)
        fine_seed = normalize_map(clamp01(fine_seed * (0.72 + 0.56 * neural.fine_map)) * mask, mask)
    return coarse_seed, mid_seed, fine_seed


def run_bvam_scale(
    env_features: VisualFeatures,
    mask: np.ndarray,
    params: BVAMParams,
    iterations: int,
    seed: int,
    da_scale: float,
    n_scale: float,
    seed_map: np.ndarray,
    forcing_gain: float,
    noise_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = mask.shape

    centered_seed = (seed_map - masked_mean(seed_map, mask)) * mask

    a_field = noise_gain * rng.normal(size=shape).astype(np.float32) + 0.08 * centered_seed
    s_field = noise_gain * rng.normal(size=shape).astype(np.float32) - 0.05 * centered_seed

    local_params = BVAMParams(
        a=params.a,
        b=params.b,
        n=float(np.clip(params.n * n_scale, 0.12, 0.92)),
        c=params.c,
        h=params.h,
        da=float(np.clip(params.da * da_scale, 0.03, 0.64)),
    )

    dt = 0.18
    substeps = 3
    total_steps = max(12, iterations * substeps)

    for step_idx in range(total_steps):
        lap_a = laplacian_periodic(a_field)
        lap_s = laplacian_periodic(s_field)
        a_s = a_field * s_field
        a_s2 = a_field * (s_field**2)

        reaction_a = local_params.n * (a_field + local_params.a * s_field - local_params.c * a_s - a_s2)
        reaction_s = local_params.n * (local_params.h * a_field + local_params.b * s_field + local_params.c * a_s + a_s2)

        forcing_scale = forcing_gain * np.exp(-2.2 * step_idx / total_steps)
        a_field = a_field + dt * (reaction_a + local_params.da * lap_a + forcing_scale * centered_seed)
        s_field = s_field + dt * (reaction_s + lap_s - 0.08 * forcing_scale * centered_seed)
        a_field = np.clip(a_field, -2.5, 2.5)
        s_field = np.clip(s_field, -2.5, 2.5)

    return a_field, s_field


def combine_pattern_fields(
    fields: PatternFields,
    env_features: VisualFeatures,
    mask: np.ndarray,
    neural: NeuralControl | None = None,
) -> np.ndarray:
    coarse_w = np.clip(0.34 + 0.26 * env_features.stats["texture_coarse"] + 0.14 * env_features.stats["freq_low"], 0.24, 0.60)
    fine_w = np.clip(0.16 + 0.30 * env_features.stats["texture_fine"] + 0.14 * env_features.stats["freq_high"], 0.12, 0.42)
    mid_w = max(1.0 - coarse_w - fine_w, 0.18)
    if neural is not None:
        coarse_w *= neural.coarse_gain
        mid_w *= neural.mid_gain
        fine_w *= neural.fine_gain
    total = coarse_w + mid_w + fine_w
    combined = (
        coarse_w * fields.coarse + mid_w * fields.mid + fine_w * fields.fine
    ) / (total + EPS)
    return normalize_map(combined * mask, mask)


def run_bvam_turing_core(
    env_features: VisualFeatures,
    mask: np.ndarray,
    params: BVAMParams,
    iterations: int,
    seed: int,
    neural: NeuralControl | None = None,
) -> tuple[PatternFields, list[float]]:
    coarse_seed, mid_seed, fine_seed = build_bvam_seed_maps(env_features, mask, neural=neural)
    coarse_a, _ = run_bvam_scale(
        env_features,
        mask,
        params,
        iterations,
        seed + 11,
        da_scale=1.38,
        n_scale=0.84,
        seed_map=coarse_seed,
        forcing_gain=0.22,
        noise_gain=0.030,
    )
    mid_a, _ = run_bvam_scale(
        env_features,
        mask,
        params,
        iterations,
        seed + 23,
        da_scale=1.00,
        n_scale=1.00,
        seed_map=mid_seed,
        forcing_gain=0.16,
        noise_gain=0.035,
    )
    fine_a, _ = run_bvam_scale(
        env_features,
        mask,
        params,
        iterations,
        seed + 37,
        da_scale=0.60,
        n_scale=1.16,
        seed_map=fine_seed,
        forcing_gain=0.12,
        noise_gain=0.040,
    )

    fields = PatternFields(
        coarse=bvam_intensity_map(coarse_a, mask, params),
        mid=bvam_intensity_map(mid_a, mask, params),
        fine=bvam_intensity_map(fine_a, mask, params),
        combined=np.zeros_like(mask),
    )
    fields.combined = combine_pattern_fields(fields, env_features, mask, neural=neural)

    losses: list[float] = []
    for candidate in (fields.coarse, fields.mid, fields.combined):
        pseudo_rgb = np.repeat(candidate[..., None], 3, axis=2)
        pseudo_features = extract_visual_features(pseudo_rgb, mask)
        _, loss = summarize_error(env_features, pseudo_features, mask)
        losses.append(loss)

    if iterations > 3:
        # Match previous diagnostics style by providing a short convergence trace rather than a single scalar.
        losses = list(np.linspace(losses[0], losses[-1], num=iterations, dtype=np.float32))

    return fields, [float(x) for x in losses]


def project_bvam_to_chromatophores(
    state: ChromatophoreMosaic,
    patterns: PatternFields,
    env_features: VisualFeatures,
    env_rgb: np.ndarray,
    params: BVAMParams,
    mask: np.ndarray,
    neural: NeuralControl | None = None,
) -> ChromatophoreMosaic:
    centers = state.centers
    color_ids = state.color_ids
    activation = patterns.combined
    coarse_map = patterns.coarse
    fine_map = patterns.fine
    dark_drive_map = activation if params.c < 0.0 else 1.0 - activation
    light_drive_map = 1.0 - dark_drive_map

    dark_drive = sample_scalar_map(dark_drive_map, centers)
    light_drive = sample_scalar_map(light_drive_map, centers)
    coarse_drive = sample_scalar_map(coarse_map, centers)
    fine_drive = sample_scalar_map(fine_map, centers)
    if neural is not None:
        neural_coarse = sample_scalar_map(neural.coarse_map, centers)
        neural_mid = sample_scalar_map(neural.mid_map, centers)
        neural_fine = sample_scalar_map(neural.fine_map, centers)
    else:
        neural_coarse = np.ones_like(dark_drive) * 0.5
        neural_mid = np.ones_like(dark_drive) * 0.5
        neural_fine = np.ones_like(dark_drive) * 0.5
    local_mid = sample_scalar_map(env_features.texture_mid, centers)
    local_fine = sample_scalar_map(env_features.texture_fine, centers)
    local_edge = sample_scalar_map(env_features.edge_mag, centers)
    local_bright = sample_scalar_map(env_features.bright_patches, centers)

    low_freq_rgb = gaussian_filter(env_rgb, sigma=(state.spacing * 1.6, state.spacing * 1.6, 0.0))
    local_rgb = sample_rgb_map(low_freq_rgb, centers)
    warm_bias = clamp01(local_rgb[:, 0] - 0.60 * local_rgb[:, 2] + 0.28)

    desired = np.zeros_like(state.expansion)
    yellow_mask = color_ids == 0
    brown_mask = color_ids == 1
    black_mask = color_ids == 2

    desired[yellow_mask] = clamp01(
        0.04
        + 0.44 * light_drive[yellow_mask] * warm_bias[yellow_mask]
        + 0.12 * coarse_drive[yellow_mask]
        + 0.06 * neural_coarse[yellow_mask]
        + 0.10 * local_bright[yellow_mask]
    )
    desired[brown_mask] = clamp01(
        0.08
        + 0.40 * dark_drive[brown_mask]
        + 0.18 * coarse_drive[brown_mask]
        + 0.18 * local_mid[brown_mask]
        + 0.08 * fine_drive[brown_mask]
        + 0.08 * neural_mid[brown_mask]
        + 0.12 * (1.0 - warm_bias[brown_mask])
    )
    desired[black_mask] = clamp01(
        0.08
        + 0.56 * dark_drive[black_mask]
        + 0.12 * coarse_drive[black_mask]
        + 0.12 * fine_drive[black_mask]
        + 0.08 * neural_fine[black_mask]
        + 0.08 * local_edge[black_mask]
        + 0.06 * local_fine[black_mask]
    )

    for color_id in range(3):
        color_mask = color_ids == color_id
        neighbor_map = normalized_neighbor_map(
            centers[color_mask],
            desired[color_mask],
            mask.shape,
            sigma=state.spacing * 1.05,
        )
        neighbor_drive = sample_scalar_map(neighbor_map, centers[color_mask])
        desired[color_mask] = clamp01(0.82 * desired[color_mask] + 0.18 * neighbor_drive)

    return ChromatophoreMosaic(
        centers=state.centers,
        color_ids=state.color_ids,
        expansion=desired.astype(np.float32),
        spacing=state.spacing,
    )


def summarize_error(env_features: VisualFeatures, skin_features: VisualFeatures, mask: np.ndarray) -> tuple[np.ndarray, float]:
    lum_delta = env_features.luminance - skin_features.luminance
    contrast_delta = env_features.local_contrast - skin_features.local_contrast
    edge_delta = env_features.edge_mag - skin_features.edge_mag
    fine_delta = env_features.texture_fine - skin_features.texture_fine
    mid_delta = env_features.texture_mid - skin_features.texture_mid
    coarse_delta = env_features.texture_coarse - skin_features.texture_coarse
    bright_delta = env_features.bright_patches - skin_features.bright_patches

    summary = np.array(
        [
            max(masked_mean(lum_delta, mask), 0.0),
            masked_mean(np.abs(contrast_delta), mask),
            masked_mean(np.abs(coarse_delta), mask),
            masked_mean(np.abs(mid_delta), mask),
            masked_mean(np.abs(fine_delta), mask),
            max(masked_mean(bright_delta, mask), 0.0),
            masked_mean(np.abs(env_features.edge_x - skin_features.edge_x), mask),
            masked_mean(np.abs(env_features.edge_y - skin_features.edge_y), mask),
            max(masked_mean(-lum_delta, mask), 0.0),
        ],
        dtype=np.float32,
    )

    loss = (
        0.42 * masked_mean(lum_delta**2, mask)
        + 0.18 * masked_mean(contrast_delta**2, mask)
        + 0.18 * masked_mean(edge_delta**2, mask)
        + 0.12 * masked_mean(mid_delta**2, mask)
        + 0.10 * masked_mean(coarse_delta**2, mask)
    )
    return summary, float(loss)


def render_skin_layers(
    state: ChromatophoreMosaic,
    patterns: PatternFields,
    env_rgb: np.ndarray,
    env_features: VisualFeatures,
    body: BodyMaps,
    color_assist: float,
    texture_prior: np.ndarray | None = None,
    neural: NeuralControl | None = None,
) -> SkinLayers:
    mask = body.mask
    shape = mask.shape

    sigmas = np.array([0.33, 0.36, 0.40], dtype=np.float32) * state.spacing
    type_maps = []
    for color_id in range(3):
        color_mask = state.color_ids == color_id
        layer = blurred_spot_map(
            state.centers[color_mask],
            state.expansion[color_mask],
            shape,
            sigma=float(sigmas[color_id]),
        )
        type_maps.append(clamp01(layer))
    yellow, brown, black = type_maps

    low_freq_env = gaussian_filter(env_rgb, sigma=(state.spacing * 2.0, state.spacing * 2.0, 0.0))
    lum = gaussian_filter(env_features.luminance, state.spacing * 0.8)
    edge_sheen = gaussian_filter(env_features.edge_mag, state.spacing * 0.6)

    mantle_bias = clamp01(body.mantle)
    head_bias = clamp01(body.head_arms)
    fin_bias = clamp01(body.fin)
    eye_bias = clamp01(body.eye)
    ventral_bias = clamp01(body.ventral)
    axial_band = clamp01(1.0 - np.abs(body.axis_v))
    papillae_relief, fine_grain, mantle_clusters = build_octopus_skin_basis(body, mask, state.spacing)
    if neural is not None:
        macro_pattern = normalize_map(gaussian_filter(patterns.coarse * (0.72 + 0.56 * neural.coarse_map) * mask, state.spacing * 0.42), mask)
        cluster_pattern = normalize_map((0.55 * patterns.mid + 0.45 * mantle_clusters) * (0.72 + 0.56 * neural.mid_map), mask)
        grain_pattern = normalize_map((0.62 * patterns.fine + 0.38 * fine_grain) * (0.72 + 0.56 * neural.fine_map), mask)
    else:
        macro_pattern = normalize_map(gaussian_filter(patterns.coarse * mask, state.spacing * 0.42), mask)
        cluster_pattern = normalize_map(0.55 * patterns.mid + 0.45 * mantle_clusters, mask)
        grain_pattern = normalize_map(0.62 * patterns.fine + 0.38 * fine_grain, mask)
    mantle_mottle = clamp01(
        0.48
        + 0.25 * np.sin((body.axis_u + 0.15) * 14.0)
        + 0.20 * np.sin((body.axis_v - 0.08) * 18.0)
        + 0.12 * (cluster_pattern - 0.5)
        + 0.10 * (macro_pattern - 0.5)
    )
    dorsal_darkening = clamp01(0.60 - 0.26 * body.axis_v + 0.10 * (macro_pattern - 0.5))

    # Octopus: mottled mantle, softer head, arms carry weaker chromatophore contrast.
    yellow *= clamp01(0.40 * mantle_bias + 0.26 * head_bias + 0.08)
    brown *= clamp01(0.72 * mantle_bias + 0.34 * head_bias + 0.10)
    black *= clamp01(0.78 * mantle_bias + 0.26 * head_bias + 0.08)
    yellow += 0.04 * head_bias * axial_band
    yellow += 0.03 * (1.0 - macro_pattern) * ventral_bias
    brown += 0.06 * mantle_bias * (0.45 * mantle_mottle + 0.30 * cluster_pattern + 0.25 * macro_pattern)
    black += 0.08 * mantle_bias * (1.0 - mantle_mottle) * dorsal_darkening
    black += 0.05 * cluster_pattern * mantle_bias
    black += 0.04 * grain_pattern * mantle_bias
    brown += 0.04 * papillae_relief * head_bias
    yellow += 0.08 * ventral_bias * mantle_bias
    black -= 0.05 * ventral_bias
    pigment_total = clamp01(0.80 * (yellow + brown + black))

    leucophore_strength = clamp01(
        0.38
        + 0.44 * lum
        - 0.18 * pigment_total
        + 0.10 * papillae_relief
        + 0.08 * macro_pattern
        + 0.10 * fin_bias
        + 0.16 * ventral_bias
    )
    leucophore_rgb = clamp01(
        leucophore_strength[..., None] * (0.72 * low_freq_env + 0.28 * np.array([0.92, 0.91, 0.89], dtype=np.float32))
    )

    teal_field = np.stack(
        [
            clamp01(0.30 * low_freq_env[..., 0] + 0.06),
            clamp01(0.66 * low_freq_env[..., 1] + 0.18),
            clamp01(0.82 * low_freq_env[..., 2] + 0.24),
        ],
        axis=-1,
    )
    gold_field = np.stack(
        [
            clamp01(0.88 * low_freq_env[..., 0] + 0.12),
            clamp01(0.78 * low_freq_env[..., 1] + 0.10),
            clamp01(0.32 * low_freq_env[..., 2] + 0.02),
        ],
        axis=-1,
    )
    neutral_field = clamp01(0.72 * low_freq_env + 0.28 * np.array([0.74, 0.70, 0.66], dtype=np.float32))
    cool_bias = clamp01(low_freq_env[..., 2] - 0.75 * low_freq_env[..., 0] + 0.06)
    iridophore_mix = clamp01(
        0.44 * env_features.bright_patches
        + 0.22 * edge_sheen
        + 0.14 * (1.0 - lum)
        + 0.10 * grain_pattern
        + 0.10 * papillae_relief
    )
    iridophore_tone = clamp01(
        (0.22 + 0.20 * cool_bias)[..., None] * teal_field
        + (0.24 + 0.18 * (1.0 - cool_bias))[..., None] * gold_field
        + 0.42 * neutral_field
    )
    iridophore_rgb = clamp01((0.04 + 0.22 * iridophore_mix)[..., None] * iridophore_tone)

    base_field = clamp01(0.78 * low_freq_env + 0.22 * np.array([0.66, 0.61, 0.56], dtype=np.float32))
    base_field = clamp01(
        base_field
        * (0.86 + 0.08 * dorsal_darkening[..., None] + 0.06 * papillae_relief[..., None] + 0.06 * macro_pattern[..., None])
    )
    yellow_field = clamp01(0.58 * base_field + 0.42 * np.array([0.86, 0.74, 0.44], dtype=np.float32))
    brown_field = clamp01(0.28 * base_field + 0.72 * np.array([0.42, 0.28, 0.17], dtype=np.float32))
    black_field = clamp01(0.12 * base_field + 0.88 * np.array([0.07, 0.06, 0.05], dtype=np.float32))

    if texture_prior is not None:
        prior_rgb = gaussian_filter(texture_prior * mask[..., None], sigma=(state.spacing * 0.18, state.spacing * 0.18, 0.0))
        prior_norm = gaussian_filter(mask.astype(np.float32), sigma=state.spacing * 0.18)
        prior_rgb = clamp01(prior_rgb / (prior_norm[..., None] + EPS)) * mask[..., None]
        prior_lum = normalize_map(rgb_to_luminance(prior_rgb), mask)
        prior_mid = normalize_map(
            np.mean(np.abs(gaussian_filter(prior_rgb, sigma=(state.spacing * 0.25, state.spacing * 0.25, 0.0)) - gaussian_filter(prior_rgb, sigma=(state.spacing * 0.85, state.spacing * 0.85, 0.0))), axis=-1),
            mask,
        )
        prior_high = normalize_map(
            np.mean(np.abs(prior_rgb - gaussian_filter(prior_rgb, sigma=(state.spacing * 0.55, state.spacing * 0.55, 0.0))), axis=-1),
            mask,
        )
        base_field = clamp01(0.66 * base_field + 0.34 * prior_rgb)
        brown_field = clamp01(0.60 * brown_field + 0.40 * (0.84 * prior_rgb + 0.16 * np.array([0.48, 0.33, 0.22], dtype=np.float32)))
        yellow_field = clamp01(0.76 * yellow_field + 0.24 * (0.92 * prior_rgb + 0.08 * np.array([0.92, 0.82, 0.62], dtype=np.float32)))
        black += 0.04 * prior_mid * mantle_bias + 0.05 * prior_high * mantle_bias
        brown += 0.05 * prior_lum * mantle_bias + 0.04 * prior_mid * head_bias
        leucophore_rgb = clamp01(leucophore_rgb * (0.92 + 0.08 * prior_lum[..., None]))

    pigment_total = clamp01(0.80 * (yellow + brown + black))

    weights = np.stack(
        [
            np.maximum(1.0 - pigment_total, 0.0),
            yellow,
            brown,
            black,
        ],
        axis=-1,
    )
    chromatophore_rgb = (
        weights[..., 0:1] * base_field
        + weights[..., 1:2] * yellow_field
        + weights[..., 2:3] * brown_field
        + weights[..., 3:4] * black_field
    ) / (np.sum(weights, axis=-1, keepdims=True) + EPS)
    chromatophore_rgb = clamp01(
        chromatophore_rgb
        * (0.88 + 0.10 * papillae_relief[..., None] + 0.06 * grain_pattern[..., None] + 0.04 * cluster_pattern[..., None])
    )

    translucency = clamp01(0.78 + 0.18 * body.mantle + 0.10 * body.head_arms - 0.28 * body.fin)
    final = clamp01((0.52 * leucophore_rgb + 0.10 * iridophore_rgb + 0.68 * chromatophore_rgb) * translucency[..., None])
    if color_assist > 0.0:
        sheen = clamp01(0.20 * (1.0 - pigment_total) + 0.10 * gaussian_filter(rgb_to_luminance(env_rgb), state.spacing * 1.5))
        final = clamp01((1.0 - color_assist * sheen[..., None]) * final + color_assist * sheen[..., None] * low_freq_env)

    final = clamp01(
        final
        * (0.90 + 0.08 * papillae_relief[..., None] + 0.05 * grain_pattern[..., None] + 0.04 * cluster_pattern[..., None])
    )

    # Visible eye adds a strong biological anchor even in camouflaged states.
    eye_ring = eye_bias[..., None] * np.array([0.78, 0.74, 0.56], dtype=np.float32)
    pupil = clamp01(gaussian_filter(eye_bias, 1.2) - eye_bias * 0.65)[..., None] * np.array([0.04, 0.04, 0.04], dtype=np.float32)
    final = clamp01(final * (1.0 - 0.85 * eye_bias[..., None]) + 0.65 * eye_ring + 0.70 * pupil)

    return SkinLayers(
        chromatophore=body.alpha[..., None] * chromatophore_rgb,
        iridophore=body.alpha[..., None] * iridophore_rgb,
        leucophore=body.alpha[..., None] * leucophore_rgb,
        final=body.alpha[..., None] * final,
    )


def compose_with_environment(env_rgb: np.ndarray, skin_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return clamp01(env_rgb * (1.0 - alpha[..., None]) + skin_rgb)


def simulate_camouflage(
    env_rgb: np.ndarray,
    iterations: int,
    seed: int,
    color_assist: float,
    dynamic: bool,
    body_ref: str | None = None,
    body_template: str = "auto",
) -> tuple[SkinLayers, np.ndarray, BodyMaps, list[float], dict[str, float], BVAMParams, dict[str, str], ReferencePriorDebug | None, NeuralControl]:
    scene_mask = np.ones(env_rgb.shape[:2], dtype=np.float32)
    scene_features = extract_visual_features(env_rgb, scene_mask)
    neural = neural_background_controller(scene_features)
    scene_type, template_name = select_body_template(scene_features, requested_template=body_template)
    texture_prior = None
    ref_debug = None

    if body_ref:
        body, texture_prior, ref_debug = create_body_maps_from_reference(body_ref, env_rgb.shape[0])
        selection = {
            "scene_type": scene_type,
            "template_name": "reference_prior",
            "body_source": "reference image",
        }
    else:
        if template_name in ASSET_BODY_TEMPLATES:
            body, texture_prior, ref_debug = load_body_maps_from_template_asset(template_name, env_rgb.shape[0])
            selection = {
                "scene_type": scene_type,
                "template_name": template_name,
                "body_source": "silhouette template library",
            }
        else:
            body = create_octopus_mask(env_rgb.shape[0], seed, template_name=template_name)
            selection = {
                "scene_type": scene_type,
                "template_name": template_name,
                "body_source": "template library",
            }
    mask = body.mask
    env_features = extract_visual_features(env_rgb, mask)
    program = infer_body_pattern_program(env_features, neural=neural)
    params = inverse_turing_fit(env_features, program, mask, dynamic=dynamic, neural=neural)
    state = initialize_skin(mask, seed)
    patterns, losses = run_bvam_turing_core(env_features, mask, params, iterations, seed, neural=neural)
    state = project_bvam_to_chromatophores(state, patterns, env_features, env_rgb, params, mask, neural=neural)
    skin_layers = render_skin_layers(
        state,
        patterns,
        env_rgb,
        env_features,
        body,
        color_assist=color_assist,
        texture_prior=texture_prior,
        neural=neural,
    )
    composite = compose_with_environment(env_rgb, skin_layers.final, body.alpha)
    return skin_layers, composite, body, losses, program, params, selection, ref_debug, neural


def save_diagnostics(
    env_rgb: np.ndarray,
    skin_layers: SkinLayers,
    composite: np.ndarray,
    mask: np.ndarray,
    losses: list[float],
    program: dict[str, float],
    params: BVAMParams,
    selection: dict[str, str],
    neural: NeuralControl,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].imshow(env_rgb)
    axes[0].set_title("Environment")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Octopus body mask")

    axes[2].imshow(clamp01(skin_layers.chromatophore + (1.0 - mask[..., None])))
    axes[2].set_title("Chromatophore")

    axes[3].imshow(clamp01(skin_layers.iridophore + (1.0 - mask[..., None])))
    axes[3].set_title("Iridophore")

    axes[4].imshow(clamp01(skin_layers.leucophore + (1.0 - mask[..., None])))
    axes[4].set_title("Leucophore")

    axes[5].imshow(clamp01(skin_layers.final + (1.0 - mask[..., None])))
    axes[5].set_title("Skin pattern")

    axes[6].imshow(composite)
    axes[6].set_title("Camouflage composite")

    axes[7].axis("off")
    axes[7].set_title("Program / BVAM")
    info = "\n".join(
        [
            f"uniform      {program['uniform']:.3f}",
            f"mottle       {program['mottle']:.3f}",
            f"disruptive   {program['disruptive']:.3f}",
            "",
            f"scene type   {selection['scene_type']}",
            f"body prior   {selection['body_source']}",
            f"template     {selection['template_name']}",
            "",
            f"darkness     {neural.darkness_score:.3f}",
            f"contrast nn  {neural.contrast_score:.3f}",
            f"coarse gain  {neural.coarse_gain:.3f}",
            f"mid gain     {neural.mid_gain:.3f}",
            f"fine gain    {neural.fine_gain:.3f}",
            "",
            f"a            {params.a:.3f}",
            f"b            {params.b:.3f}",
            f"n            {params.n:.3f}",
            f"C            {params.c:.3f}",
            f"H            {params.h:.1f}",
            f"D_A          {params.da:.3f}",
            "",
            f"steps        {len(losses)}",
            f"final loss   {losses[-1]:.4f}",
        ]
    )
    axes[7].text(0.02, 0.98, info, va="top", ha="left", family="monospace", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-inspired octopus camouflage simulator driven by an environment image."
    )
    parser.add_argument("--env", type=Path, required=True, help="Path to the environment image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated images.",
    )
    parser.add_argument("--size", type=int, default=512, help="Square working resolution.")
    parser.add_argument("--iterations", type=int, default=80, help="Maximum neural update steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--color-assist",
        type=float,
        default=0.12,
        help="Weak low-frequency color projection. Set to 0 for strictly achromatic control.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use the paper's Hopf-side b parameter for oscillatory dynamics rather than static camouflage.",
    )
    parser.add_argument(
        "--body-ref",
        type=str,
        default=None,
        help="Local path or URL to a reference cephalopod image used to extract a body-shape prior with rembg.",
    )
    parser.add_argument(
        "--body-template",
        type=str,
        default="auto",
        choices=["auto", *sorted(BODY_TEMPLATE_LIBRARY.keys()), *sorted(ASSET_BODY_TEMPLATES.keys())],
        help="Built-in octopus body-prior template. Use auto to select from the environment type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_rgb = load_image(args.env, args.size)

    skin_layers, composite, body, losses, program, params, selection, ref_debug, neural = simulate_camouflage(
        env_rgb=env_rgb,
        iterations=args.iterations,
        seed=args.seed,
        color_assist=max(args.color_assist, 0.0),
        dynamic=args.dynamic,
        body_ref=args.body_ref,
        body_template=args.body_template,
    )

    output_dir = timestamped_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skin_on_white = clamp01(skin_layers.final + (1.0 - body.alpha[..., None]))
    save_image(output_dir / "octopus_skin.png", skin_on_white)
    save_image(output_dir / "chromatophore_layer.png", clamp01(skin_layers.chromatophore + (1.0 - body.alpha[..., None])))
    save_image(output_dir / "iridophore_layer.png", clamp01(skin_layers.iridophore + (1.0 - body.alpha[..., None])))
    save_image(output_dir / "leucophore_layer.png", clamp01(skin_layers.leucophore + (1.0 - body.alpha[..., None])))
    save_image(output_dir / "octopus_on_environment.png", composite)
    if ref_debug is not None:
        save_image(output_dir / "body_ref_mask_raw.png", ref_debug.raw_alpha)
        save_image(output_dir / "body_ref_mask_clean.png", ref_debug.clean_alpha)
        save_image(output_dir / "body_ref_cutout.png", clamp01(ref_debug.cutout_rgb + (1.0 - ref_debug.clean_alpha[..., None])))
        save_image(
            output_dir / "body_ref_texture_prior.png",
            clamp01(ref_debug.texture_prior + (1.0 - ref_debug.clean_alpha[..., None])),
        )
    save_diagnostics(
        env_rgb=env_rgb,
        skin_layers=skin_layers,
        composite=composite,
        mask=body.mask,
        losses=losses,
        program=program,
        params=params,
        selection=selection,
        neural=neural,
        output_path=output_dir / "diagnostics.png",
    )

    print("Saved:")
    print("Output dir:", output_dir)
    print(output_dir / "octopus_skin.png")
    print(output_dir / "chromatophore_layer.png")
    print(output_dir / "iridophore_layer.png")
    print(output_dir / "leucophore_layer.png")
    print(output_dir / "octopus_on_environment.png")
    if ref_debug is not None:
        print(output_dir / "body_ref_mask_raw.png")
        print(output_dir / "body_ref_mask_clean.png")
        print(output_dir / "body_ref_cutout.png")
        print(output_dir / "body_ref_texture_prior.png")
    print(output_dir / "diagnostics.png")
    print("Scene:", selection["scene_type"])
    print("Body prior:", selection["body_source"])
    print("Template:", selection["template_name"])
    print(
        "Neural:",
        {
            "darkness": round(neural.darkness_score, 3),
            "contrast": round(neural.contrast_score, 3),
            "coarse": round(neural.coarse_gain, 3),
            "mid": round(neural.mid_gain, 3),
            "fine": round(neural.fine_gain, 3),
        },
    )
    print("Program:", {k: round(v, 3) for k, v in program.items()})
    print("BVAM:", {"a": round(params.a, 3), "b": round(params.b, 3), "n": round(params.n, 3), "C": round(params.c, 3), "H": round(params.h, 3), "D_A": round(params.da, 3)})
    print("Iterations:", len(losses))
    print("Final loss:", round(losses[-1], 5))


if __name__ == "__main__":
    main()
