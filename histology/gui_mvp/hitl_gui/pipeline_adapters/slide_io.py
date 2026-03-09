from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..domain import LoadedSlide

try:
    import openslide
except Exception:  # pragma: no cover - runtime environment dependent
    openslide = None

try:
    import tifffile
except Exception:  # pragma: no cover - runtime environment dependent
    tifffile = None

try:
    import zarr
except Exception:  # pragma: no cover - runtime environment dependent
    zarr = None


_SESSION_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="histology_gui_session_"))


def cleanup_session_temp_root() -> None:
    shutil.rmtree(_SESSION_TEMP_ROOT, ignore_errors=True)


atexit.register(cleanup_session_temp_root)


def openslide_available() -> bool:
    return openslide is not None


def crop_label_from_macro(macro: Image.Image) -> Image.Image:
    w, h = macro.size
    label_w = max(1, int(round(w * 0.28)))
    return macro.crop((0, 0, label_w, h)).convert("RGB")


def _prepare_session_proxy_bundle(slide_path: Path, overview_rgb: np.ndarray, label_rgb: np.ndarray, metadata: dict) -> Path:
    proxy_dir = _SESSION_TEMP_ROOT / slide_path.stem.replace(";", "_")
    proxy_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overview_rgb.astype(np.uint8), mode="RGB").save(proxy_dir / "overview_proxy.png", format="PNG", compress_level=1, optimize=False)
    Image.fromarray(label_rgb.astype(np.uint8), mode="RGB").save(proxy_dir / "label_proxy.png", format="PNG", compress_level=1, optimize=False)
    (proxy_dir / "proxy_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return proxy_dir


def _read_with_openslide(slide_path: Path, stain: str) -> LoadedSlide:
    if openslide is None:
        raise RuntimeError("openslide is not available in the current Python environment")
    slide = openslide.OpenSlide(str(slide_path))
    overview_level = slide.level_count - 1
    w, h = slide.level_dimensions[overview_level]
    overview = slide.read_region((0, 0), overview_level, (w, h)).convert("RGB")
    if "macro" in slide.associated_images:
        macro = slide.associated_images["macro"].convert("RGB")
        label_preview = crop_label_from_macro(macro)
    else:
        label_preview = Image.new("RGB", (256, 128), (32, 32, 32))
    return LoadedSlide(
        slide_path=slide_path,
        slide_name=slide_path.name,
        stain=stain,
        expected_labels=[],
        label_preview=label_preview,
        overview=overview,
        proposals=[],
        level_count=slide.level_count,
        overview_level=overview_level,
        overview_size=slide.level_dimensions[overview_level],
        level_dimensions=tuple(slide.level_dimensions),
        level_downsamples=tuple(float(x) for x in slide.level_downsamples),
        backend="openslide",
    )


def _read_with_tifffile_proxy(slide_path: Path, stain: str, fallback_reason: str) -> LoadedSlide:
    if tifffile is None:
        raise RuntimeError(f"OpenSlide failed and tifffile is unavailable.\nOriginal error: {fallback_reason}")

    with tifffile.TiffFile(str(slide_path)) as tf:
        if len(tf.pages) < 3:
            raise RuntimeError(f"tifffile fallback requires at least 3 pages, found {len(tf.pages)}")
        page0 = tf.pages[0]
        midres_page_index = 1
        macro_page_index = 2
        full_h, full_w = page0.shape[:2]

        midres_rgb = tf.pages[midres_page_index].asarray()
        macro_rgb = tf.pages[macro_page_index].asarray()
        if macro_rgb.ndim == 2:
            macro_rgb = np.stack([macro_rgb] * 3, axis=-1)

    midres_downsample = float(full_w) / float(midres_rgb.shape[1])
    overview_scale_from_midres = 8.0
    overview_w = max(1, int(round(midres_rgb.shape[1] / overview_scale_from_midres)))
    overview_h = max(1, int(round(midres_rgb.shape[0] / overview_scale_from_midres)))
    overview_rgb = cv2.resize(midres_rgb, (overview_w, overview_h), interpolation=cv2.INTER_AREA)
    overview = Image.fromarray(overview_rgb.astype(np.uint8), mode="RGB")
    label_preview = crop_label_from_macro(Image.fromarray(macro_rgb.astype(np.uint8), mode="RGB"))
    proxy_dir = _prepare_session_proxy_bundle(
        slide_path,
        overview_rgb,
        np.asarray(label_preview),
        {
            "source_path": str(slide_path),
            "backend": "tifffile_proxy",
            "fallback_reason": fallback_reason,
            "midres_page_index": midres_page_index,
            "macro_page_index": macro_page_index,
            "midres_downsample": midres_downsample,
            "overview_scale_from_midres": overview_scale_from_midres,
            "session_temp_root": str(_SESSION_TEMP_ROOT),
        },
    )
    level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, midres_downsample * overview_scale_from_midres)
    level_dimensions = []
    for ds in level_downsamples:
        level_dimensions.append((max(1, int(round(full_w / ds))), max(1, int(round(full_h / ds)))))
    level_dimensions[-1] = (overview_w, overview_h)
    return LoadedSlide(
        slide_path=slide_path,
        slide_name=slide_path.name,
        stain=stain,
        expected_labels=[],
        label_preview=label_preview,
        overview=overview,
        proposals=[],
        level_count=len(level_downsamples),
        overview_level=len(level_downsamples) - 1,
        overview_size=(overview_w, overview_h),
        level_dimensions=tuple(level_dimensions),
        level_downsamples=tuple(level_downsamples),
        backend="tifffile_proxy",
        temp_proxy_dir=proxy_dir,
        fallback_reason=fallback_reason,
        tifffile_midres_page_index=midres_page_index,
        tifffile_midres_downsample=midres_downsample,
        tifffile_overview_scale_from_midres=overview_scale_from_midres,
    )


def load_slide_bundle(slide_path: Path, stain: str) -> LoadedSlide:
    openslide_error: Optional[Exception] = None
    if openslide is not None:
        try:
            return _read_with_openslide(slide_path, stain)
        except Exception as exc:
            openslide_error = exc
    reason = str(openslide_error) if openslide_error is not None else "openslide unavailable"
    return _read_with_tifffile_proxy(slide_path, stain, reason)


def _overview_bbox_with_pad(loaded_slide: LoadedSlide, proposal, pad_ratio: float = 0.08) -> tuple[int, int, int, int]:
    pad = max(24, int(round(max(proposal.w, proposal.h) * pad_ratio)))
    x1 = max(0, proposal.x - pad)
    y1 = max(0, proposal.y - pad)
    x2 = min(loaded_slide.overview_size[0], proposal.x + proposal.w + pad)
    y2 = min(loaded_slide.overview_size[1], proposal.y + proposal.h + pad)
    return x1, y1, x2, y2


def _level0_bbox_for_proposal(slide: openslide.OpenSlide, loaded_slide: LoadedSlide, proposal, pad_ratio: float = 0.08) -> tuple[int, int, int, int]:
    downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    x1, y1, x2, y2 = _overview_bbox_with_pad(loaded_slide, proposal, pad_ratio=pad_ratio)
    x0 = int(round(x1 * downsample))
    y0 = int(round(y1 * downsample))
    w0 = min(int(round((x2 - x1) * downsample)), slide.dimensions[0] - x0)
    h0 = min(int(round((y2 - y1) * downsample)), slide.dimensions[1] - y0)
    return x0, y0, w0, h0


def extract_crop_from_handle(slide_handle, loaded_slide: LoadedSlide, proposal, crop_level: int = 4) -> np.ndarray:
    crop_level = min(crop_level, slide_handle.level_count - 1)
    x0, y0, w0, h0 = _level0_bbox_for_proposal(slide_handle, loaded_slide, proposal)
    downsample = float(slide_handle.level_downsamples[crop_level])
    out_w = max(1, int(round(w0 / downsample)))
    out_h = max(1, int(round(h0 / downsample)))
    return np.asarray(slide_handle.read_region((x0, y0), crop_level, (out_w, out_h)).convert("RGB"))


def _extract_crop_with_tifffile(loaded_slide: LoadedSlide, proposal, crop_level: int = 4) -> np.ndarray:
    if tifffile is None or zarr is None:
        raise RuntimeError("tifffile fallback requires both tifffile and zarr")
    if loaded_slide.tifffile_midres_page_index is None or loaded_slide.tifffile_midres_downsample is None or loaded_slide.tifffile_overview_scale_from_midres is None:
        raise RuntimeError("Missing tifffile fallback metadata")

    crop_level = min(crop_level, loaded_slide.level_count - 1)
    target_downsample = float(loaded_slide.level_downsamples[crop_level])
    midres_downsample = float(loaded_slide.tifffile_midres_downsample)
    overview_to_midres = float(loaded_slide.tifffile_overview_scale_from_midres)
    x1_ov, y1_ov, x2_ov, y2_ov = _overview_bbox_with_pad(loaded_slide, proposal)

    with tifffile.TiffFile(str(loaded_slide.slide_path)) as tf:
        if target_downsample >= midres_downsample:
            arr = zarr.open(tf.pages[loaded_slide.tifffile_midres_page_index].aszarr(), mode="r")
            x1 = int(round(x1_ov * overview_to_midres))
            y1 = int(round(y1_ov * overview_to_midres))
            x2 = int(round(x2_ov * overview_to_midres))
            y2 = int(round(y2_ov * overview_to_midres))
            x1 = max(0, min(arr.shape[1] - 1, x1))
            y1 = max(0, min(arr.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(arr.shape[1], x2))
            y2 = max(y1 + 1, min(arr.shape[0], y2))
            crop = np.asarray(arr[y1:y2, x1:x2, :], dtype=np.uint8)
            if target_downsample > midres_downsample:
                scale = midres_downsample / target_downsample
                out_w = max(1, int(round(crop.shape[1] * scale)))
                out_h = max(1, int(round(crop.shape[0] * scale)))
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            return crop

        page0 = tf.pages[0]
        arr0 = zarr.open(page0.aszarr(), mode="r")
        full_w = page0.shape[1]
        full_h = page0.shape[0]
        overview_downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
        x0 = int(round(x1_ov * overview_downsample))
        y0 = int(round(y1_ov * overview_downsample))
        x1 = max(0, min(full_w - 1, x0))
        y1 = max(0, min(full_h - 1, y0))
        x2 = max(x1 + 1, min(full_w, int(round(x2_ov * overview_downsample))))
        y2 = max(y1 + 1, min(full_h, int(round(y2_ov * overview_downsample))))
        crop0 = np.asarray(arr0[y1:y2, x1:x2, :], dtype=np.uint8)
        out_w = max(1, int(round((x2 - x1) / target_downsample)))
        out_h = max(1, int(round((y2 - y1) / target_downsample)))
        return cv2.resize(crop0, (out_w, out_h), interpolation=cv2.INTER_AREA)


def extract_crop_for_preview(loaded_slide: LoadedSlide, proposal, crop_level: int = 4, slide_handle=None) -> np.ndarray:
    if loaded_slide.backend == "tifffile_proxy":
        return _extract_crop_with_tifffile(loaded_slide, proposal, crop_level=crop_level)
    if slide_handle is not None:
        return extract_crop_from_handle(slide_handle, loaded_slide, proposal, crop_level=crop_level)
    if openslide is None:
        raise RuntimeError("openslide is not available in the current Python environment")
    slide = openslide.OpenSlide(str(loaded_slide.slide_path))
    try:
        return extract_crop_from_handle(slide, loaded_slide, proposal, crop_level=crop_level)
    finally:
        slide.close()


def open_slide_handle(loaded_slide: LoadedSlide):
    if loaded_slide.backend != "openslide" or openslide is None:
        return None
    return openslide.OpenSlide(str(loaded_slide.slide_path))


def write_png_lossless_fast(path: Path, array: np.ndarray, mode: str) -> None:
    Image.fromarray(array.astype(np.uint8), mode=mode).save(
        path,
        format="PNG",
        compress_level=1,
        optimize=False,
    )
