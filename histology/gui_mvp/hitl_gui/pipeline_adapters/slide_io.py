from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..domain import LoadedSlide
from .tool_bridge import proposal_crop_rect_overview_gui

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
_BACKEND_HINTS_CACHE: Optional[dict] = None


def _persistent_cache_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        root = Path(local_appdata) / "histology_gui"
    else:
        root = Path.home() / ".cache" / "histology_gui"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _backend_hints_path() -> Path:
    return _persistent_cache_root() / "backend_hints_v1.json"


def _proxy_cache_root() -> Path:
    root = _persistent_cache_root() / "proxy_bundles_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slide_identity(slide_path: Path) -> dict[str, object]:
    stat = slide_path.stat()
    return {
        "path": str(slide_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _slide_cache_key(slide_path: Path) -> str:
    resolved = str(slide_path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    stem = slide_path.stem.replace(";", "_")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{safe}_{digest}"


def _load_backend_hints() -> dict:
    global _BACKEND_HINTS_CACHE
    if _BACKEND_HINTS_CACHE is not None:
        return _BACKEND_HINTS_CACHE
    path = _backend_hints_path()
    if not path.exists():
        _BACKEND_HINTS_CACHE = {"slides": {}}
        return _BACKEND_HINTS_CACHE
    try:
        _BACKEND_HINTS_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _BACKEND_HINTS_CACHE = {"slides": {}}
    if not isinstance(_BACKEND_HINTS_CACHE, dict):
        _BACKEND_HINTS_CACHE = {"slides": {}}
    _BACKEND_HINTS_CACHE.setdefault("slides", {})
    return _BACKEND_HINTS_CACHE


def _save_backend_hints() -> None:
    path = _backend_hints_path()
    payload = _load_backend_hints()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _lookup_backend_hint(slide_path: Path) -> Optional[dict]:
    hints = _load_backend_hints().get("slides", {})
    key = str(slide_path.resolve())
    entry = hints.get(key)
    if not isinstance(entry, dict):
        return None
    ident = _slide_identity(slide_path)
    if int(entry.get("size_bytes", -1)) != ident["size_bytes"]:
        return None
    if int(entry.get("mtime_ns", -1)) != ident["mtime_ns"]:
        return None
    return entry


def _update_backend_hint(slide_path: Path, backend: str, fallback_reason: str | None = None) -> None:
    payload = _load_backend_hints()
    slides = payload.setdefault("slides", {})
    ident = _slide_identity(slide_path)
    slides[str(slide_path.resolve())] = {
        **ident,
        "preferred_backend": backend,
        "fallback_reason": fallback_reason,
        "updated_at_unix": int(time.time()),
    }
    _save_backend_hints()


def clear_backend_hint(slide_path: Path) -> None:
    payload = _load_backend_hints()
    slides = payload.setdefault("slides", {})
    slides.pop(str(slide_path.resolve()), None)
    _save_backend_hints()


def clear_proxy_cache(slide_path: Path) -> None:
    shutil.rmtree(_proxy_cache_dir(slide_path), ignore_errors=True)


@dataclass
class _TiffFileProxyHandle:
    tf: object
    page0_arr: object
    midres_arr: object
    full_w: int
    full_h: int
    midres_downsample: float
    overview_to_midres: float

    def close(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass


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


def _proxy_cache_dir(slide_path: Path) -> Path:
    return _proxy_cache_root() / _slide_cache_key(slide_path)


def _proxy_cache_manifest(slide_path: Path) -> Path:
    return _proxy_cache_dir(slide_path) / "proxy_manifest.json"


def _write_persistent_proxy_bundle(slide_path: Path, overview_rgb: np.ndarray, label_rgb: np.ndarray, metadata: dict) -> None:
    cache_dir = _proxy_cache_dir(slide_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overview_rgb.astype(np.uint8), mode="RGB").save(cache_dir / "overview_proxy.png", format="PNG", compress_level=1, optimize=False)
    Image.fromarray(label_rgb.astype(np.uint8), mode="RGB").save(cache_dir / "label_proxy.png", format="PNG", compress_level=1, optimize=False)
    (cache_dir / "proxy_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_persistent_proxy_bundle(slide_path: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    manifest_path = _proxy_cache_manifest(slide_path)
    overview_path = manifest_path.parent / "overview_proxy.png"
    label_path = manifest_path.parent / "label_proxy.png"
    if not (manifest_path.exists() and overview_path.exists() and label_path.exists()):
        return None
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ident = _slide_identity(slide_path)
    cache_ident = meta.get("source_identity", {})
    if not isinstance(cache_ident, dict):
        return None
    if int(cache_ident.get("size_bytes", -1)) != ident["size_bytes"]:
        return None
    if int(cache_ident.get("mtime_ns", -1)) != ident["mtime_ns"]:
        return None
    overview_rgb = np.asarray(Image.open(overview_path).convert("RGB"))
    label_rgb = np.asarray(Image.open(label_path).convert("RGB"))
    return overview_rgb, label_rgb, meta


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
    mpp_x = None
    mpp_y = None
    objective_power = None
    try:
        if "openslide.mpp-x" in slide.properties:
            mpp_x = float(slide.properties["openslide.mpp-x"])
        if "openslide.mpp-y" in slide.properties:
            mpp_y = float(slide.properties["openslide.mpp-y"])
        if "openslide.objective-power" in slide.properties:
            objective_power = float(slide.properties["openslide.objective-power"])
    except Exception:
        mpp_x = None
        mpp_y = None
        objective_power = None
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
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        objective_power=objective_power,
    )


def _read_with_tifffile_proxy(slide_path: Path, stain: str, fallback_reason: str) -> LoadedSlide:
    if tifffile is None:
        raise RuntimeError(f"OpenSlide failed and tifffile is unavailable.\nOriginal error: {fallback_reason}")

    cached = _read_persistent_proxy_bundle(slide_path)
    if cached is not None:
        overview_rgb, label_rgb, cached_meta = cached
        overview = Image.fromarray(overview_rgb.astype(np.uint8), mode="RGB")
        label_preview = Image.fromarray(label_rgb.astype(np.uint8), mode="RGB")
        full_w = int(cached_meta["full_w"])
        full_h = int(cached_meta["full_h"])
        midres_page_index = int(cached_meta["midres_page_index"])
        midres_downsample = float(cached_meta["midres_downsample"])
        overview_scale_from_midres = float(cached_meta["overview_scale_from_midres"])
        level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, midres_downsample * overview_scale_from_midres)
        level_dimensions = []
        for ds in level_downsamples:
            level_dimensions.append((max(1, int(round(full_w / ds))), max(1, int(round(full_h / ds)))))
        level_dimensions[-1] = (overview_rgb.shape[1], overview_rgb.shape[0])
        proxy_dir = _prepare_session_proxy_bundle(
            slide_path,
            overview_rgb,
            label_rgb,
            {
                **cached_meta,
                "source_path": str(slide_path),
                "backend": "tifffile_proxy_cached",
                "fallback_reason": fallback_reason,
                "session_temp_root": str(_SESSION_TEMP_ROOT),
            },
        )
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
            overview_size=(overview_rgb.shape[1], overview_rgb.shape[0]),
            level_dimensions=tuple(level_dimensions),
            level_downsamples=tuple(level_downsamples),
            backend="tifffile_proxy",
            mpp_x=None,
            mpp_y=None,
            objective_power=None,
            temp_proxy_dir=proxy_dir,
            fallback_reason=f"{fallback_reason} | proxy_cache_hit",
            tifffile_midres_page_index=midres_page_index,
            tifffile_midres_downsample=midres_downsample,
            tifffile_overview_scale_from_midres=overview_scale_from_midres,
        )

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
            "source_identity": _slide_identity(slide_path),
            "source_path": str(slide_path),
            "backend": "tifffile_proxy",
            "full_w": full_w,
            "full_h": full_h,
            "fallback_reason": fallback_reason,
            "midres_page_index": midres_page_index,
            "macro_page_index": macro_page_index,
            "midres_downsample": midres_downsample,
            "overview_scale_from_midres": overview_scale_from_midres,
            "session_temp_root": str(_SESSION_TEMP_ROOT),
        },
    )
    _write_persistent_proxy_bundle(
        slide_path,
        overview_rgb,
        np.asarray(label_preview),
        {
            "source_identity": _slide_identity(slide_path),
            "source_path": str(slide_path),
            "backend": "tifffile_proxy",
            "full_w": full_w,
            "full_h": full_h,
            "midres_page_index": midres_page_index,
            "macro_page_index": macro_page_index,
            "midres_downsample": midres_downsample,
            "overview_scale_from_midres": overview_scale_from_midres,
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
        mpp_x=None,
        mpp_y=None,
        objective_power=None,
        temp_proxy_dir=proxy_dir,
        fallback_reason=fallback_reason,
        tifffile_midres_page_index=midres_page_index,
        tifffile_midres_downsample=midres_downsample,
        tifffile_overview_scale_from_midres=overview_scale_from_midres,
    )


def load_slide_bundle(slide_path: Path, stain: str) -> LoadedSlide:
    hint = _lookup_backend_hint(slide_path)
    preferred_backend = str(hint.get("preferred_backend")) if hint else None
    if preferred_backend == "tifffile_proxy":
        reason = str(hint.get("fallback_reason") or "cached backend hint: tifffile_proxy")
        loaded = _read_with_tifffile_proxy(slide_path, stain, reason)
        loaded.fallback_reason = f"{reason} | backend_hint_cache"
        return loaded

    openslide_error: Optional[Exception] = None
    if openslide is not None:
        try:
            loaded = _read_with_openslide(slide_path, stain)
            _update_backend_hint(slide_path, "openslide", None)
            return loaded
        except Exception as exc:
            openslide_error = exc
    reason = str(openslide_error) if openslide_error is not None else "openslide unavailable"
    loaded = _read_with_tifffile_proxy(slide_path, stain, reason)
    _update_backend_hint(slide_path, "tifffile_proxy", reason)
    return loaded


def _overview_bbox_with_pad(
    loaded_slide: LoadedSlide,
    proposal,
    pad_ratio: float = 0.08,
    *,
    min_pad: int = 24,
) -> tuple[int, int, int, int]:
    pad = max(min_pad, int(round(max(proposal.w, proposal.h) * pad_ratio)))
    x1 = max(0, proposal.x - pad)
    y1 = max(0, proposal.y - pad)
    x2 = min(loaded_slide.overview_size[0], proposal.x + proposal.w + pad)
    y2 = min(loaded_slide.overview_size[1], proposal.y + proposal.h + pad)
    return x1, y1, x2, y2


def _overview_bbox_for_proposal(loaded_slide: LoadedSlide, proposal) -> tuple[int, int, int, int]:
    if loaded_slide.stain.lower() == "gallyas":
        return _overview_bbox_with_pad(loaded_slide, proposal, pad_ratio=0.03, min_pad=16)
    try:
        return proposal_crop_rect_overview_gui(loaded_slide, proposal)
    except Exception:
        return _overview_bbox_with_pad(loaded_slide, proposal, pad_ratio=0.08)


def effective_crop_rect_overview(loaded_slide: LoadedSlide, proposal) -> tuple[int, int, int, int]:
    return _overview_bbox_for_proposal(loaded_slide, proposal)


def effective_crop_bbox_level0(loaded_slide: LoadedSlide, proposal) -> tuple[int, int, int, int]:
    downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    x1, y1, x2, y2 = _overview_bbox_for_proposal(loaded_slide, proposal)
    x0 = int(round(x1 * downsample))
    y0 = int(round(y1 * downsample))
    w0 = min(int(round((x2 - x1) * downsample)), loaded_slide.level_dimensions[0][0] - x0)
    h0 = min(int(round((y2 - y1) * downsample)), loaded_slide.level_dimensions[0][1] - y0)
    return x0, y0, w0, h0


def _level0_bbox_for_proposal(slide: openslide.OpenSlide, loaded_slide: LoadedSlide, proposal, pad_ratio: float = 0.08) -> tuple[int, int, int, int]:
    downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    x1, y1, x2, y2 = _overview_bbox_for_proposal(loaded_slide, proposal)
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
    x1_ov, y1_ov, x2_ov, y2_ov = _overview_bbox_for_proposal(loaded_slide, proposal)

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


def _extract_crop_with_tifffile_handle(
    handle: _TiffFileProxyHandle,
    loaded_slide: LoadedSlide,
    proposal,
    crop_level: int = 4,
) -> np.ndarray:
    crop_level = min(crop_level, loaded_slide.level_count - 1)
    target_downsample = float(loaded_slide.level_downsamples[crop_level])
    x1_ov, y1_ov, x2_ov, y2_ov = _overview_bbox_for_proposal(loaded_slide, proposal)

    if target_downsample >= handle.midres_downsample:
        x1 = int(round(x1_ov * handle.overview_to_midres))
        y1 = int(round(y1_ov * handle.overview_to_midres))
        x2 = int(round(x2_ov * handle.overview_to_midres))
        y2 = int(round(y2_ov * handle.overview_to_midres))
        x1 = max(0, min(handle.midres_arr.shape[1] - 1, x1))
        y1 = max(0, min(handle.midres_arr.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(handle.midres_arr.shape[1], x2))
        y2 = max(y1 + 1, min(handle.midres_arr.shape[0], y2))
        crop = np.asarray(handle.midres_arr[y1:y2, x1:x2, :], dtype=np.uint8)
        if target_downsample > handle.midres_downsample:
            scale = handle.midres_downsample / target_downsample
            out_w = max(1, int(round(crop.shape[1] * scale)))
            out_h = max(1, int(round(crop.shape[0] * scale)))
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return crop

    overview_downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    x0 = int(round(x1_ov * overview_downsample))
    y0 = int(round(y1_ov * overview_downsample))
    x1 = max(0, min(handle.full_w - 1, x0))
    y1 = max(0, min(handle.full_h - 1, y0))
    x2 = max(x1 + 1, min(handle.full_w, int(round(x2_ov * overview_downsample))))
    y2 = max(y1 + 1, min(handle.full_h, int(round(y2_ov * overview_downsample))))
    crop0 = np.asarray(handle.page0_arr[y1:y2, x1:x2, :], dtype=np.uint8)
    out_w = max(1, int(round((x2 - x1) / target_downsample)))
    out_h = max(1, int(round((y2 - y1) / target_downsample)))
    return cv2.resize(crop0, (out_w, out_h), interpolation=cv2.INTER_AREA)


def extract_crop_for_preview(loaded_slide: LoadedSlide, proposal, crop_level: int = 4, slide_handle=None) -> np.ndarray:
    if loaded_slide.backend == "tifffile_proxy":
        if isinstance(slide_handle, _TiffFileProxyHandle):
            return _extract_crop_with_tifffile_handle(slide_handle, loaded_slide, proposal, crop_level=crop_level)
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
    if loaded_slide.backend == "openslide":
        if openslide is None:
            return None
        return openslide.OpenSlide(str(loaded_slide.slide_path))
    if loaded_slide.backend == "tifffile_proxy":
        if tifffile is None or zarr is None:
            return None
        if (
            loaded_slide.tifffile_midres_page_index is None
            or loaded_slide.tifffile_midres_downsample is None
            or loaded_slide.tifffile_overview_scale_from_midres is None
        ):
            return None
        tf = tifffile.TiffFile(str(loaded_slide.slide_path))
        page0 = tf.pages[0]
        page0_arr = zarr.open(page0.aszarr(), mode="r")
        midres_arr = zarr.open(tf.pages[loaded_slide.tifffile_midres_page_index].aszarr(), mode="r")
        return _TiffFileProxyHandle(
            tf=tf,
            page0_arr=page0_arr,
            midres_arr=midres_arr,
            full_w=int(page0.shape[1]),
            full_h=int(page0.shape[0]),
            midres_downsample=float(loaded_slide.tifffile_midres_downsample),
            overview_to_midres=float(loaded_slide.tifffile_overview_scale_from_midres),
        )
    return None


def write_png_lossless_fast(path: Path, array: np.ndarray, mode: str) -> None:
    Image.fromarray(array.astype(np.uint8), mode=mode).save(
        path,
        format="PNG",
        compress_level=1,
        optimize=False,
    )
