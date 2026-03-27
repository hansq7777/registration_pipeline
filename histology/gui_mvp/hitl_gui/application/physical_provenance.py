from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ..domain import ProposalBox
from ..pipeline_adapters.slide_io import cleanup_session_temp_root, load_slide_bundle

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None

PHYSICAL_PROVENANCE_VERSION = "physical_provenance_v1"

# Observed cohort-wide values from the current Nanozoomer collections.
DEFAULT_STAIN_MPP: dict[str, tuple[float, float]] = {
    "gallyas": (0.225073148773, 0.225073148773),
    "nissl": (0.450166561628, 0.450146297547),
}

SECTION_UID_SPLIT_RE = re.compile(
    r"^(?P<stain>[a-zA-Z]+)_(?P<sample>\d+)_(?P<section>\d+)__(?P<slide_hint>.+)__r(?P<rank>\d+)$"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "identity_method": "path_size_mtime",
        "path": str(path),
        "source_slide_checksum": None,
        "size_bytes": int(stat.st_size),
        "mtime_unix_sec": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_crop_shape_hw(crop_path: Path) -> tuple[int, int]:
    with Image.open(crop_path) as im:
        return int(im.height), int(im.width)


def _section_uid_parts(section_uid: str | None) -> dict[str, Any] | None:
    if not section_uid:
        return None
    match = SECTION_UID_SPLIT_RE.match(section_uid)
    if match is None:
        return None
    return {
        "stain": match.group("stain").lower(),
        "sample_id": match.group("sample"),
        "section_id": int(match.group("section")),
        "slide_hint": match.group("slide_hint"),
        "proposal_rank": int(match.group("rank")),
    }


def _proposal_from_metadata(metadata: dict[str, Any], section_label: str, section_stain: str) -> ProposalBox:
    bbox = dict(metadata.get("bbox_overview") or {})
    if not all(k in bbox for k in ("x", "y", "w", "h")):
        raise ValueError("bbox_overview is missing from section metadata")
    return ProposalBox(
        label=str(metadata.get("label") or section_label),
        stain=str(metadata.get("stain") or section_stain),
        sample_id=str(metadata.get("sample_id") or ""),
        section_id=int(metadata.get("section_id") or 0),
        proposal_rank=int(metadata.get("proposal_rank") or 0),
        x=int(bbox["x"]),
        y=int(bbox["y"]),
        w=int(bbox["w"]),
        h=int(bbox["h"]),
    )


def _provenance_from_existing_metadata(metadata: dict[str, Any], stain: str) -> dict[str, Any] | None:
    if not (
        isinstance(metadata.get("source_slide"), dict)
        and isinstance(metadata.get("source_slide_identity"), dict)
        and isinstance(metadata.get("crop_bbox_level0"), dict)
        and isinstance(metadata.get("export_canvas"), dict)
        and isinstance(metadata.get("canvas_to_slide_level0"), dict)
    ):
        return None

    source_slide = dict(metadata.get("source_slide") or {})
    mpp_x = source_slide.get("mpp_x")
    mpp_y = source_slide.get("mpp_y")
    mpp_method = "slide_header"
    if mpp_x is None or mpp_y is None:
        default_mpp = DEFAULT_STAIN_MPP.get(stain)
        if default_mpp is None:
            return None
        mpp_x = float(default_mpp[0])
        mpp_y = float(default_mpp[1])
        source_slide = {**source_slide, "mpp_x": float(mpp_x), "mpp_y": float(mpp_y)}
        mpp_method = "stain_default"

    canvas_to_slide_level0 = dict(metadata.get("canvas_to_slide_level0") or {})
    scale = dict(canvas_to_slide_level0.get("scale_level0_per_canvas_px") or {})
    scale_x = float(scale.get("x") or 0.0)
    scale_y = float(scale.get("y") or 0.0)
    if scale_x <= 0.0 or scale_y <= 0.0:
        return None

    proposal_bbox = metadata.get("proposal_bbox_overview_xywh") or metadata.get("bbox_overview")
    crop_bbox_um = metadata.get("crop_bbox_level0_um_relative_to_slide_origin")
    if not isinstance(crop_bbox_um, dict):
        crop_xywh = dict((metadata.get("crop_bbox_level0") or {}).get("xywh") or {})
        if all(k in crop_xywh for k in ("x", "y", "w", "h")):
            crop_bbox_um = {
                "x_um": float(crop_xywh["x"]) * float(mpp_x),
                "y_um": float(crop_xywh["y"]) * float(mpp_y),
                "w_um": float(crop_xywh["w"]) * float(mpp_x),
                "h_um": float(crop_xywh["h"]) * float(mpp_y),
            }
    canvas_um = metadata.get("canvas_to_slide_um_per_px")
    if not isinstance(canvas_um, dict):
        canvas_um = {
            "x_um_per_px": float(scale_x * float(mpp_x)),
            "y_um_per_px": float(scale_y * float(mpp_y)),
        }

    return {
        "version": PHYSICAL_PROVENANCE_VERSION,
        "recovered_at_utc": _utc_now_iso(),
        "recovery_method": "legacy_metadata_backfill",
        "slide_resolution_method": "existing_metadata",
        "mpp_recovery_method": mpp_method,
        "source_slide_identity": metadata["source_slide_identity"],
        "source_slide": source_slide,
        "proposal_bbox_overview_xywh": proposal_bbox,
        "crop_bbox_level0": metadata["crop_bbox_level0"],
        "export_canvas": metadata["export_canvas"],
        "canvas_to_slide_level0": metadata["canvas_to_slide_level0"],
        "crop_bbox_level0_um_relative_to_slide_origin": crop_bbox_um,
        "canvas_to_slide_um_per_px": canvas_um,
        "physical_calibration_available": True,
    }


def _parse_slide_groups(slide_stem: str) -> tuple[str, list[tuple[str, set[int]]]] | None:
    if "_" not in slide_stem:
        return None
    stain, rest = slide_stem.split("_", 1)
    stain = stain.lower()
    groups: list[tuple[str, set[int]]] = []
    for raw_group in rest.split(";"):
        raw_group = raw_group.strip()
        if not raw_group or "_" not in raw_group:
            return None
        sample_id, spec = raw_group.split("_", 1)
        numbers = [int(x) for x in spec.split("-") if x.strip()]
        if not numbers:
            return None
        if len(numbers) == 2 and numbers[1] >= numbers[0]:
            step = 6 if (numbers[1] - numbers[0]) % 6 == 0 else 1
            sections = set(range(numbers[0], numbers[1] + 1, step))
        else:
            sections = set(numbers)
        groups.append((sample_id, sections))
    return stain, groups


def _build_slide_lookup(slide_dir: Path, stain: str) -> dict[tuple[str, int], list[Path]]:
    lookup: dict[tuple[str, int], list[Path]] = {}
    for path in sorted(slide_dir.glob(f"{stain.lower()}_*.ndpi")):
        parsed = _parse_slide_groups(path.stem)
        if parsed is None:
            continue
        parsed_stain, groups = parsed
        if parsed_stain != stain.lower():
            continue
        for sample_id, section_ids in groups:
            for section_id in section_ids:
                lookup.setdefault((sample_id, int(section_id)), []).append(path)
    return lookup


def _load_header_only_slide_geometry(slide_path: Path) -> dict[str, Any]:
    if openslide is None:
        raise RuntimeError("openslide unavailable")
    slide = openslide.OpenSlide(str(slide_path))
    try:
        mpp_x = float(slide.properties["openslide.mpp-x"]) if "openslide.mpp-x" in slide.properties else None
        mpp_y = float(slide.properties["openslide.mpp-y"]) if "openslide.mpp-y" in slide.properties else None
    except Exception:
        mpp_x = None
        mpp_y = None
    return {
        "backend": "openslide_header_only",
        "slide_name": slide_path.name,
        "overview_level": slide.level_count - 1,
        "overview_size": tuple(int(x) for x in slide.level_dimensions[slide.level_count - 1]),
        "level_dimensions": tuple((int(w), int(h)) for (w, h) in slide.level_dimensions),
        "level_downsamples": tuple(float(x) for x in slide.level_downsamples),
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "objective_power": slide.properties.get("openslide.objective-power"),
    }


def _load_slide_geometry(slide_path: Path, stain: str) -> dict[str, Any]:
    try:
        return _load_header_only_slide_geometry(slide_path)
    except Exception:
        loaded = load_slide_bundle(slide_path, stain)
        return {
            "backend": loaded.backend,
            "slide_name": loaded.slide_name,
            "overview_level": int(loaded.overview_level),
            "overview_size": tuple(int(x) for x in loaded.overview_size),
            "level_dimensions": tuple((int(w), int(h)) for (w, h) in loaded.level_dimensions),
            "level_downsamples": tuple(float(x) for x in loaded.level_downsamples),
            "mpp_x": loaded.mpp_x,
            "mpp_y": loaded.mpp_y,
            "objective_power": loaded.objective_power,
            "fallback_reason": loaded.fallback_reason or None,
        }


def _recover_gallyas_bbox_level0(header_ctx: dict[str, Any], proposal: ProposalBox) -> tuple[int, int, int, int]:
    overview_w, overview_h = header_ctx["overview_size"]
    pad = max(16, int(round(max(proposal.w, proposal.h) * 0.03)))
    x1 = max(0, proposal.x - pad)
    y1 = max(0, proposal.y - pad)
    x2 = min(overview_w, proposal.x + proposal.w + pad)
    y2 = min(overview_h, proposal.y + proposal.h + pad)
    downsample = float(header_ctx["level_downsamples"][header_ctx["overview_level"]])
    x0 = int(round(x1 * downsample))
    y0 = int(round(y1 * downsample))
    w0 = min(int(round((x2 - x1) * downsample)), int(header_ctx["level_dimensions"][0][0]) - x0)
    h0 = min(int(round((y2 - y1) * downsample)), int(header_ctx["level_dimensions"][0][1]) - y0)
    return x0, y0, w0, h0


def _recover_nissl_bbox_level0(header_ctx: dict[str, Any], proposal: ProposalBox) -> tuple[int, int, int, int]:
    overview_w, overview_h = header_ctx["overview_size"]
    pad = max(24, int(round(max(proposal.w, proposal.h) * 0.08)))
    x1 = max(0, proposal.x - pad)
    y1 = max(0, proposal.y - pad)
    x2 = min(overview_w, proposal.x + proposal.w + pad)
    y2 = min(overview_h, proposal.y + proposal.h + pad)
    downsample = float(header_ctx["level_downsamples"][header_ctx["overview_level"]])
    x0 = int(round(x1 * downsample))
    y0 = int(round(y1 * downsample))
    w0 = min(int(round((x2 - x1) * downsample)), int(header_ctx["level_dimensions"][0][0]) - x0)
    h0 = min(int(round((y2 - y1) * downsample)), int(header_ctx["level_dimensions"][0][1]) - y0)
    return x0, y0, w0, h0


def _section_uid_slide_hint_candidates(slide_hint: str | None) -> list[str]:
    if not slide_hint:
        return []
    candidates = [slide_hint]
    # Legacy metadata stored semicolon-separated slide groups with underscores.
    repaired = re.sub(r"(?<=\d)_(?=\d{4}_)", ";", slide_hint)
    if repaired != slide_hint:
        candidates.append(repaired)
    return candidates


def _resolve_slide_path(
    section_root: Path,
    metadata: dict[str, Any],
    *,
    slide_lookup_cache: dict[tuple[str, str], dict[tuple[str, int], list[Path]]] | None = None,
) -> tuple[Path, str]:
    stain = str(metadata.get("stain") or "").lower()
    section_label = str(metadata.get("label") or section_root.name)
    sample_id = str(metadata.get("sample_id") or section_label.split("_", 1)[0])
    try:
        section_id = int(metadata.get("section_id") or section_label.split("_", 1)[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to derive section id for {section_root}") from exc

    # Already backfilled current path wins.
    for block_key in ("physical_provenance",):
        block = dict(metadata.get(block_key) or {})
        src = dict(block.get("source_slide") or {})
        src_path = src.get("path")
        if src_path:
            p = Path(str(src_path))
            if p.exists():
                return p, "metadata_physical_provenance"
    src = dict(metadata.get("source_slide") or {})
    src_path = src.get("path")
    if src_path:
        p = Path(str(src_path))
        if p.exists():
            return p, "metadata_source_slide"

    workspace_root = section_root.parent
    slide_dir = workspace_root.parent
    uid_info = _section_uid_parts(str(metadata.get("section_uid") or ""))
    slide_hint = uid_info["slide_hint"] if uid_info is not None else None
    for candidate_stem in _section_uid_slide_hint_candidates(slide_hint):
        exact = slide_dir / f"{candidate_stem}.ndpi"
        if exact.exists():
            return exact, "section_uid_hint"

    cache_key = (str(slide_dir.resolve()), stain)
    if slide_lookup_cache is None:
        slide_lookup_cache = {}
    lookup = slide_lookup_cache.get(cache_key)
    if lookup is None:
        lookup = _build_slide_lookup(slide_dir, stain)
        slide_lookup_cache[cache_key] = lookup
    candidates = lookup.get((sample_id, section_id), [])
    if not candidates:
        raise FileNotFoundError(f"No source NDPI found for {section_label}")
    if len(candidates) == 1:
        return candidates[0], "sample_section_lookup"

    normalized_hint = (slide_hint or "").replace(";", "_").lower()
    for path in candidates:
        if path.stem.replace(";", "_").lower() == normalized_hint:
            return path, "sample_section_lookup_disambiguated"
    return candidates[0], "sample_section_lookup_ambiguous_first"


def recover_section_physical_provenance(
    section_root: Path,
    *,
    slide_lookup_cache: dict[tuple[str, str], dict[tuple[str, int], list[Path]]] | None = None,
    loaded_slide_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = section_root / "metadata.json"
    crop_path = section_root / "crop_raw.png"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stain = str(metadata.get("stain") or "").lower()
    label = str(metadata.get("label") or section_root.name)
    existing = _provenance_from_existing_metadata(metadata, stain)
    if existing is not None:
        return metadata, existing
    crop_h, crop_w = _read_crop_shape_hw(crop_path)
    slide_path, slide_resolution_method = _resolve_slide_path(
        section_root,
        metadata,
        slide_lookup_cache=slide_lookup_cache,
    )
    cache_key = str(slide_path.resolve())
    if loaded_slide_cache is None:
        loaded_slide_cache = {}

    proposal = _proposal_from_metadata(metadata, label, stain)
    slide_ctx = loaded_slide_cache.get(cache_key)
    if slide_ctx is None:
        slide_ctx = _load_slide_geometry(slide_path, stain)
        loaded_slide_cache[cache_key] = slide_ctx
    if stain == "gallyas":
        level0_x, level0_y, level0_w, level0_h = _recover_gallyas_bbox_level0(slide_ctx, proposal)
    else:
        level0_x, level0_y, level0_w, level0_h = _recover_nissl_bbox_level0(slide_ctx, proposal)
    slide_backend = str(slide_ctx["backend"])
    overview_level = int(slide_ctx["overview_level"])
    overview_size = tuple(slide_ctx["overview_size"])
    level_dimensions = tuple(slide_ctx["level_dimensions"])
    level_downsamples = tuple(slide_ctx["level_downsamples"])
    objective_power = slide_ctx["objective_power"]
    slide_name = str(slide_ctx["slide_name"])
    mpp_x = slide_ctx["mpp_x"]
    mpp_y = slide_ctx["mpp_y"]
    scale_x = float(level0_w) / float(crop_w)
    scale_y = float(level0_h) / float(crop_h)
    crop_level = int(
        (dict(metadata.get("physical_provenance") or {}).get("source_slide") or {}).get(
            "crop_level",
            (dict(metadata.get("source_slide") or {}).get("crop_level") or min(3, len(level_dimensions) - 1)),
        )
    )
    crop_downsample = float(
        (dict(metadata.get("physical_provenance") or {}).get("source_slide") or {}).get(
            "crop_downsample",
            (dict(metadata.get("source_slide") or {}).get("crop_downsample") or level_downsamples[crop_level]),
        )
    )

    mpp_method = "slide_header"
    if mpp_x is None or mpp_y is None:
        default_mpp = DEFAULT_STAIN_MPP.get(stain)
        if default_mpp is None:
            raise ValueError(f"MPP unavailable for stain {stain} on {label}")
        mpp_x = float(default_mpp[0])
        mpp_y = float(default_mpp[1])
        mpp_method = "stain_default"

    provenance = {
        "version": PHYSICAL_PROVENANCE_VERSION,
        "recovered_at_utc": _utc_now_iso(),
        "recovery_method": "section_uid+bbox_overview+ndpi",
        "slide_resolution_method": slide_resolution_method,
        "mpp_recovery_method": mpp_method,
        "source_slide_identity": _file_identity(slide_path),
        "source_slide": {
            "path": str(slide_path),
            "name": slide_name,
            "backend": slide_backend,
            "fallback_reason": slide_ctx.get("fallback_reason"),
            "overview_level": int(overview_level),
            "overview_size_px": {
                "w": int(overview_size[0]),
                "h": int(overview_size[1]),
            },
            "level0_size_px": {
                "w": int(level_dimensions[0][0]),
                "h": int(level_dimensions[0][1]),
            },
            "overview_downsample": float(level_downsamples[overview_level]),
            "crop_level": int(crop_level),
            "crop_downsample": float(crop_downsample),
            "mpp_x": float(mpp_x),
            "mpp_y": float(mpp_y),
            "objective_power": objective_power,
        },
        "proposal_bbox_overview_xywh": proposal.bbox_dict(),
        "crop_bbox_level0": {
            "xyxy": {
                "x1": int(level0_x),
                "y1": int(level0_y),
                "x2": int(level0_x + level0_w),
                "y2": int(level0_y + level0_h),
            },
            "xywh": {
                "x": int(level0_x),
                "y": int(level0_y),
                "w": int(level0_w),
                "h": int(level0_h),
            },
        },
        "export_canvas": {
            "width_px": int(crop_w),
            "height_px": int(crop_h),
        },
        "canvas_to_slide_level0": {
            "mirror_x_applied": bool(
                dict(metadata.get("canvas_to_slide_level0") or {}).get("mirror_x_applied", False)
            ),
            "origin_level0_xy": {"x": int(level0_x), "y": int(level0_y)},
            "scale_level0_per_canvas_px": {"x": float(scale_x), "y": float(scale_y)},
            "mapping_note": (
                "If mirror_x_applied is false: slide_x = origin_x + canvas_x * scale_x. "
                "If mirror_x_applied is true: slide_x = origin_x + (canvas_width - 1 - canvas_x) * scale_x. "
                "slide_y = origin_y + canvas_y * scale_y."
            ),
        },
        "crop_bbox_level0_um_relative_to_slide_origin": {
            "x_um": float(level0_x * mpp_x),
            "y_um": float(level0_y * mpp_y),
            "w_um": float(level0_w * mpp_x),
            "h_um": float(level0_h * mpp_y),
        },
        "canvas_to_slide_um_per_px": {
            "x_um_per_px": float(scale_x * mpp_x),
            "y_um_per_px": float(scale_y * mpp_y),
        },
        "physical_calibration_available": True,
    }
    return metadata, provenance


def merge_physical_provenance_into_metadata(metadata: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    updated = dict(metadata)
    updated["physical_provenance"] = provenance
    updated["source_slide_identity"] = provenance["source_slide_identity"]
    updated["source_slide"] = provenance["source_slide"]
    updated["proposal_bbox_overview_xywh"] = provenance["proposal_bbox_overview_xywh"]
    updated["crop_bbox_level0"] = provenance["crop_bbox_level0"]
    updated["export_canvas"] = provenance["export_canvas"]
    updated["canvas_to_slide_level0"] = provenance["canvas_to_slide_level0"]
    updated["crop_bbox_level0_um_relative_to_slide_origin"] = provenance["crop_bbox_level0_um_relative_to_slide_origin"]
    updated["canvas_to_slide_um_per_px"] = provenance["canvas_to_slide_um_per_px"]
    reader_conf = dict(updated.get("reader_confidence") or {})
    reader_conf["physical_calibration_available"] = True
    reader_conf["physical_provenance_version"] = provenance["version"]
    reader_conf["mpp_recovery_method"] = provenance["mpp_recovery_method"]
    updated["reader_confidence"] = reader_conf
    return updated


def backfill_section_metadata_physical_provenance(
    section_root: Path,
    *,
    slide_lookup_cache: dict[tuple[str, str], dict[tuple[str, int], list[Path]]] | None = None,
    loaded_slide_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata, provenance = recover_section_physical_provenance(
        section_root,
        slide_lookup_cache=slide_lookup_cache,
        loaded_slide_cache=loaded_slide_cache,
    )
    updated = merge_physical_provenance_into_metadata(metadata, provenance)
    (section_root / "metadata.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return provenance


def recover_or_load_section_physical_provenance(
    section_root: Path,
    *,
    slide_lookup_cache: dict[tuple[str, str], dict[tuple[str, int], list[Path]]] | None = None,
    loaded_slide_cache: dict[str, Any] | None = None,
    write_back_if_missing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = section_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = dict(metadata.get("physical_provenance") or {})
    if provenance:
        return metadata, provenance
    metadata, provenance = recover_section_physical_provenance(
        section_root,
        slide_lookup_cache=slide_lookup_cache,
        loaded_slide_cache=loaded_slide_cache,
    )
    if write_back_if_missing:
        updated = merge_physical_provenance_into_metadata(metadata, provenance)
        metadata_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        metadata = updated
    return metadata, provenance


__all__ = [
    "PHYSICAL_PROVENANCE_VERSION",
    "DEFAULT_STAIN_MPP",
    "backfill_section_metadata_physical_provenance",
    "cleanup_session_temp_root",
    "merge_physical_provenance_into_metadata",
    "recover_or_load_section_physical_provenance",
    "recover_section_physical_provenance",
]
