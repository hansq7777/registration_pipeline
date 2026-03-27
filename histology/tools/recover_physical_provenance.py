#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None

Image.MAX_IMAGE_PIXELS = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registration_pipeline.histology.gui_mvp.hitl_gui.application.section_workspace import (  # noqa: E402
    WorkspaceSection,
    list_workspace_sections,
    load_workspace_metadata,
)
from registration_pipeline.histology.gui_mvp.hitl_gui.application.physical_provenance import (  # noqa: E402
    backfill_section_metadata_physical_provenance,
)
from registration_pipeline.histology.gui_mvp.hitl_gui.domain.entities import ProposalBox  # noqa: E402
from registration_pipeline.histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import (  # noqa: E402
    parse_slide_labels,
    propose_from_overview,
)
from registration_pipeline.histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import (  # noqa: E402
    cleanup_session_temp_root,
    effective_crop_bbox_level0,
    load_slide_bundle,
)


DEFAULT_WORKSPACE_ROOTS = [
    Path("/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks"),
    Path("/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks"),
]
DEFAULT_REPORT_ROOT = Path("/mnt/d/Research/Image Analysis/Nanozoomer scans")

SECTION_UID_RE = re.compile(r"^(?P<stain>[a-zA-Z]+)_(?P<sample>\d+)_(?P<section>\d+)__(?P<slide>[a-zA-Z]+_\d+_\d+-\d+)__r(?P<rank>\d+)$")
SLIDE_STEM_RE = re.compile(r"^(?P<stain>[a-zA-Z]+)_(?P<sample>\d+)_(?P<start>\d+)-(?P<end>\d+)$")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_windowsish(path_text: str | None) -> str | None:
    if not path_text:
        return None
    return str(path_text).replace("\\", "/").lower()


def _read_crop_shape(item: WorkspaceSection) -> tuple[int, int]:
    with Image.open(item.crop_path) as im:
        return int(im.height), int(im.width)


def _load_manifest_index(workspace_root: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest_path in (
        workspace_root / "_export_manifest.json",
        workspace_root / "test" / "_export_manifest.json",
    ):
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.get("items", []):
            label = str(item.get("label") or "")
            if label:
                index[label].append(item)
    return index


def _section_uid_parts(section_uid: str) -> dict[str, Any] | None:
    m = SECTION_UID_RE.match(section_uid)
    if not m:
        return None
    return {
        "stain": m.group("stain").lower(),
        "sample_id": m.group("sample"),
        "section_id": int(m.group("section")),
        "slide_stem": m.group("slide"),
        "proposal_rank": int(m.group("rank")),
    }


def _slide_range_info(slide_stem: str) -> dict[str, Any] | None:
    m = SLIDE_STEM_RE.match(slide_stem)
    if not m:
        return None
    return {
        "stain": m.group("stain").lower(),
        "sample_id": m.group("sample"),
        "start_section": int(m.group("start")),
        "end_section": int(m.group("end")),
    }


def _find_slide_path(workspace_root: Path, slide_stem: str) -> Path | None:
    candidate = workspace_root.parent / f"{slide_stem}.ndpi"
    if candidate.exists():
        return candidate
    exact = list(workspace_root.parent.glob(f"{slide_stem}.ndpi"))
    if exact:
        return exact[0]
    fallback = list(workspace_root.parent.rglob(f"{slide_stem}.ndpi"))
    if fallback:
        return fallback[0]
    return None


def _load_header_only_geometry(slide_path: Path, stain: str) -> Any:
    if openslide is None:
        raise RuntimeError("openslide unavailable")
    slide = openslide.OpenSlide(str(slide_path))
    mpp_x = None
    mpp_y = None
    try:
        if "openslide.mpp-x" in slide.properties:
            mpp_x = float(slide.properties["openslide.mpp-x"])
        if "openslide.mpp-y" in slide.properties:
            mpp_y = float(slide.properties["openslide.mpp-y"])
    except Exception:
        mpp_x = None
        mpp_y = None
    return {
        "backend": "openslide_header_only",
        "stain": stain,
        "overview_level": slide.level_count - 1,
        "overview_size": slide.level_dimensions[slide.level_count - 1],
        "level_dimensions": tuple(slide.level_dimensions),
        "level_downsamples": tuple(float(x) for x in slide.level_downsamples),
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
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


def _proposal_from_metadata(item: WorkspaceSection, metadata: dict[str, Any]) -> ProposalBox | None:
    bbox = metadata.get("bbox_overview") or {}
    if not all(k in bbox for k in ("x", "y", "w", "h")):
        return None
    return ProposalBox(
        label=str(metadata.get("label") or item.label),
        stain=str(metadata.get("stain") or item.stain),
        sample_id=str(metadata.get("sample_id") or ""),
        section_id=int(metadata.get("section_id") or 0),
        proposal_rank=int(metadata.get("proposal_rank") or 0),
        x=int(bbox["x"]),
        y=int(bbox["y"]),
        w=int(bbox["w"]),
        h=int(bbox["h"]),
    )


def _manifest_match_record(
    manifest_entries: list[dict[str, Any]],
    slide_path: Path | None,
    recovered_bbox_level0: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    if not manifest_entries:
        return {
            "manifest_status": "missing",
            "manifest_match": None,
            "manifest_entries": 0,
        }
    if len(manifest_entries) != 1:
        return {
            "manifest_status": "ambiguous",
            "manifest_match": None,
            "manifest_entries": len(manifest_entries),
        }
    item = manifest_entries[0]
    manifest_slide = _normalize_windowsish((item.get("source_slide") or {}).get("path"))
    current_slide = _normalize_windowsish(str(slide_path)) if slide_path is not None else None
    manifest_bbox = item.get("crop_bbox_level0", {}).get("xywh") or {}
    bbox_match = None
    if recovered_bbox_level0 is not None and all(k in manifest_bbox for k in ("x", "y", "w", "h")):
        bbox_match = (
            int(manifest_bbox["x"]) == int(recovered_bbox_level0[0])
            and int(manifest_bbox["y"]) == int(recovered_bbox_level0[1])
            and int(manifest_bbox["w"]) == int(recovered_bbox_level0[2])
            and int(manifest_bbox["h"]) == int(recovered_bbox_level0[3])
        )
    slide_match = None if current_slide is None else (manifest_slide == current_slide)
    if slide_match is True and bbox_match is True:
        status = "exact"
    elif slide_match is False or bbox_match is False:
        status = "mismatch"
    else:
        status = "partial"
    return {
        "manifest_status": status,
        "manifest_match": {
            "slide_match": slide_match,
            "bbox_match": bbox_match,
        },
        "manifest_entries": 1,
    }


def recover_workspace(
    workspace_root: Path,
    *,
    slide_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_index = _load_manifest_index(workspace_root)
    sections = list_workspace_sections(workspace_root)
    results: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, WorkspaceSection, dict[str, Any], list[str], dict[str, Any] | None, dict[str, Any] | None, Path | None, int, int]] = []
    for idx, item in enumerate(sections, start=1):
        metadata = load_workspace_metadata(item)
        issues: list[str] = []
        uid_info = _section_uid_parts(str(metadata.get("section_uid") or ""))
        if uid_info is None:
            issues.append("section_uid_unparseable")
        slide_stem = uid_info["slide_stem"] if uid_info is not None else None
        slide_range = _slide_range_info(slide_stem) if slide_stem is not None else None
        if slide_stem is None:
            issues.append("slide_stem_missing")
        if slide_range is None and slide_stem is not None:
            issues.append("slide_stem_range_unparseable")
        slide_path = _find_slide_path(workspace_root, slide_stem) if slide_stem is not None else None
        if slide_path is None:
            issues.append("slide_missing")
        crop_h, crop_w = _read_crop_shape(item)
        proposal = _proposal_from_metadata(item, metadata)
        if proposal is None:
            issues.append("bbox_overview_missing")
        pending.append((idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w))

    grouped_counts = Counter(
        str(Path(str(slide_path)).resolve())
        for _, _, _, _, _, _, slide_path, _, _ in pending
        if slide_path is not None
    )
    loaded_slide_counter = 0

    def finalize_record(
        *,
        idx: int,
        item: WorkspaceSection,
        metadata: dict[str, Any],
        issues: list[str],
        uid_info: dict[str, Any] | None,
        slide_range: dict[str, Any] | None,
        slide_path: Path | None,
        crop_h: int,
        crop_w: int,
        recovered_bbox_level0: tuple[int, int, int, int] | None,
        mpp_x: float | None,
        mpp_y: float | None,
        backend: str | None,
        overview_level: int | None,
        overview_downsample: float | None,
    ) -> None:
        scale_x = None
        scale_y = None
        um_per_px_x = None
        um_per_px_y = None
        crop_bbox_um = None
        if recovered_bbox_level0 is not None:
            x0, y0, w0, h0 = recovered_bbox_level0
            scale_x = float(w0) / float(crop_w) if crop_w > 0 else None
            scale_y = float(h0) / float(crop_h) if crop_h > 0 else None
            if scale_x is None or scale_y is None or not math.isfinite(scale_x) or not math.isfinite(scale_y):
                issues.append("canvas_scale_invalid")
            if mpp_x is not None and mpp_y is not None and scale_x is not None and scale_y is not None:
                um_per_px_x = scale_x * mpp_x
                um_per_px_y = scale_y * mpp_y
                crop_bbox_um = {
                    "x_um": float(x0 * mpp_x),
                    "y_um": float(y0 * mpp_y),
                    "w_um": float(w0 * mpp_x),
                    "h_um": float(h0 * mpp_y),
                }
        section_in_range = None
        if slide_range is not None and uid_info is not None:
            section_in_range = slide_range["start_section"] <= uid_info["section_id"] <= slide_range["end_section"]
            if not section_in_range:
                issues.append("section_outside_slide_range")
            if slide_range["sample_id"] != uid_info["sample_id"]:
                issues.append("sample_id_mismatch_vs_slide")
        manifest_info = _manifest_match_record(manifest_index.get(item.label, []), slide_path, recovered_bbox_level0)
        recoverable = (
            slide_path is not None
            and recovered_bbox_level0 is not None
            and mpp_x is not None
            and mpp_y is not None
            and _proposal_from_metadata(item, metadata) is not None
        )
        status = "recoverable" if recoverable else "incomplete"
        results.append(
            {
                "workspace_root": str(workspace_root),
                "section_dir": str(item.section_dir),
                "label": item.label,
                "stain": item.stain,
                "sample_id": metadata.get("sample_id"),
                "section_id": metadata.get("section_id"),
                "section_uid": metadata.get("section_uid"),
                "crop_raw_shape_hw": [crop_h, crop_w],
                "bbox_overview": metadata.get("bbox_overview"),
                "slide_stem": uid_info["slide_stem"] if uid_info is not None else None,
                "slide_path": str(slide_path) if slide_path is not None else None,
                "slide_backend": backend,
                "overview_level": overview_level,
                "overview_downsample": overview_downsample,
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "recovered_crop_bbox_level0_xywh": (
                    {
                        "x": int(recovered_bbox_level0[0]),
                        "y": int(recovered_bbox_level0[1]),
                        "w": int(recovered_bbox_level0[2]),
                        "h": int(recovered_bbox_level0[3]),
                    }
                    if recovered_bbox_level0 is not None
                    else None
                ),
                "recovered_canvas_to_slide_level0": (
                    {
                        "origin_level0_xy": {"x": int(recovered_bbox_level0[0]), "y": int(recovered_bbox_level0[1])},
                        "scale_level0_per_canvas_px": {"x": scale_x, "y": scale_y},
                    }
                    if recovered_bbox_level0 is not None
                    else None
                ),
                "recovered_canvas_to_slide_um_per_px": (
                    {"x_um_per_px": um_per_px_x, "y_um_per_px": um_per_px_y}
                    if um_per_px_x is not None and um_per_px_y is not None
                    else None
                ),
                "recovered_crop_bbox_level0_um": crop_bbox_um,
                "section_in_named_slide_range": section_in_range,
                "manifest_check": manifest_info,
                "status": status,
                "issues": issues,
                "issue_count": len(issues),
                "index_in_workspace": idx,
            }
        )

    grouped_entries: dict[str, list[tuple[int, WorkspaceSection, dict[str, Any], list[str], dict[str, Any] | None, dict[str, Any] | None, Path, int, int]]] = defaultdict(list)
    for entry in pending:
        idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w = entry
        if slide_path is None or _proposal_from_metadata(item, metadata) is None:
            finalize_record(
                idx=idx,
                item=item,
                metadata=metadata,
                issues=issues,
                uid_info=uid_info,
                slide_range=slide_range,
                slide_path=slide_path,
                crop_h=crop_h,
                crop_w=crop_w,
                recovered_bbox_level0=None,
                mpp_x=None,
                mpp_y=None,
                backend=None,
                overview_level=None,
                overview_downsample=None,
            )
            continue
        grouped_entries[str(slide_path.resolve())].append((idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w))

    for slide_key, entries in grouped_entries.items():
        idx0, item0, metadata0, _, _, _, slide_path0, _, _ = entries[0]
        assert slide_path0 is not None
        loaded_slide_counter += 1
        print(f"[slide {loaded_slide_counter}/{len(grouped_counts)}] {slide_path0.name} ({len(entries)} section(s))", flush=True)
        try:
            if item0.stain.lower() == "gallyas":
                header_ctx = slide_cache.get(slide_key)
                if header_ctx is None:
                    header_ctx = _load_header_only_geometry(slide_path0, item0.stain)
                    slide_cache[slide_key] = header_ctx
                for idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w in entries:
                    proposal = _proposal_from_metadata(item, metadata)
                    assert proposal is not None
                    recovered_bbox_level0 = _recover_gallyas_bbox_level0(header_ctx, proposal)
                    if header_ctx["mpp_x"] is None or header_ctx["mpp_y"] is None:
                        issues.append("mpp_missing")
                    finalize_record(
                        idx=idx,
                        item=item,
                        metadata=metadata,
                        issues=issues,
                        uid_info=uid_info,
                        slide_range=slide_range,
                        slide_path=slide_path,
                        crop_h=crop_h,
                        crop_w=crop_w,
                        recovered_bbox_level0=recovered_bbox_level0,
                        mpp_x=header_ctx["mpp_x"],
                        mpp_y=header_ctx["mpp_y"],
                        backend=header_ctx["backend"],
                        overview_level=int(header_ctx["overview_level"]),
                        overview_downsample=float(header_ctx["level_downsamples"][header_ctx["overview_level"]]),
                    )
            else:
                context = slide_cache.get(slide_key)
                if context is None:
                    loaded = load_slide_bundle(slide_path0, item0.stain)
                    parsed_stain, labels = parse_slide_labels(slide_path0.stem)
                    proposals = propose_from_overview(slide_path0, parsed_stain, labels, np.asarray(loaded.overview))
                    context = {
                        "loaded": loaded,
                        "by_label": {p.label: p for p in proposals},
                    }
                    slide_cache[slide_key] = context
                loaded = context["loaded"]
                by_label = context["by_label"]
                for idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w in entries:
                    proposal = by_label.get(item.label) or _proposal_from_metadata(item, metadata)
                    assert proposal is not None
                    if proposal.bbox_dict() != (metadata.get("bbox_overview") or {}):
                        issues = list(issues)
                        issues.append("proposal_bbox_mismatch_vs_slide_recompute")
                    recovered_bbox_level0 = effective_crop_bbox_level0(loaded, proposal)
                    if loaded.mpp_x is None or loaded.mpp_y is None:
                        issues.append("mpp_missing")
                    finalize_record(
                        idx=idx,
                        item=item,
                        metadata=metadata,
                        issues=issues,
                        uid_info=uid_info,
                        slide_range=slide_range,
                        slide_path=slide_path,
                        crop_h=crop_h,
                        crop_w=crop_w,
                        recovered_bbox_level0=recovered_bbox_level0,
                        mpp_x=loaded.mpp_x,
                        mpp_y=loaded.mpp_y,
                        backend=loaded.backend,
                        overview_level=int(loaded.overview_level),
                        overview_downsample=float(loaded.level_downsamples[loaded.overview_level]),
                    )
        except Exception as exc:
            for idx, item, metadata, issues, uid_info, slide_range, slide_path, crop_h, crop_w in entries:
                issues = list(issues)
                issues.append(f"slide_load_failed:{type(exc).__name__}")
                finalize_record(
                    idx=idx,
                    item=item,
                    metadata=metadata,
                    issues=issues,
                    uid_info=uid_info,
                    slide_range=slide_range,
                    slide_path=slide_path,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    recovered_bbox_level0=None,
                    mpp_x=None,
                    mpp_y=None,
                    backend=None,
                    overview_level=None,
                    overview_downsample=None,
                )
    return results


def backfill_workspace(
    workspace_root: Path,
    *,
    slide_lookup_cache: dict[tuple[str, str], dict[tuple[str, int], list[Path]]],
    loaded_slide_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(list_workspace_sections(workspace_root), start=1):
        try:
            provenance = backfill_section_metadata_physical_provenance(
                item.section_dir,
                slide_lookup_cache=slide_lookup_cache,
                loaded_slide_cache=loaded_slide_cache,
            )
            rows.append(
                {
                    "label": item.label,
                    "stain": item.stain,
                    "section_dir": str(item.section_dir),
                    "slide_path": str((provenance.get("source_slide") or {}).get("path") or ""),
                    "mpp_x": (provenance.get("source_slide") or {}).get("mpp_x"),
                    "mpp_y": (provenance.get("source_slide") or {}).get("mpp_y"),
                    "status": "backfilled",
                    "issues": [],
                    "index_in_workspace": idx,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "label": item.label,
                    "stain": item.stain,
                    "section_dir": str(item.section_dir),
                    "slide_path": None,
                    "mpp_x": None,
                    "mpp_y": None,
                    "status": "failed",
                    "issues": [f"{type(exc).__name__}:{exc}"],
                    "index_in_workspace": idx,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {
                "label": row["label"],
                "stain": row["stain"],
                "status": row["status"],
                "issues": ";".join(row["issues"]),
                "slide_stem": row.get("slide_stem"),
                "slide_path": row.get("slide_path"),
                "mpp_x": row.get("mpp_x"),
                "mpp_y": row.get("mpp_y"),
                "crop_raw_h": (row.get("crop_raw_shape_hw") or [None, None])[0],
                "crop_raw_w": (row.get("crop_raw_shape_hw") or [None, None])[1],
                "bbox_overview": json.dumps(row.get("bbox_overview"), ensure_ascii=False),
                "crop_bbox_level0": json.dumps(row.get("recovered_crop_bbox_level0_xywh"), ensure_ascii=False),
                "canvas_to_slide_um_per_px": json.dumps(row.get("recovered_canvas_to_slide_um_per_px"), ensure_ascii=False),
                "manifest_status": (
                    row["manifest_check"]["manifest_status"]
                    if isinstance(row.get("manifest_check"), dict) and "manifest_status" in row["manifest_check"]
                    else None
                ),
                "manifest_match": (
                    json.dumps(row["manifest_check"]["manifest_match"], ensure_ascii=False)
                    if isinstance(row.get("manifest_check"), dict) and "manifest_match" in row["manifest_check"]
                    else None
                ),
                "section_dir": row["section_dir"],
            }
        )
    if not flat_rows:
        return
    fieldnames = list(flat_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover/check physical provenance for histology section workspaces.")
    parser.add_argument(
        "--workspace-root",
        action="append",
        dest="workspace_roots",
        help="Workspace root to scan. Repeatable. Defaults to current myelin + nissl Tissue&Masks roots.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory where the report folder will be written. Defaults to Nanozoomer scans root.",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Backfill recovered physical_provenance into each section metadata.json.",
    )
    args = parser.parse_args()

    workspace_roots = [Path(p) for p in (args.workspace_roots or DEFAULT_WORKSPACE_ROOTS)]
    out_root = args.out_dir or DEFAULT_REPORT_ROOT
    report_dir = out_root / f"physical_provenance_recovery_check_{_utc_now_compact()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    slide_cache: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    root_summaries: dict[str, Any] = {}
    try:
        for root in workspace_roots:
            rows = (
                backfill_workspace(root, slide_lookup_cache={}, loaded_slide_cache=slide_cache)
                if args.write_metadata
                else recover_workspace(root, slide_cache=slide_cache)
            )
            all_rows.extend(rows)
            counts = Counter(row["status"] for row in rows)
            issue_counts = Counter(issue for row in rows for issue in row["issues"])
            root_summaries[str(root)] = {
                "total_sections": len(rows),
                "recoverable_sections": counts.get("recoverable", 0),
                "incomplete_sections": counts.get("incomplete", 0),
                "backfilled_sections": counts.get("backfilled", 0),
                "failed_sections": counts.get("failed", 0),
                "top_issues": issue_counts.most_common(20),
            }
    finally:
        cleanup_session_temp_root()

    counts = Counter(row["status"] for row in all_rows)
    issue_counts = Counter(issue for row in all_rows for issue in row["issues"])
    manifest_counts = Counter(
        row["manifest_check"]["manifest_status"]
        for row in all_rows
        if isinstance(row.get("manifest_check"), dict) and "manifest_status" in row["manifest_check"]
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_roots": [str(p) for p in workspace_roots],
        "total_sections": len(all_rows),
        "recoverable_sections": counts.get("recoverable", 0),
        "incomplete_sections": counts.get("incomplete", 0),
        "backfilled_sections": counts.get("backfilled", 0),
        "failed_sections": counts.get("failed", 0),
        "recoverable_fraction": (counts.get("recoverable", 0) / len(all_rows)) if all_rows else None,
        "manifest_check_counts": dict(manifest_counts),
        "issue_counts": dict(issue_counts),
        "by_workspace_root": root_summaries,
    }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "per_section.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    _write_csv(report_dir / "per_section.csv", all_rows)
    failures = [row for row in all_rows if row["status"] != "recoverable"]
    _write_csv(report_dir / "incomplete_sections.csv", failures)

    print(json.dumps({"report_dir": str(report_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
