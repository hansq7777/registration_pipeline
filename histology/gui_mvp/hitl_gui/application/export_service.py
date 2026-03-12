from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from ..domain import ExportPlanItem, LoadedSlide
from ..pipeline_adapters import build_export_payload
from ..pipeline_adapters.slide_io import (
    effective_crop_bbox_level0,
    effective_crop_rect_overview,
    extract_crop_for_preview,
    open_slide_handle,
    write_png_lossless_fast,
)


@lru_cache(maxsize=1)
def _git_context() -> dict:
    repo_root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(status)
    except Exception:
        dirty = None
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _file_identity(path: Path) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {
            "identity_method": "path_size_mtime",
            "path": str(path),
            "source_slide_checksum": None,
            "size_bytes": None,
            "mtime_unix_sec": None,
            "mtime_ns": None,
        }
    return {
        "identity_method": "path_size_mtime",
        "path": str(path),
        "source_slide_checksum": None,
        "size_bytes": int(stat.st_size),
        "mtime_unix_sec": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _proposal_context(plan: ExportPlanItem, all_proposals) -> dict:
    total = len(all_proposals)
    row_break = (total + 1) // 2 if total > 0 else 0
    rank = int(plan.proposal.proposal_rank)
    row_index = None
    column_index = None
    if row_break > 0:
        row_index = 0 if rank <= row_break else 1
        column_index = rank - 1 if rank <= row_break else rank - row_break - 1
    snapshot = [
        {
            "label": proposal.label,
            "proposal_rank": int(proposal.proposal_rank),
            "bbox_overview_xywh": proposal.bbox_dict(),
            "mirror_enabled": bool(proposal.mirror_enabled),
        }
        for proposal in all_proposals
    ]
    return {
        "expected_label": plan.expected_label,
        "candidate_rank": rank,
        "row_index": row_index,
        "column_index": column_index,
        "proposal_count": total,
        "all_candidate_boxes_snapshot": snapshot,
    }


def _mask_qc_stats(
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
    usable_mask: np.ndarray,
    ownership_strict: np.ndarray | None = None,
) -> dict:
    tissue = tissue_mask > 0
    artifact = artifact_mask > 0
    usable = usable_mask > 0
    h, w = tissue.shape[:2]
    mask_area = int(tissue.sum())
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue.astype(np.uint8), 8)
    component_areas = sorted([int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, num)], reverse=True)

    band = max(1, int(round(min(h, w) * 0.01)))
    border_band = np.zeros_like(tissue, dtype=bool)
    border_band[:band, :] = True
    border_band[-band:, :] = True
    border_band[:, :band] = True
    border_band[:, -band:] = True
    border_touch_ratio = float((tissue & border_band).sum() / max(mask_area, 1))
    neighbor_occupancy_ratio = None
    if ownership_strict is not None:
        neighbor_occupancy_ratio = float((tissue & (~ownership_strict)).sum() / max(mask_area, 1))

    return {
        "tissue_area_px": mask_area,
        "artifact_area_px": int(artifact.sum()),
        "usable_area_px": int(usable.sum()),
        "tissue_connected_components": max(0, int(num - 1)),
        "tissue_component_areas_desc_px": component_areas,
        "largest_tissue_component_px": int(component_areas[0]) if component_areas else 0,
        "border_touch_ratio": border_touch_ratio,
        "neighbor_occupancy_ratio": neighbor_occupancy_ratio,
    }


class ExportWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        loaded_slide: LoadedSlide,
        plan_items: list[ExportPlanItem],
        export_root: Path,
        crop_level: int,
        profile_name: str = "review_mask",
    ) -> None:
        super().__init__()
        self.loaded_slide = loaded_slide
        self.plan_items = plan_items
        self.export_root = export_root
        self.crop_level = crop_level
        self.profile_name = profile_name
        self.max_workers = max(1, min(4, (os.cpu_count() or 1)))
        self._cancel_event = threading.Event()

    @staticmethod
    def _resize_mask_to_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        target_h, target_w = shape_hw
        if mask.shape[:2] == (target_h, target_w):
            return mask.astype(np.uint8)
        resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8) * 255

    def _build_metadata(
        self,
        plan: ExportPlanItem,
        crop_rgb: np.ndarray,
        tissue_mask: np.ndarray,
        artifact_mask: np.ndarray,
        usable_mask: np.ndarray,
        export_hash: str,
    ) -> dict:
        proposal = plan.proposal
        crop_rect_ov = effective_crop_rect_overview(self.loaded_slide, proposal)
        crop_bbox_level0 = effective_crop_bbox_level0(self.loaded_slide, proposal)
        crop_h, crop_w = crop_rgb.shape[:2]
        level0_x, level0_y, level0_w, level0_h = crop_bbox_level0
        overview_downsample = float(self.loaded_slide.level_downsamples[self.loaded_slide.overview_level])
        crop_downsample = float(
            self.loaded_slide.level_downsamples[min(self.crop_level, len(self.loaded_slide.level_downsamples) - 1)]
        )
        scale_x = float(level0_w / max(1, crop_w))
        scale_y = float(level0_h / max(1, crop_h))
        ownership_strict = None
        try:
            from ..pipeline_adapters.tool_bridge import load_histology_tool_module, proposal_to_tool_candidate

            tool = load_histology_tool_module()
            all_candidates = [proposal_to_tool_candidate(p, rank=idx + 1) for idx, p in enumerate(self.loaded_slide.proposals)]
            target_rank = self.loaded_slide.proposals.index(proposal) + 1
            target_candidate = proposal_to_tool_candidate(proposal, rank=target_rank)
            ownership_strict, _, _ = tool.build_crop_ownership_masks(
                target_candidate=target_candidate,
                all_candidates=all_candidates,
                crop_bbox_level0=crop_bbox_level0,
                crop_shape=crop_rgb.shape[:2],
                crop_downsample=crop_downsample,
                overview_downsample=overview_downsample,
            )
        except Exception:
            ownership_strict = None
        if plan.proposal.mask_preset == "legacy_simple":
            mask_algorithm_version = "gui_legacy_simple_autoseg_v1"
        elif plan.proposal.mask_preset == "hybrid_balanced":
            mask_algorithm_version = "gui_hybrid_balanced_autoseg_v2"
        elif self.loaded_slide.stain.lower() == "nissl":
            mask_algorithm_version = "gui_nissl_tool_baseline_autoseg_v1"
        elif self.loaded_slide.stain.lower() == "gallyas":
            mask_algorithm_version = "gui_contextual_autoseg_v1"
        else:
            mask_algorithm_version = "gui_simple_autoseg_v1"
        bbox_algorithm_version = (
            "gallyas_bbox_hybrid_topfloor55_wide24_v4"
            if self.loaded_slide.stain.lower() == "gallyas"
            else "coverage_first_bbox_v1"
        )

        metadata = {
            "export_format_version": "review_mask_v2",
            "label": proposal.label,
            "stain": proposal.stain,
            "sample_id": proposal.sample_id,
            "section_id": proposal.section_id,
            "proposal_rank": proposal.proposal_rank,
            "mirror_enabled": proposal.mirror_enabled,
            "section_uid": plan.section_uid,
            "revision_id": plan.revision_id,
            "profile_name": self.profile_name,
            "export_hash": export_hash,
            "source_slide_identity": _file_identity(self.loaded_slide.slide_path),
            "source_slide": {
                "path": str(self.loaded_slide.slide_path),
                "name": self.loaded_slide.slide_name,
                "backend": self.loaded_slide.backend,
                "fallback_reason": self.loaded_slide.fallback_reason or None,
                "overview_level": self.loaded_slide.overview_level,
                "overview_size_px": {
                    "w": int(self.loaded_slide.overview_size[0]),
                    "h": int(self.loaded_slide.overview_size[1]),
                },
                "level0_size_px": {
                    "w": int(self.loaded_slide.level_dimensions[0][0]),
                    "h": int(self.loaded_slide.level_dimensions[0][1]),
                },
                "overview_downsample": overview_downsample,
                "crop_level": int(self.crop_level),
                "crop_downsample": crop_downsample,
                "mpp_x": self.loaded_slide.mpp_x,
                "mpp_y": self.loaded_slide.mpp_y,
                "objective_power": self.loaded_slide.objective_power,
                "backend_details": {
                    "temp_proxy_dir": str(self.loaded_slide.temp_proxy_dir) if self.loaded_slide.temp_proxy_dir else None,
                    "tifffile_midres_page_index": self.loaded_slide.tifffile_midres_page_index,
                    "tifffile_midres_downsample": self.loaded_slide.tifffile_midres_downsample,
                    "tifffile_overview_scale_from_midres": self.loaded_slide.tifffile_overview_scale_from_midres,
                },
            },
            "algorithm_context": {
                "algorithm_version": "histology_gui_export_v3",
                "bbox_algorithm_version": bbox_algorithm_version,
                "mask_algorithm_version": mask_algorithm_version,
                "mask_preset_selected": plan.proposal.mask_preset,
                "crop_policy_name": bbox_algorithm_version,
                "proposal_source": "gui_overview_proposal",
                "mask_source_layer": "tissue_mask_final/artifact_mask_final",
                "mask_policy_name": mask_algorithm_version,
                **_git_context(),
            },
            "proposal_context": _proposal_context(plan, self.loaded_slide.proposals),
            "proposal_bbox_overview_xywh": proposal.bbox_dict(),
            "crop_bbox_overview": {
                "xyxy": {
                    "x1": int(crop_rect_ov[0]),
                    "y1": int(crop_rect_ov[1]),
                    "x2": int(crop_rect_ov[2]),
                    "y2": int(crop_rect_ov[3]),
                },
                "xywh": {
                    "x": int(crop_rect_ov[0]),
                    "y": int(crop_rect_ov[1]),
                    "w": int(crop_rect_ov[2] - crop_rect_ov[0]),
                    "h": int(crop_rect_ov[3] - crop_rect_ov[1]),
                },
            },
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
                "mirror_x_applied": bool(proposal.mirror_enabled),
                "origin_level0_xy": {"x": int(level0_x), "y": int(level0_y)},
                "scale_level0_per_canvas_px": {"x": scale_x, "y": scale_y},
                "mapping_note": (
                    "If mirror_x_applied is false: slide_x = origin_x + canvas_x * scale_x. "
                    "If mirror_x_applied is true: slide_x = origin_x + (canvas_width - 1 - canvas_x) * scale_x. "
                    "slide_y = origin_y + canvas_y * scale_y."
                ),
            },
            "mask_qc_stats": _mask_qc_stats(tissue_mask, artifact_mask, usable_mask, ownership_strict),
            "manual_edit_summary": {
                "manually_edited": bool(plan.revision_count > 0 or plan.manual_mask_version > 0),
                "revision_count": int(plan.revision_count),
                "manual_mask_version": int(plan.manual_mask_version),
                "latest_revision_id": plan.revision_id,
                "latest_review_note": plan.review_notes,
                "review_status": plan.review_status,
                "mask_work_level": plan.proposal.mask_work_level,
                "mask_work_shape_px": (
                    {
                        "h": int(plan.proposal.mask_work_shape[0]),
                        "w": int(plan.proposal.mask_work_shape[1]),
                    }
                    if plan.proposal.mask_work_shape is not None
                    else None
                ),
            },
            "physical_orientation": {
                "mirror_enabled": bool(proposal.mirror_enabled),
                "mirror_state": plan.mirror_state,
                "left_right_label_state": "unknown",
            },
            "reader_confidence": {
                "reader_backend": self.loaded_slide.backend,
                "fallback_used": bool(self.loaded_slide.backend != "openslide"),
                "fallback_geometry_is_approximate": False,
                "level0_mapping_confidence": "high",
                "physical_calibration_available": bool(
                    self.loaded_slide.mpp_x is not None and self.loaded_slide.mpp_y is not None
                ),
                "geometry_note": (
                    "level0 mapping is derived from the same slide pyramid/downsample chain; "
                    "physical calibration may be unavailable under fallback readers"
                ),
                "fallback_reason": self.loaded_slide.fallback_reason or None,
            },
            "derived_outputs": {
                "usable_tissue_mask": "tissue_mask_final minus artifact_mask_final",
                "foreground_rgba": "crop_raw RGB with alpha from usable_tissue_mask",
                "foreground_rgb_white": "derivable later from crop_raw + usable_tissue_mask",
                "foreground_rgb_black": "derivable later from crop_raw + usable_tissue_mask",
            },
            "output_files": [
                "crop_raw.png",
                "tissue_mask_final.png",
                "artifact_mask_final.png",
                "usable_tissue_mask.png",
                "foreground_rgba.png",
                "metadata.json",
            ],
        }
        if self.loaded_slide.mpp_x is not None and self.loaded_slide.mpp_y is not None:
            metadata["crop_bbox_level0_um_relative_to_slide_origin"] = {
                "x_um": float(level0_x * self.loaded_slide.mpp_x),
                "y_um": float(level0_y * self.loaded_slide.mpp_y),
                "w_um": float(level0_w * self.loaded_slide.mpp_x),
                "h_um": float(level0_h * self.loaded_slide.mpp_y),
            }
            metadata["canvas_to_slide_um_per_px"] = {
                "x_um_per_px": float(scale_x * self.loaded_slide.mpp_x),
                "y_um_per_px": float(scale_y * self.loaded_slide.mpp_y),
            }
        return metadata

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        work_queue: queue.Queue = queue.Queue(maxsize=max(2, self.max_workers))
        manifest_entries: list[dict] = []
        manifest_lock = threading.Lock()
        exported: list[str] = []
        skipped: list[str] = []
        slide_handle = None

        def writer_loop() -> None:
            while True:
                item = work_queue.get()
                if item is None:
                    work_queue.task_done()
                    return
                try:
                    sec_dir: Path = item["section_dir"]
                    try:
                        sec_dir.mkdir(parents=True, exist_ok=False)
                    except FileExistsError:
                        skipped.append(item["label"])
                        self.progress.emit(f"Skipped during write: {item['label']}")
                        continue

                    crop_rgb = item["crop_rgb"]
                    payload = item["payload"]
                    metadata = item["metadata"]
                    write_png_lossless_fast(sec_dir / "crop_raw.png", crop_rgb, "RGB")
                    write_png_lossless_fast(sec_dir / "tissue_mask_final.png", payload["tissue_mask_final"], "L")
                    write_png_lossless_fast(sec_dir / "artifact_mask_final.png", payload["artifact_mask_final"], "L")
                    write_png_lossless_fast(sec_dir / "usable_tissue_mask.png", payload["usable_tissue_mask"], "L")
                    write_png_lossless_fast(sec_dir / "foreground_rgba.png", payload["foreground_rgba"], "RGBA")
                    (sec_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    with manifest_lock:
                        manifest_entries.append(metadata)
                    exported.append(item["label"])
                    self.progress.emit(f"Wrote: {item['label']}")
                finally:
                    work_queue.task_done()

        try:
            self.progress.emit(
                f"Export planner: {len(self.plan_items)} new folders to write with {self.max_workers} writer threads."
            )
            slide_handle = open_slide_handle(self.loaded_slide)

            writers = [threading.Thread(target=writer_loop, daemon=True) for _ in range(self.max_workers)]
            for writer in writers:
                writer.start()

            total = len(self.plan_items)
            for idx, plan in enumerate(self.plan_items, start=1):
                if self._cancel_event.is_set():
                    self.progress.emit("Export cancelled.")
                    break
                self.progress.emit(f"Reading crop {idx}/{total}: {plan.proposal.label}")
                crop_rgb = extract_crop_for_preview(
                    self.loaded_slide,
                    plan.proposal,
                    crop_level=self.crop_level,
                    slide_handle=slide_handle,
                )
                tissue = plan.proposal.tissue_mask_final
                artifact = plan.proposal.artifact_mask_final
                if tissue is None:
                    tissue = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
                if artifact is None:
                    artifact = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
                tissue = self._resize_mask_to_shape(tissue, crop_rgb.shape[:2])
                artifact = self._resize_mask_to_shape(artifact, crop_rgb.shape[:2])
                if plan.proposal.mirror_enabled:
                    crop_rgb = crop_rgb[:, ::-1, :]
                    tissue = tissue[:, ::-1]
                    artifact = artifact[:, ::-1]

                payload = build_export_payload(crop_rgb, tissue, artifact)
                export_hash = hashlib.sha1(
                    json.dumps(
                        {
                            "section_uid": plan.section_uid,
                            "revision_id": plan.revision_id,
                            "bbox": plan.proposal.bbox_dict(),
                            "mirror_enabled": plan.proposal.mirror_enabled,
                            "profile_name": self.profile_name,
                            "crop_level": self.crop_level,
                        },
                        sort_keys=True,
                ).encode("utf-8")
                ).hexdigest()
                metadata = self._build_metadata(
                    plan,
                    crop_rgb,
                    payload["tissue_mask_final"],
                    payload["artifact_mask_final"],
                    payload["usable_tissue_mask"],
                    export_hash,
                )
                work_queue.put(
                    {
                        "label": plan.proposal.label,
                        "section_dir": plan.section_dir,
                        "crop_rgb": crop_rgb,
                        "payload": payload,
                        "metadata": metadata,
                    }
                )

            for _ in writers:
                work_queue.put(None)
            work_queue.join()
            for writer in writers:
                writer.join()

            manifest = {
                "profile_name": self.profile_name,
                "slide_name": self.loaded_slide.slide_name,
                "backend": self.loaded_slide.backend,
                "items": sorted(manifest_entries, key=lambda x: x["label"]),
            }
            (self.export_root / "_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.finished.emit(
                {
                    "export_root": str(self.export_root),
                    "exported": sorted(exported),
                    "skipped_during_write": sorted(skipped),
                    "planned_count": len(self.plan_items),
                }
            )
        except Exception as exc:
            self.failed.emit(f"Export failed: {exc}")
        finally:
            if slide_handle is not None:
                slide_handle.close()
