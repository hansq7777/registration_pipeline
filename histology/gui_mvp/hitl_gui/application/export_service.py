from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..domain import ExportPlanItem, LoadedSlide
from ..pipeline_adapters import build_export_payload
from ..pipeline_adapters.slide_io import extract_crop_for_preview, open_slide_handle, write_png_lossless_fast


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
                    write_png_lossless_fast(sec_dir / "foreground_rgb_white.png", payload["foreground_rgb_white"], "RGB")
                    write_png_lossless_fast(sec_dir / "foreground_rgb_black.png", payload["foreground_rgb_black"], "RGB")
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
            if self.loaded_slide.backend == "openslide":
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
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                metadata = {
                    "label": plan.proposal.label,
                    "stain": plan.proposal.stain,
                    "sample_id": plan.proposal.sample_id,
                    "section_id": plan.proposal.section_id,
                    "proposal_rank": plan.proposal.proposal_rank,
                    "mirror_enabled": plan.proposal.mirror_enabled,
                    "bbox_overview": plan.proposal.bbox_dict(),
                    "section_uid": plan.section_uid,
                    "revision_id": plan.revision_id,
                    "profile_name": self.profile_name,
                    "export_hash": export_hash,
                }
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
