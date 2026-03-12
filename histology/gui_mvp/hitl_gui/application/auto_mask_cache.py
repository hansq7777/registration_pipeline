from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from ..domain import LoadedSlide, ProposalBox


def _persistent_cache_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        root = Path(local_appdata) / "histology_gui"
    else:
        root = Path.home() / ".cache" / "histology_gui"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slide_identity(slide_path: Path) -> dict[str, object]:
    stat = slide_path.stat()
    return {
        "path": str(slide_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


class AutoMaskResultCache:
    def __init__(self) -> None:
        self.version = "auto_mask_cache_v1"
        self.memory: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.root = _persistent_cache_root() / self.version
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _algorithm_version(self, stain: str, method: str) -> str:
        stain_key = stain.lower()
        if method == "legacy_simple":
            return "gui_legacy_simple_autoseg_v1"
        if method == "hybrid_balanced":
            return "gui_hybrid_balanced_autoseg_v2"
        if stain_key == "nissl":
            return "gui_nissl_tool_baseline_autoseg_v1"
        if stain_key == "gallyas":
            return "gui_contextual_autoseg_v1"
        return "gui_simple_autoseg_v1"

    def _bbox_algorithm_version(self, stain: str) -> str:
        return "gallyas_bbox_hybrid_topfloor55_wide24_v4" if stain.lower() == "gallyas" else "coverage_first_bbox_v1"

    def _key_payload(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
    ) -> dict:
        return {
            "cache_version": self.version,
            "slide_identity": _slide_identity(loaded_slide.slide_path),
            "stain": loaded_slide.stain,
            "mask_method": mask_method,
            "mask_algorithm_version": self._algorithm_version(loaded_slide.stain, mask_method),
            "bbox_algorithm_version": self._bbox_algorithm_version(loaded_slide.stain),
            "crop_level": int(crop_level),
            "target_proposal": {
                "label": proposal.label,
                "proposal_rank": int(proposal.proposal_rank),
                "bbox_overview_xywh": proposal.bbox_dict(),
            },
            "all_proposals_snapshot": [
                {
                    "label": p.label,
                    "proposal_rank": int(p.proposal_rank),
                    "bbox_overview_xywh": p.bbox_dict(),
                }
                for p in all_proposals
            ],
        }

    def _cache_key(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
    ) -> str:
        payload = self._key_payload(
            loaded_slide,
            proposal,
            all_proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def clear(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
    ) -> None:
        key = self._cache_key(
            loaded_slide,
            proposal,
            all_proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        with self._lock:
            self.memory.pop(key, None)
            path = self._cache_path(key)
            if path.exists():
                path.unlink()

    def contains(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
    ) -> bool:
        key = self._cache_key(
            loaded_slide,
            proposal,
            all_proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        with self._lock:
            if key in self.memory:
                return True
            return self._cache_path(key).exists()

    def get(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        key = self._cache_key(
            loaded_slide,
            proposal,
            all_proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        with self._lock:
            if key in self.memory:
                tissue, artifact = self.memory[key]
                return tissue.copy(), artifact.copy()
            path = self._cache_path(key)
            if not path.exists():
                return None
            try:
                with np.load(path) as data:
                    tissue = np.asarray(data["tissue"], dtype=np.uint8)
                    artifact = np.asarray(data["artifact"], dtype=np.uint8)
                self.memory[key] = (tissue, artifact)
                return tissue.copy(), artifact.copy()
            except Exception:
                return None

    def put(
        self,
        loaded_slide: LoadedSlide,
        proposal: ProposalBox,
        all_proposals: list[ProposalBox],
        *,
        crop_level: int,
        mask_method: str,
        tissue: np.ndarray,
        artifact: np.ndarray,
    ) -> None:
        key = self._cache_key(
            loaded_slide,
            proposal,
            all_proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        tissue_u8 = np.asarray(tissue, dtype=np.uint8)
        artifact_u8 = np.asarray(artifact, dtype=np.uint8)
        with self._lock:
            self.memory[key] = (tissue_u8.copy(), artifact_u8.copy())
            path = self._cache_path(key)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, tissue=tissue_u8, artifact=artifact_u8)
            tmp.replace(path)
