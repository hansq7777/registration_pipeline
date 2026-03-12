from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class ProposalBox:
    label: str
    stain: str
    sample_id: str
    section_id: int
    proposal_rank: int
    x: int
    y: int
    w: int
    h: int
    tissue_mask_auto: Optional[np.ndarray] = None
    artifact_mask_auto: Optional[np.ndarray] = None
    tissue_mask_final: Optional[np.ndarray] = None
    artifact_mask_final: Optional[np.ndarray] = None
    mask_work_level: Optional[int] = None
    mask_work_shape: Optional[tuple[int, int]] = None
    mask_preset: str = "latest_contextual"
    mirror_enabled: bool = False
    latest_revision_id: Optional[str] = None
    notes: str = ""

    def bbox_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class LoadedSlide:
    slide_path: Path
    slide_name: str
    stain: str
    expected_labels: list[str]
    label_preview: Image.Image
    overview: Image.Image
    proposals: list[ProposalBox]
    level_count: int
    overview_level: int
    overview_size: tuple[int, int]
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    backend: str = "openslide"
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None
    objective_power: Optional[float] = None
    temp_proxy_dir: Optional[Path] = None
    fallback_reason: str = ""
    tifffile_midres_page_index: Optional[int] = None
    tifffile_midres_downsample: Optional[float] = None
    tifffile_overview_scale_from_midres: Optional[float] = None


@dataclass
class SlideLoadResult:
    slide: LoadedSlide
    messages: list[str] = field(default_factory=list)


@dataclass
class ExportPlanItem:
    proposal: ProposalBox
    section_uid: str
    revision_id: Optional[str]
    section_dir: Path
    expected_label: Optional[str] = None
    manual_mask_version: int = 0
    revision_count: int = 0
    review_notes: str = ""
    review_status: str = "proposed"
    mirror_state: str = "original"
