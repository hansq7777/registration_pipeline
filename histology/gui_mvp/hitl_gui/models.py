from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Stain(str, Enum):
    NISSL = "nissl"
    GALLYAS = "gallyas"


class ReviewStatus(str, Enum):
    PROPOSED = "proposed"
    PROPOSAL_REVIEWED = "proposal_reviewed"
    MASK_REVIEWED = "mask_reviewed"
    PAIR_REVIEWED = "pair_reviewed"
    ORIENTATION_LOCKED = "orientation_locked"
    EXPORT_READY = "export_ready"
    EXPORTED = "exported"


class MirrorState(str, Enum):
    ORIGINAL = "original"
    MIRRORED_LR = "mirrored_lr"


class PairStatus(str, Enum):
    UNPAIRED = "unpaired"
    SUGGESTED = "suggested"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class ProjectRecord:
    project_id: str
    project_name: str
    workspace_root: Path
    nissl_root: Path
    gallyas_root: Path
    default_review_profile: str = "review_mask"
    default_cyclegan_profile: str = "cyclegan_train"
    default_registration_profile: str = "registration_fullres"


@dataclass
class SlideRecord:
    slide_id: str
    project_id: str
    stain: Stain
    source_path: Path
    source_name: str
    readable: bool = True
    level_count: int = 0
    width_level0: int = 0
    height_level0: int = 0
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None
    focal_metadata_json: str = "{}"


@dataclass
class SectionRecord:
    section_uid: str
    project_id: str
    slide_id: str
    stain: Stain
    sample_id: str
    section_id: int
    proposal_rank: int
    proposal_method: str
    proposal_bbox_overview_json: str
    proposal_bbox_level0_json: str
    proposal_qc_flags_json: str = "{}"
    crop_profile: str = "review_mask"
    crop_bbox_level0_json: str = "{}"
    crop_canvas_w: int = 0
    crop_canvas_h: int = 0
    crop_level: int = 0
    target_mpp: Optional[float] = None
    mirror_state: MirrorState = MirrorState.ORIGINAL
    orientation_method: str = "unset"
    orientation_score_original: Optional[float] = None
    orientation_score_mirror: Optional[float] = None
    orientation_recommended: Optional[str] = None
    orientation_ambiguous: bool = False
    pair_status: PairStatus = PairStatus.UNPAIRED
    review_status: ReviewStatus = ReviewStatus.PROPOSED
    manual_review_status: str = "unreviewed"
    manual_mask_version: int = 0
    notes: str = ""


@dataclass
class SectionFileRecord:
    file_id: str
    section_uid: str
    file_role: str
    path: Path
    profile_name: Optional[str] = None
    checksum: Optional[str] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    metadata_json: str = "{}"


@dataclass
class PairRecord:
    pair_id: str
    project_id: str
    nissl_section_uid: str
    gallyas_section_uid: str
    sample_id: str
    section_delta: int
    pair_score_shape: Optional[float] = None
    pair_score_size: Optional[float] = None
    pair_score_orientation: Optional[float] = None
    pair_score_total: Optional[float] = None
    pair_status: PairStatus = PairStatus.SUGGESTED
    manual_override: bool = False
    notes: str = ""


@dataclass
class RevisionRecord:
    revision_id: str
    section_uid: str
    revision_type: str
    author: str
    timestamp: str
    base_revision_id: Optional[str] = None
    delta_json: str = "{}"
    note: str = ""


@dataclass
class SectionViewState:
    section_uid: str
    active_layer: str = "tissue_mask_final"
    active_tool: str = "brush"
    mask_opacity: float = 0.35
    zoom_level: float = 1.0
    dirty: bool = False
    visible_layers: list[str] = field(
        default_factory=lambda: [
            "raw_crop",
            "tissue_mask_final",
            "artifact_mask_final",
            "usable_tissue_mask",
        ]
    )
