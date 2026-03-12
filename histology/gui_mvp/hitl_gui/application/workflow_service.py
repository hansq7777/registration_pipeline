from __future__ import annotations

from pathlib import Path

import numpy as np

from ..db import transaction
from ..domain import ExportPlanItem, LoadedSlide, ProposalBox, SlideLoadResult
from ..pipeline_adapters import compute_auto_masks, load_slide_bundle, parse_slide_labels, propose_from_overview
from ..pipeline_adapters.segmentation_adapter import default_mask_preset_for_stain
from ..repositories import PairRepository, ProjectRepository, RevisionRepository, SectionRepository, SlideRepository
from .auto_mask_cache import AutoMaskResultCache
from .export_service import ExportWorker
from .session_cache import SlideSessionCache


class WorkflowService:
    def __init__(
        self,
        *,
        conn,
        project_repository: ProjectRepository,
        slide_repository: SlideRepository,
        section_repository: SectionRepository,
        revision_repository: RevisionRepository,
        pair_repository: PairRepository,
        project_id: str,
    ) -> None:
        self.conn = conn
        self.project_repository = project_repository
        self.slide_repository = slide_repository
        self.section_repository = section_repository
        self.revision_repository = revision_repository
        self.pair_repository = pair_repository
        self.project_id = project_id
        self.session_cache = SlideSessionCache()
        self.auto_mask_cache = AutoMaskResultCache()

    @property
    def current_slide(self) -> LoadedSlide | None:
        return self.session_cache.current_slide

    def preview_crop_level_for_slide(self, slide: LoadedSlide) -> int:
        return min(5, slide.level_count - 1)

    def mask_work_crop_level_for_slide(self, slide: LoadedSlide) -> int:
        return min(4, slide.level_count - 1)

    def export_crop_level_for_slide(self, slide: LoadedSlide) -> int:
        return min(3, slide.level_count - 1)

    def list_ndpi_files(self, folder: Path) -> list[Path]:
        return sorted(p for p in folder.glob("*.ndpi") if not p.name.startswith("._"))

    def load_slide(self, slide_path: Path) -> SlideLoadResult:
        stain, labels = parse_slide_labels(slide_path.stem)
        loaded = load_slide_bundle(slide_path, stain)
        loaded.expected_labels = [label.short_label for label in labels]
        loaded.proposals = propose_from_overview(slide_path, stain, labels, np.asarray(loaded.overview))

        with transaction(self.conn):
            slide_id = f"{loaded.stain}_{slide_path.stem}"
            self.slide_repository.upsert_slide(
                project_id=self.project_id,
                slide_id=slide_id,
                stain=loaded.stain,
                source_path=str(slide_path),
                source_name=slide_path.name,
                readable=True,
                level_count=loaded.level_count,
                width_level0=loaded.level_dimensions[0][0],
                height_level0=loaded.level_dimensions[0][1],
                focal_metadata_json="{}",
            )
            for proposal in loaded.proposals:
                self.section_repository.upsert_proposal(
                    project_id=self.project_id,
                    slide_id=slide_id,
                    section_uid=self.section_uid(loaded, proposal),
                    stain=proposal.stain,
                    sample_id=proposal.sample_id,
                    section_id=proposal.section_id,
                    proposal_rank=proposal.proposal_rank,
                    bbox_overview=proposal.bbox_dict(),
                )

        self.session_cache.set_slide(loaded)
        messages = [
            f"slide: {loaded.slide_name}",
            f"stain: {loaded.stain}",
            f"backend: {loaded.backend}",
            f"expected labels: {', '.join(loaded.expected_labels)}",
            f"fallback_reason: {loaded.fallback_reason or 'none'}",
            f"temp_proxy_dir: {loaded.temp_proxy_dir or 'none'}",
        ]
        return SlideLoadResult(slide=loaded, messages=messages)

    def section_uid(self, slide: LoadedSlide, proposal: ProposalBox) -> str:
        slide_short = slide.slide_name.replace(".ndpi", "").replace(";", "_")
        return f"{proposal.stain}_{proposal.sample_id}_{proposal.section_id}__{slide_short}__r{proposal.proposal_rank:02d}"

    def get_preview_crop(self, idx: int, crop_level: int) -> np.ndarray:
        slide = self.require_slide()
        return self.session_cache.get_preview_crop(slide.proposals[idx], crop_level)

    def update_proposal_bbox(self, idx: int, *, x: int, y: int, w: int, h: int) -> ProposalBox:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        proposal.x = int(x)
        proposal.y = int(y)
        proposal.w = int(max(1, w))
        proposal.h = int(max(1, h))
        self.session_cache.invalidate_proposal(proposal)
        with transaction(self.conn):
            self.section_repository.upsert_proposal(
                project_id=self.project_id,
                slide_id=f"{slide.stain}_{slide.slide_path.stem}",
                section_uid=self.section_uid(slide, proposal),
                stain=proposal.stain,
                sample_id=proposal.sample_id,
                section_id=proposal.section_id,
                proposal_rank=proposal.proposal_rank,
                bbox_overview=proposal.bbox_dict(),
            )
        return proposal

    def add_proposal(self) -> int:
        slide = self.require_slide()
        ow, oh = slide.overview.size
        w = max(120, ow // 6)
        h = max(120, oh // 6)
        x = max(0, (ow - w) // 2)
        y = max(0, (oh - h) // 2)
        rank = len(slide.proposals) + 1
        proposal = ProposalBox(
            label=f"manual_{rank}",
            stain=slide.stain,
            sample_id="manual",
            section_id=rank,
            proposal_rank=rank,
            x=x,
            y=y,
            w=w,
            h=h,
            mask_preset=default_mask_preset_for_stain(slide.stain),
        )
        slide.proposals.append(proposal)
        with transaction(self.conn):
            self.section_repository.upsert_proposal(
                project_id=self.project_id,
                slide_id=f"{slide.stain}_{slide.slide_path.stem}",
                section_uid=self.section_uid(slide, proposal),
                stain=proposal.stain,
                sample_id=proposal.sample_id,
                section_id=proposal.section_id,
                proposal_rank=proposal.proposal_rank,
                bbox_overview=proposal.bbox_dict(),
            )
        return len(slide.proposals) - 1

    def remove_proposal(self, idx: int) -> None:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        self.session_cache.invalidate_proposal(proposal)
        with transaction(self.conn):
            self.section_repository.delete_section(self.section_uid(slide, proposal))
        del slide.proposals[idx]

    def ensure_proposal_count(self, target_count: int) -> None:
        slide = self.require_slide()
        while len(slide.proposals) < target_count:
            self.add_proposal()
        while len(slide.proposals) > target_count and slide.proposals:
            self.remove_proposal(len(slide.proposals) - 1)

    def generate_mask_preview(
        self,
        idx: int,
        crop_level: int,
        *,
        mask_method: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        if mask_method is None:
            mask_method = proposal.mask_preset
        proposal.mask_preset = mask_method
        crop_rgb = self.session_cache.get_section_crop(proposal, crop_level)
        cached = self.auto_mask_cache.get(
            slide,
            proposal,
            slide.proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )
        if cached is None:
            tissue, artifact = compute_auto_masks(
                crop_rgb,
                slide.stain,
                method=mask_method,
                loaded_slide=slide,
                target_proposal=proposal,
                all_proposals=slide.proposals,
                crop_level=crop_level,
            )
            self.auto_mask_cache.put(
                slide,
                proposal,
                slide.proposals,
                crop_level=crop_level,
                mask_method=mask_method,
                tissue=tissue,
                artifact=artifact,
            )
        else:
            tissue, artifact = cached
        proposal.tissue_mask_auto = tissue.copy()
        proposal.artifact_mask_auto = artifact.copy()
        proposal.tissue_mask_final = tissue.copy()
        proposal.artifact_mask_final = artifact.copy()
        proposal.mask_work_level = int(crop_level)
        proposal.mask_work_shape = tuple(int(x) for x in tissue.shape[:2])
        return crop_rgb, tissue, artifact

    def has_section_crop_cached(self, idx: int, crop_level: int) -> bool:
        slide = self.require_slide()
        return self.session_cache.has_section_crop(slide.proposals[idx], crop_level)

    def store_section_crop(self, idx: int, crop_level: int, crop_rgb: np.ndarray) -> None:
        slide = self.require_slide()
        self.session_cache.store_section_crop(slide.proposals[idx], crop_level, crop_rgb)

    def has_auto_mask_cached(self, idx: int, crop_level: int, *, mask_method: str | None = None) -> bool:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        if mask_method is None:
            mask_method = proposal.mask_preset
        return self.auto_mask_cache.contains(
            slide,
            proposal,
            slide.proposals,
            crop_level=crop_level,
            mask_method=mask_method,
        )

    def get_section_for_edit(
        self,
        idx: int,
        crop_level: int,
        *,
        mask_method: str | None = None,
    ) -> tuple[ProposalBox, np.ndarray, np.ndarray, np.ndarray]:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        if mask_method is None:
            mask_method = proposal.mask_preset
        proposal.mask_preset = mask_method
        crop_rgb = self.session_cache.get_section_crop(proposal, crop_level)
        if proposal.tissue_mask_final is None or proposal.artifact_mask_final is None:
            cached = self.auto_mask_cache.get(
                slide,
                proposal,
                slide.proposals,
                crop_level=crop_level,
                mask_method=mask_method,
            )
            if cached is None:
                tissue, artifact = compute_auto_masks(
                    crop_rgb,
                    slide.stain,
                    method=mask_method,
                    loaded_slide=slide,
                    target_proposal=proposal,
                    all_proposals=slide.proposals,
                    crop_level=crop_level,
                )
                self.auto_mask_cache.put(
                    slide,
                    proposal,
                    slide.proposals,
                    crop_level=crop_level,
                    mask_method=mask_method,
                    tissue=tissue,
                    artifact=artifact,
                )
            else:
                tissue, artifact = cached
            proposal.tissue_mask_auto = tissue.copy()
            proposal.artifact_mask_auto = artifact.copy()
            proposal.tissue_mask_final = tissue.copy()
            proposal.artifact_mask_final = artifact.copy()
            proposal.mask_work_level = int(crop_level)
            proposal.mask_work_shape = tuple(int(x) for x in tissue.shape[:2])
        return proposal, crop_rgb, proposal.tissue_mask_final.copy(), proposal.artifact_mask_final.copy()

    def save_revision(self, idx: int, *, tissue_mask: np.ndarray, artifact_mask: np.ndarray, mirror_enabled: bool, notes: str = "") -> str:
        slide = self.require_slide()
        proposal = slide.proposals[idx]
        proposal.tissue_mask_final = tissue_mask.copy()
        proposal.artifact_mask_final = artifact_mask.copy()
        proposal.mask_work_shape = tuple(int(x) for x in tissue_mask.shape[:2])
        proposal.mirror_enabled = mirror_enabled
        proposal.notes = notes
        section_uid = self.section_uid(slide, proposal)
        with transaction(self.conn):
            revision_id = self.revision_repository.create_mask_revision(
                section_uid=section_uid,
                tissue_mask=tissue_mask,
                artifact_mask=artifact_mask,
                mirror_enabled=mirror_enabled,
                bbox_overview=proposal.bbox_dict(),
                notes=notes,
            )
            current_version = self.section_repository.get_manual_mask_version(section_uid)
            self.section_repository.update_review_state(
                section_uid=section_uid,
                mirror_state="mirrored_lr" if mirror_enabled else "original",
                review_status="mask_reviewed",
                manual_mask_version=current_version + 1,
                notes=notes,
            )
        proposal.latest_revision_id = revision_id
        return revision_id

    def latest_revision_id(self, proposal: ProposalBox) -> str | None:
        slide = self.require_slide()
        return proposal.latest_revision_id or self.revision_repository.get_latest_revision_id(self.section_uid(slide, proposal))

    def plan_export(self, export_root: Path) -> tuple[list[ExportPlanItem], list[str]]:
        slide = self.require_slide()
        plan_items: list[ExportPlanItem] = []
        skipped: list[str] = []
        for proposal in slide.proposals:
            section_dir = export_root / proposal.label
            if section_dir.exists():
                skipped.append(proposal.label)
                continue
            section_uid = self.section_uid(slide, proposal)
            section_state = self.section_repository.get_section_state(section_uid)
            plan_items.append(
                ExportPlanItem(
                    proposal=proposal,
                    section_uid=section_uid,
                    revision_id=self.latest_revision_id(proposal),
                    section_dir=section_dir,
                    expected_label=proposal.label if proposal.label in slide.expected_labels else None,
                    manual_mask_version=section_state["manual_mask_version"],
                    revision_count=self.revision_repository.count_revisions(section_uid),
                    review_notes=section_state["notes"],
                    review_status=section_state["review_status"],
                    mirror_state=section_state["mirror_state"],
                )
            )
        return plan_items, skipped

    def create_export_worker(self, export_root: Path, crop_level: int, profile_name: str = "review_mask") -> ExportWorker:
        plan_items, _ = self.plan_export(export_root)
        return ExportWorker(self.require_slide(), plan_items, export_root, crop_level, profile_name=profile_name)

    def require_slide(self) -> LoadedSlide:
        if self.session_cache.current_slide is None:
            raise RuntimeError("No slide loaded")
        return self.session_cache.current_slide
