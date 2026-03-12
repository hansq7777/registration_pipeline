from __future__ import annotations

import numpy as np

from .application.auto_mask_cache import AutoMaskResultCache
from .domain import LoadedSlide, ProposalBox
from .pipeline_adapters.segmentation_adapter import build_export_payload, compute_auto_masks, parse_slide_labels, propose_from_overview
from .pipeline_adapters.slide_io import (
    cleanup_session_temp_root,
    extract_crop_for_preview,
    load_slide_bundle,
    openslide_available,
    write_png_lossless_fast,
)


_AUTO_MASK_CACHE = AutoMaskResultCache()


def load_ndpi_with_proposals(slide_path):
    stain, labels = parse_slide_labels(slide_path.stem)
    loaded = load_slide_bundle(slide_path, stain)
    loaded.expected_labels = [label.short_label for label in labels]
    loaded.proposals = propose_from_overview(slide_path, stain, labels, np.asarray(loaded.overview))
    return loaded


def generate_masks_for_proposal(loaded_slide: LoadedSlide, proposal: ProposalBox, crop_level: int = 3):
    crop = extract_crop_for_preview(loaded_slide, proposal, crop_level=crop_level)
    cached = _AUTO_MASK_CACHE.get(
        loaded_slide,
        proposal,
        loaded_slide.proposals,
        crop_level=crop_level,
        mask_method=proposal.mask_preset,
    )
    if cached is None:
        tissue, artifact = compute_auto_masks(
            crop,
            loaded_slide.stain,
            method=proposal.mask_preset,
            loaded_slide=loaded_slide,
            target_proposal=proposal,
            all_proposals=loaded_slide.proposals,
            crop_level=crop_level,
        )
        _AUTO_MASK_CACHE.put(
            loaded_slide,
            proposal,
            loaded_slide.proposals,
            crop_level=crop_level,
            mask_method=proposal.mask_preset,
            tissue=tissue,
            artifact=artifact,
        )
    else:
        tissue, artifact = cached
    proposal.tissue_mask_auto = tissue.copy()
    proposal.artifact_mask_auto = artifact.copy()
    proposal.tissue_mask_final = tissue.copy()
    proposal.artifact_mask_final = artifact.copy()
    return crop, tissue, artifact


__all__ = [
    "LoadedSlide",
    "ProposalBox",
    "build_export_payload",
    "cleanup_session_temp_root",
    "compute_auto_masks",
    "extract_crop_for_preview",
    "generate_masks_for_proposal",
    "load_ndpi_with_proposals",
    "openslide_available",
    "write_png_lossless_fast",
]
