from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..domain import LoadedSlide, ProposalBox


@lru_cache(maxsize=1)
def load_histology_tool_module() -> Any:
    here = Path(__file__).resolve()
    tool_path = here.parents[3] / "tools" / "run_ndpi_review_experiment.py"
    spec = importlib.util.spec_from_file_location("histology_ndpi_tool_singleton", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load histology tool module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proposal_to_tool_candidate(proposal: ProposalBox, rank: int | None = None):
    tool = load_histology_tool_module()
    candidate_rank = proposal.proposal_rank if rank is None else rank
    section = tool.SectionLabel(
        stain=proposal.stain,
        sample_id=proposal.sample_id or "manual",
        section_id=int(proposal.section_id),
    )
    return tool.CandidateBox(
        candidate_rank=int(candidate_rank),
        x=int(proposal.x),
        y=int(proposal.y),
        w=int(proposal.w),
        h=int(proposal.h),
        area=int(proposal.w * proposal.h),
        cx=float(proposal.x + proposal.w / 2.0),
        cy=float(proposal.y + proposal.h / 2.0),
        touches_border=False,
        section=section,
    )


def proposal_crop_rect_overview_gui(loaded_slide: LoadedSlide, proposal: ProposalBox) -> tuple[int, int, int, int]:
    tool = load_histology_tool_module()
    overview_rgb = np.asarray(loaded_slide.overview)
    candidate = proposal_to_tool_candidate(proposal)
    return tool.proposal_crop_rect_overview(candidate, overview_rgb, loaded_slide.stain)


def proposal_bbox_level0_gui(loaded_slide: LoadedSlide, proposal: ProposalBox) -> tuple[int, int, int, int]:
    tool = load_histology_tool_module()
    crop_rect_overview = proposal_crop_rect_overview_gui(loaded_slide, proposal)
    overview_downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    x1, y1, x2, y2 = crop_rect_overview
    x0 = int(round(x1 * overview_downsample))
    y0 = int(round(y1 * overview_downsample))
    w0 = int(round((x2 - x1) * overview_downsample))
    h0 = int(round((y2 - y1) * overview_downsample))
    w0 = min(w0, loaded_slide.level_dimensions[0][0] - x0)
    h0 = min(h0, loaded_slide.level_dimensions[0][1] - y0)
    return x0, y0, w0, h0
