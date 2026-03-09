from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain import LoadedSlide, ProposalBox
from ..pipeline_adapters.slide_io import extract_crop_for_preview, open_slide_handle


class SlideSessionCache:
    def __init__(self) -> None:
        self.current_slide: Optional[LoadedSlide] = None
        self.slide_handle = None
        self.proposal_crop_cache: dict[tuple, np.ndarray] = {}
        self.section_raw_crop_cache: dict[tuple, np.ndarray] = {}
        self.preview_thumbnail_cache: dict[tuple, np.ndarray] = {}

    def close(self) -> None:
        if self.slide_handle is not None:
            try:
                self.slide_handle.close()
            except Exception:
                pass
        self.slide_handle = None
        self.current_slide = None
        self.proposal_crop_cache.clear()
        self.section_raw_crop_cache.clear()
        self.preview_thumbnail_cache.clear()

    def set_slide(self, loaded_slide: LoadedSlide) -> None:
        self.close()
        self.current_slide = loaded_slide
        self.slide_handle = open_slide_handle(loaded_slide)

    def _crop_key(self, proposal: ProposalBox, crop_level: int, purpose: str) -> tuple:
        return (
            purpose,
            proposal.label,
            proposal.x,
            proposal.y,
            proposal.w,
            proposal.h,
            crop_level,
            proposal.mirror_enabled,
        )

    def get_preview_crop(self, proposal: ProposalBox, crop_level: int) -> np.ndarray:
        key = self._crop_key(proposal, crop_level, "preview")
        if key not in self.preview_thumbnail_cache:
            self.preview_thumbnail_cache[key] = extract_crop_for_preview(
                self.current_slide,
                proposal,
                crop_level=crop_level,
                slide_handle=self.slide_handle,
            )
        return self.preview_thumbnail_cache[key].copy()

    def get_section_crop(self, proposal: ProposalBox, crop_level: int) -> np.ndarray:
        key = self._crop_key(proposal, crop_level, "section")
        if key not in self.section_raw_crop_cache:
            self.section_raw_crop_cache[key] = extract_crop_for_preview(
                self.current_slide,
                proposal,
                crop_level=crop_level,
                slide_handle=self.slide_handle,
            )
        return self.section_raw_crop_cache[key].copy()

    def invalidate_proposal(self, proposal: ProposalBox) -> None:
        prefixes = {
            ("preview", proposal.label),
            ("section", proposal.label),
        }
        for cache in [self.preview_thumbnail_cache, self.section_raw_crop_cache, self.proposal_crop_cache]:
            remove_keys = [key for key in cache if (key[0], key[1]) in prefixes]
            for key in remove_keys:
                cache.pop(key, None)
