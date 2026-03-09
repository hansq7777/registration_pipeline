from __future__ import annotations

from enum import Enum


class AppPage(str, Enum):
    PROJECT = "project"
    SLIDE_REVIEW = "slide_review"
    SECTION_REVIEW = "section_review"
    PAIR_REVIEW = "pair_review"
    EXPORT_MANAGER = "export_manager"


class EventName(str, Enum):
    OPEN_PROJECT = "open_project"
    IMPORT_RUN = "import_run"
    SELECT_SLIDE = "select_slide"
    SELECT_SECTION = "select_section"
    SELECT_PAIR = "select_pair"
    ACCEPT_PROPOSAL = "accept_proposal"
    REJECT_PROPOSAL = "reject_proposal"
    SAVE_MASK_REVISION = "save_mask_revision"
    SAVE_ORIENTATION = "save_orientation"
    SAVE_PAIR_DECISION = "save_pair_decision"
    EXPORT_PROFILE = "export_profile"

