PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    nissl_root TEXT NOT NULL,
    gallyas_root TEXT NOT NULL,
    workspace_root TEXT NOT NULL,
    default_review_profile TEXT NOT NULL,
    default_cyclegan_profile TEXT NOT NULL,
    default_registration_profile TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slides (
    slide_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    stain TEXT NOT NULL,
    sample_group TEXT,
    source_path TEXT NOT NULL,
    source_name TEXT NOT NULL,
    readable INTEGER NOT NULL DEFAULT 1,
    level_count INTEGER,
    width_level0 INTEGER,
    height_level0 INTEGER,
    mpp_x REAL,
    mpp_y REAL,
    focal_metadata_json TEXT NOT NULL DEFAULT '{}',
    import_status TEXT NOT NULL DEFAULT 'imported',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sections (
    section_uid TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    stain TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    proposal_rank INTEGER NOT NULL,
    proposal_method TEXT NOT NULL,
    proposal_bbox_overview_json TEXT NOT NULL,
    proposal_bbox_level0_json TEXT NOT NULL,
    proposal_qc_flags_json TEXT NOT NULL DEFAULT '{}',
    crop_profile TEXT NOT NULL,
    crop_bbox_level0_json TEXT NOT NULL,
    crop_canvas_w INTEGER NOT NULL,
    crop_canvas_h INTEGER NOT NULL,
    crop_level INTEGER NOT NULL,
    target_mpp REAL,
    mirror_state TEXT NOT NULL DEFAULT 'original',
    orientation_method TEXT NOT NULL DEFAULT 'unset',
    orientation_score_original REAL,
    orientation_score_mirror REAL,
    orientation_recommended TEXT,
    orientation_ambiguous INTEGER NOT NULL DEFAULT 0,
    pair_status TEXT NOT NULL DEFAULT 'unpaired',
    review_status TEXT NOT NULL DEFAULT 'proposed',
    manual_review_status TEXT NOT NULL DEFAULT 'unreviewed',
    manual_mask_version INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
    FOREIGN KEY(slide_id) REFERENCES slides(slide_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS section_files (
    file_id TEXT PRIMARY KEY,
    section_uid TEXT NOT NULL,
    file_role TEXT NOT NULL,
    profile_name TEXT,
    path TEXT NOT NULL,
    checksum TEXT,
    width_px INTEGER,
    height_px INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(section_uid) REFERENCES sections(section_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pairs (
    pair_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    nissl_section_uid TEXT NOT NULL,
    gallyas_section_uid TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    section_delta INTEGER NOT NULL,
    pair_score_shape REAL,
    pair_score_size REAL,
    pair_score_orientation REAL,
    pair_score_total REAL,
    pair_status TEXT NOT NULL DEFAULT 'suggested',
    manual_override INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
    FOREIGN KEY(nissl_section_uid) REFERENCES sections(section_uid) ON DELETE CASCADE,
    FOREIGN KEY(gallyas_section_uid) REFERENCES sections(section_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    section_uid TEXT NOT NULL,
    revision_type TEXT NOT NULL,
    author TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    base_revision_id TEXT,
    delta_json TEXT NOT NULL DEFAULT '{}',
    note TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(section_uid) REFERENCES sections(section_uid) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_slides_project ON slides(project_id);
CREATE INDEX IF NOT EXISTS idx_sections_project ON sections(project_id);
CREATE INDEX IF NOT EXISTS idx_sections_slide ON sections(slide_id);
CREATE INDEX IF NOT EXISTS idx_sections_sample_section ON sections(sample_id, section_id);
CREATE INDEX IF NOT EXISTS idx_section_files_section_uid ON section_files(section_uid);
CREATE INDEX IF NOT EXISTS idx_pairs_project ON pairs(project_id);
CREATE INDEX IF NOT EXISTS idx_pairs_sample_id ON pairs(sample_id);
CREATE INDEX IF NOT EXISTS idx_revisions_section_uid ON revisions(section_uid);
