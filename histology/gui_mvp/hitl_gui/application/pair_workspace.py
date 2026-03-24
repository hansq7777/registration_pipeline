from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .section_workspace import WorkspaceSection, list_workspace_sections


_LABEL_RE = re.compile(r"^(\d+)_(\d+)$")


@dataclass(frozen=True)
class WorkspacePair:
    pair_key: str
    animal_id: int
    myelin_sec: int
    nissl_sec: int
    myelin_item: WorkspaceSection
    nissl_item: WorkspaceSection

    @property
    def display_label(self) -> str:
        return f"{self.animal_id} | {self.myelin_item.label} <-> {self.nissl_item.label}"


def parse_section_label(label: str) -> tuple[int, int] | None:
    match = _LABEL_RE.match(str(label).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def default_pairing_override_path(myelin_root: Path | None, nissl_root: Path | None) -> Path | None:
    roots = [p for p in (myelin_root, nissl_root) if p is not None]
    if not roots:
        return None
    common = Path(os.path.commonpath([str(p.resolve()) for p in roots]))
    return common / "histology_pairing_overrides.json"


def load_pairing_overrides(path: Path | None) -> dict[int, list[tuple[str, str]]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    animals = payload.get("animals")
    if not isinstance(animals, dict):
        return {}
    parsed: dict[int, list[tuple[str, str]]] = {}
    for animal_key, animal_payload in animals.items():
        try:
            animal_id = int(animal_key)
        except Exception:
            continue
        if not isinstance(animal_payload, dict):
            continue
        raw_pairs = animal_payload.get("pairs")
        if not isinstance(raw_pairs, list):
            continue
        pairs: list[tuple[str, str]] = []
        for entry in raw_pairs:
            if not isinstance(entry, dict):
                continue
            myelin_label = str(entry.get("myelin", "")).strip()
            nissl_label = str(entry.get("nissl", "")).strip()
            if not myelin_label or not nissl_label:
                continue
            if parse_section_label(myelin_label) is None or parse_section_label(nissl_label) is None:
                continue
            pairs.append((myelin_label, nissl_label))
        if pairs:
            parsed[animal_id] = pairs
    return parsed


def list_cross_stain_pairs(myelin_root: Path, nissl_root: Path) -> list[WorkspacePair]:
    myelin_sections = [
        item
        for item in list_workspace_sections(myelin_root)
        if item.stain in {"gallyas", "myelin", ""}
    ]
    nissl_sections = [
        item
        for item in list_workspace_sections(nissl_root)
        if item.stain in {"nissl", ""}
    ]

    myelin_index: dict[int, list[tuple[int, WorkspaceSection]]] = {}
    nissl_index: dict[int, list[tuple[int, WorkspaceSection]]] = {}
    myelin_by_label: dict[str, WorkspaceSection] = {}
    nissl_by_label: dict[str, WorkspaceSection] = {}
    for item in myelin_sections:
        parsed = parse_section_label(item.label)
        if parsed is None:
            continue
        animal_id, sec = parsed
        myelin_index.setdefault(animal_id, []).append((sec, item))
        myelin_by_label[item.label] = item
    for item in nissl_sections:
        parsed = parse_section_label(item.label)
        if parsed is None:
            continue
        animal_id, sec = parsed
        nissl_index.setdefault(animal_id, []).append((sec, item))
        nissl_by_label[item.label] = item

    pairing_overrides = load_pairing_overrides(default_pairing_override_path(myelin_root, nissl_root))
    pairs: list[WorkspacePair] = []
    for animal_id in sorted(set(myelin_index) | set(nissl_index)):
        used_myelin_labels: set[str] = set()
        used_nissl_labels: set[str] = set()
        override_pairs = pairing_overrides.get(animal_id)
        if override_pairs:
            for myelin_label, nissl_label in override_pairs:
                myelin_item = myelin_by_label.get(myelin_label)
                nissl_item = nissl_by_label.get(nissl_label)
                if myelin_item is None or nissl_item is None:
                    continue
                myelin_parsed = parse_section_label(myelin_label)
                nissl_parsed = parse_section_label(nissl_label)
                if myelin_parsed is None or nissl_parsed is None:
                    continue
                pair_key = f"{myelin_item.label}__{nissl_item.label}"
                pairs.append(
                    WorkspacePair(
                        pair_key=pair_key,
                        animal_id=animal_id,
                        myelin_sec=myelin_parsed[1],
                        nissl_sec=nissl_parsed[1],
                        myelin_item=myelin_item,
                        nissl_item=nissl_item,
                    )
                )
                used_myelin_labels.add(myelin_item.label)
                used_nissl_labels.add(nissl_item.label)
        nissl_by_sec = {sec: item for sec, item in sorted(nissl_index.get(animal_id, []))}
        for myelin_sec, myelin_item in sorted(myelin_index.get(animal_id, [])):
            if myelin_item.label in used_myelin_labels:
                continue
            for delta in (-1, 0, 1):
                nissl_sec = myelin_sec + delta
                nissl_item = nissl_by_sec.get(nissl_sec)
                if nissl_item is None:
                    continue
                if nissl_item.label in used_nissl_labels:
                    continue
                pair_key = f"{myelin_item.label}__{nissl_item.label}"
                pairs.append(
                    WorkspacePair(
                        pair_key=pair_key,
                        animal_id=animal_id,
                        myelin_sec=myelin_sec,
                        nissl_sec=nissl_sec,
                        myelin_item=myelin_item,
                        nissl_item=nissl_item,
                    )
                )
                used_myelin_labels.add(myelin_item.label)
                used_nissl_labels.add(nissl_item.label)
                break
    return pairs


def default_pair_registry_path(myelin_root: Path | None, nissl_root: Path | None) -> Path | None:
    roots = [p for p in (myelin_root, nissl_root) if p is not None]
    if not roots:
        return None
    common = Path(os.path.commonpath([str(p.resolve()) for p in roots]))
    return common / "histology_pair_qc_registry.json"


def default_pair_registration_masks_root(myelin_root: Path | None, nissl_root: Path | None) -> Path | None:
    roots = [p for p in (myelin_root, nissl_root) if p is not None]
    if not roots:
        return None
    common = Path(os.path.commonpath([str(p.resolve()) for p in roots]))
    return common / "histology_pair_registration_masks"


def pair_registration_mask_paths(root: Path | None, pair_key: str) -> dict[str, Path]:
    if root is None:
        return {}
    pair_dir = root / pair_key
    return {
        "myelin": pair_dir / "myelin_mask_labels.png",
        "nissl": pair_dir / "nissl_mask_labels.png",
    }


def load_pair_registry(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    pairs = data.get("pairs")
    return pairs if isinstance(pairs, dict) else {}


def save_pair_registry(path: Path, records: dict[str, Any]) -> None:
    payload = {
        "pairs": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
