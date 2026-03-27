from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


NANOZOOMER_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans")
MYELIN_ROOT = NANOZOOMER_ROOT / "20250327 rat myelin quantification" / "Tissue&Masks"
NISSL_ROOT = NANOZOOMER_ROOT / "20250424 Nissl cytoarchitectonic counterpart" / "Tissue&Masks"
PAIR_REGISTRY = NANOZOOMER_ROOT / "histology_pair_qc_registry.json"
PAIR_MASKS_ROOT = NANOZOOMER_ROOT / "histology_pair_registration_masks"


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    return pairs if isinstance(pairs, dict) else {}


def registry_relpath_to_abs(relpath: str) -> Path:
    rel_norm = str(relpath).replace("\\", "/").lstrip("/")
    return NANOZOOMER_ROOT / Path(rel_norm)


def load_mask_labels(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def component_rank_map(mask_labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    union = (mask_labels > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    entries: list[tuple[int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            entries.append((label_idx, area))
    entries.sort(key=lambda x: x[1], reverse=True)
    rank_to_label = {rank: label for rank, (label, _) in enumerate(entries, start=1)}
    return labels, rank_to_label


def apply_whole_flip(rgb: np.ndarray, labels: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return rgb, labels
    return rgb[:, ::-1, :].copy(), labels[:, ::-1].copy()


def apply_group_flips(
    rgb: np.ndarray,
    labels: np.ndarray,
    component_groups: dict[str, list[int]] | dict[int, list[int]] | None,
    group_flip_lr: dict[str, bool] | dict[int, bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not component_groups or not group_flip_lr:
        return rgb, labels

    labels_cc, rank_to_label = component_rank_map(labels)
    out_rgb = rgb.copy()
    out_labels = labels.copy()

    for raw_group_id, raw_enabled in dict(group_flip_lr).items():
        if not bool(raw_enabled):
            continue
        group_id = str(raw_group_id)
        ranks = component_groups.get(group_id) if isinstance(component_groups, dict) else None
        if not isinstance(ranks, list):
            continue
        for raw_rank in ranks:
            try:
                rank = int(raw_rank)
            except Exception:
                continue
            label_idx = rank_to_label.get(rank)
            if label_idx is None:
                continue
            comp = labels_cc == label_idx
            ys, xs = np.where(comp)
            if ys.size == 0 or xs.size == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            out_rgb[y1:y2, x1:x2] = out_rgb[y1:y2, x1:x2][:, ::-1, :]
            out_labels[y1:y2, x1:x2] = out_labels[y1:y2, x1:x2][:, ::-1]
            labels_cc[y1:y2, x1:x2] = labels_cc[y1:y2, x1:x2][:, ::-1]

    return out_rgb, out_labels


def crop_to_union(rgb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    union = labels > 0
    ys, xs = np.where(union)
    if ys.size == 0 or xs.size == 0:
        return rgb.copy(), labels.copy()
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    rgb_crop = rgb[y1:y2, x1:x2].copy()
    labels_crop = labels[y1:y2, x1:x2].copy()
    union_crop = labels_crop > 0
    rgb_crop[~union_crop] = 255
    return rgb_crop, labels_crop


def process_side(
    section_root: Path,
    mask_path: Path,
    whole_flip: bool,
    component_groups: dict[str, list[int]] | None,
    group_flip_lr: dict[str, bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(cv2.imread(str(section_root / "crop_raw.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    labels = load_mask_labels(mask_path)
    rgb, labels = apply_whole_flip(rgb, labels, whole_flip)
    rgb, labels = apply_group_flips(rgb, labels, component_groups, group_flip_lr)
    return crop_to_union(rgb, labels)


def export_one(payload: dict[str, Any]) -> dict[str, Any]:
    pair_key = str(payload["pair_key"])
    myelin_label = str(payload["myelin_label"])
    nissl_label = str(payload["nissl_label"])
    myelin_rgb, myelin_labels = process_side(
        Path(payload["myelin_root"]),
        Path(payload["myelin_mask_path"]),
        bool(payload["flip_myelin_lr"]),
        dict(payload["myelin_component_groups"]),
        dict(payload["myelin_group_flip_lr"]),
    )
    nissl_rgb, nissl_labels = process_side(
        Path(payload["nissl_root"]),
        Path(payload["nissl_mask_path"]),
        bool(payload["flip_nissl_lr"]),
        dict(payload["nissl_component_groups"]),
        dict(payload["nissl_group_flip_lr"]),
    )

    out_root = Path(payload["out_root"])
    stem = str(payload["stem"])
    myelin_img_path = out_root / "trainA" / f"{stem}_myelin.png"
    nissl_img_path = out_root / "trainB" / f"{stem}_nissl.png"
    myelin_mask_out = out_root / "masksA" / f"{stem}_myelin_mask_labels.png"
    nissl_mask_out = out_root / "masksB" / f"{stem}_nissl_mask_labels.png"

    cv2.imwrite(str(myelin_img_path), cv2.cvtColor(myelin_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(nissl_img_path), cv2.cvtColor(nissl_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(myelin_mask_out), myelin_labels)
    cv2.imwrite(str(nissl_mask_out), nissl_labels)

    return {
        "pair_key": pair_key,
        "myelin_label": myelin_label,
        "nissl_label": nissl_label,
        "myelin_image": str(myelin_img_path.relative_to(out_root)),
        "nissl_image": str(nissl_img_path.relative_to(out_root)),
        "myelin_mask": str(myelin_mask_out.relative_to(out_root)),
        "nissl_mask": str(nissl_mask_out.relative_to(out_root)),
        "flip_myelin_lr": bool(payload["flip_myelin_lr"]),
        "flip_nissl_lr": bool(payload["flip_nissl_lr"]),
        "myelin_group_flip_lr": json.dumps(payload["myelin_group_flip_lr"], ensure_ascii=True),
        "nissl_group_flip_lr": json.dumps(payload["nissl_group_flip_lr"], ensure_ascii=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default=str(Path(r"/mnt/d/Research/Image Analysis/cyclegan_usable_pairs_20260324")),
    )
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    train_a = out_root / "trainA"
    train_b = out_root / "trainB"
    masks_a = out_root / "masksA"
    masks_b = out_root / "masksB"
    for d in (train_a, train_b, masks_a, masks_b):
        d.mkdir(parents=True, exist_ok=True)

    registry = load_registry(PAIR_REGISTRY)
    usable_items = sorted(
        (k, v) for k, v in registry.items() if isinstance(v, dict) and v.get("registration_status") == "usable"
    )

    tasks: list[dict[str, Any]] = []
    for index, (pair_key, record) in enumerate(usable_items, start=1):
        myelin_label = str(record["myelin_label"])
        nissl_label = str(record["nissl_label"])
        reg_files = dict(record.get("registration_mask_files") or {})
        myelin_mask_path = registry_relpath_to_abs(reg_files["myelin"])
        nissl_mask_path = registry_relpath_to_abs(reg_files["nissl"])
        myelin_root = MYELIN_ROOT / myelin_label
        nissl_root = NISSL_ROOT / nissl_label
        stem = f"{index:03d}_{myelin_label}__{nissl_label}"
        tasks.append(
            {
                "pair_key": pair_key,
                "myelin_label": myelin_label,
                "nissl_label": nissl_label,
                "myelin_root": str(myelin_root),
                "nissl_root": str(nissl_root),
                "myelin_mask_path": str(myelin_mask_path),
                "nissl_mask_path": str(nissl_mask_path),
                "flip_myelin_lr": bool((record.get("flip_lr") or {}).get("myelin", False)),
                "flip_nissl_lr": bool((record.get("flip_lr") or {}).get("nissl", False)),
                "myelin_component_groups": dict((record.get("component_groups") or {}).get("myelin") or {}),
                "nissl_component_groups": dict((record.get("component_groups") or {}).get("nissl") or {}),
                "myelin_group_flip_lr": dict((record.get("group_flip_lr") or {}).get("myelin") or {}),
                "nissl_group_flip_lr": dict((record.get("group_flip_lr") or {}).get("nissl") or {}),
                "out_root": str(out_root),
                "stem": stem,
            }
        )

    manifest_rows: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for row in ex.map(export_one, tasks):
            manifest_rows.append(row)

    manifest_rows.sort(key=lambda r: r["pair_key"])

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary = {
        "usable_pairs_exported": len(manifest_rows),
        "trainA_count": len(list(train_a.glob("*.png"))),
        "trainB_count": len(list(train_b.glob("*.png"))),
        "masksA_count": len(list(masks_a.glob("*.png"))),
        "masksB_count": len(list(masks_b.glob("*.png"))),
        "out_root": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
