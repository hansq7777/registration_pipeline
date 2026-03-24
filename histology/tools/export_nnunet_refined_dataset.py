from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def natural_key(name: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)_(\d+)$", name)
    if not match:
        return (10**9, 10**9, name)
    return (int(match.group(1)), int(match.group(2)), name)


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(long_edge) / float(max(height, width)))
    if scale >= 1.0:
        return image
    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def export_one(
    section_id: str,
    case_id: str,
    section_dir_str: str,
    out_root_str: str,
    long_edge: int,
    work_name: str,
) -> dict[str, object]:
    section_dir = Path(section_dir_str)
    out_root = Path(out_root_str)
    images_dir = out_root / "imagesTr"
    labels_dir = out_root / "labelsTr"
    metadata_dir = out_root / "metadata"

    raw_path = section_dir / "crop_raw.png"
    work_path = section_dir / work_name
    mask_path = section_dir / "tissue_mask_final.png"
    fg_path = section_dir / "foreground_rgba.png"
    meta_path = section_dir / "metadata.json"

    if not (raw_path.exists() and mask_path.exists() and fg_path.exists() and meta_path.exists()):
        return {"section_id": section_id, "status": "skip_missing"}

    image_out = images_dir / f"{case_id}_0000.png"
    label_out = labels_dir / f"{case_id}.png"
    meta_out = metadata_dir / f"{case_id}.json"

    if work_path.exists():
        image = cv2.imread(str(work_path), cv2.IMREAD_COLOR)
        work_source = f"existing_{work_name}"
    else:
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            return {"section_id": section_id, "status": "skip_bad_raw"}
        image = resize_long_edge(raw, long_edge)
        work_source = f"generated_from_crop_raw_longedge_{long_edge}"

    if image is None:
        return {"section_id": section_id, "status": "skip_bad_image"}

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"section_id": section_id, "status": "skip_bad_mask"}

    height, width = image.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    label = (mask > 0).astype(np.uint8)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_out), image):
        return {"section_id": section_id, "status": "skip_write_image"}
    if not cv2.imwrite(str(label_out), label):
        return {"section_id": section_id, "status": "skip_write_label"}
    shutil.copy2(meta_path, meta_out)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_id = metadata.get("sample_id")
    section_num = metadata.get("section_id")

    return {
        "section_id": section_id,
        "case_id": case_id,
        "status": "ok",
        "sample_id": sample_id,
        "section_num": section_num,
        "work_source": work_source,
        "image_width_px": int(width),
        "image_height_px": int(height),
        "stain": metadata.get("stain"),
        "pipeline_stage": metadata.get("pipeline_stage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--upto", required=True)
    parser.add_argument("--work-name", default="crop_work_standard.png")
    parser.add_argument("--long-edge", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()

    items = sorted(
        [p for p in args.src_root.iterdir() if p.is_dir() and p.name != "test"],
        key=lambda p: natural_key(p.name),
    )
    names = [p.name for p in items]
    if args.upto.upper() == "ALL":
        subset = items
    else:
        if args.upto not in names:
            raise SystemExit(f"upto section not found: {args.upto}")
        subset = items[: names.index(args.upto) + 1]

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for section_dir in subset:
            futures.append(
                pool.submit(
                    export_one,
                    section_dir.name,
                    f"{args.case_prefix}{section_dir.name}",
                    str(section_dir),
                    str(out_root),
                    args.long_edge,
                    args.work_name,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            if result.get("status") == "ok":
                sid = str(result["section_id"])
                case_id = str(result["case_id"])
                rows.append(
                    {
                        "case_id": case_id,
                        "section_id": sid,
                        "sample_id": result.get("sample_id"),
                        "section_num": result.get("section_num"),
                        "source_section_dir": str(args.src_root / sid),
                        "image_file": str(out_root / "imagesTr" / f"{case_id}_0000.png"),
                        "label_file": str(out_root / "labelsTr" / f"{case_id}.png"),
                        "metadata_file": str(out_root / "metadata" / f"{case_id}.json"),
                        "refined_indicator": "foreground_rgba_present",
                        "work_source": result.get("work_source"),
                        "image_width_px": result.get("image_width_px"),
                        "image_height_px": result.get("image_height_px"),
                        "stain": result.get("stain"),
                        "pipeline_stage": result.get("pipeline_stage"),
                        "label_values": "0,1",
                    }
                )
            else:
                skipped.append(result)

    manifest_path = out_root / "manifest.csv"
    existing_rows: list[dict[str, object]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))

    merged: dict[str, dict[str, object]] = {}
    for row in existing_rows:
        existing_case_id = str(row.get("case_id") or row.get("section_id"))
        row["case_id"] = existing_case_id
        merged[existing_case_id] = row
    for row in rows:
        merged[str(row["case_id"])] = row
    merged_rows = list(merged.values())
    merged_rows.sort(key=lambda row: str(row["case_id"]))

    if merged_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
            writer.writeheader()
            writer.writerows(merged_rows)

    dataset_json = {
        "channel_names": {"0": "RGB"},
        "labels": {"background": 0, "tissue": 1},
        "numTraining": len(merged_rows),
        "file_ending": ".png",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    summary = {
        "dataset_name": out_root.name,
        "selection_rule": f"natural-order folders up to and including {args.upto} AND foreground_rgba present",
        "selected_section_dirs": len(subset),
        "created_pairs_total": len(rows),
        "manifest_pairs_total": len(merged_rows),
        "workers": args.workers,
        "long_edge": args.long_edge,
        "work_name": args.work_name,
        "case_prefix": args.case_prefix,
        "label_encoding": "binary uint8 with values {0,1}",
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
    }
    (out_root / args.summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "README.txt").write_text(
        "\n".join(
            [
                "nnU-Net refined-only training export prepared from Tissue&Masks.",
                "imagesTr contains RGB PNG inputs named <section_id>_0000.png.",
                "labelsTr contains binary mask PNG labels named <section_id>.png with values 0 and 1.",
                "Only sections with foreground_rgba.png were included.",
                "metadata contains copies of per-section metadata.json for traceability.",
                "manifest.csv records source folder and image provenance.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
