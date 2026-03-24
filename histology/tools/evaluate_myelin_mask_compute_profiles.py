from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registration_pipeline.histology.gui_mvp.hitl_gui.pipeline_adapters import (
    MASK_COMPUTE_PROFILE_FAST,
    MASK_COMPUTE_PROFILE_FULL,
    MASK_COMPUTE_PROFILE_STANDARD,
    MASK_PRESET_M3_HYST_ENTRES_GUARD,
    compute_auto_masks_resampled,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


PROFILES = (
    MASK_COMPUTE_PROFILE_FULL,
    MASK_COMPUTE_PROFILE_STANDARD,
    MASK_COMPUTE_PROFILE_FAST,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    inter = int((pred_b & gt_b).sum())
    denom = int(pred_b.sum()) + int(gt_b.sum())
    return 1.0 if denom == 0 else (2.0 * inter / float(denom))


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = int((pred_b & gt_b).sum())
    pp = int(pred_b.sum())
    return 1.0 if pp == 0 else (tp / float(pp))


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = int((pred_b & gt_b).sum())
    ap = int(gt_b.sum())
    return 1.0 if ap == 0 else (tp / float(ap))


def fp_over_gt(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    fp = int((pred_b & ~gt_b).sum())
    denom = max(1, int(gt_b.sum()))
    return fp / float(denom)


def pred_over_gt(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred > 0).sum()) / float(max(1, int((gt > 0).sum())))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol_px: int = 16) -> float:
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_b, cv2.MORPH_GRADIENT, kernel) > 0
    gt_edge = cv2.morphologyEx(gt_b, cv2.MORPH_GRADIENT, kernel) > 0
    if not pred_edge.any() and not gt_edge.any():
        return 1.0
    if not pred_edge.any() or not gt_edge.any():
        return 0.0
    pred_dist = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 5)
    gt_dist = cv2.distanceTransform((~gt_edge).astype(np.uint8), cv2.DIST_L2, 5)
    pred_match = pred_edge & (gt_dist <= float(tol_px))
    gt_match = gt_edge & (pred_dist <= float(tol_px))
    bp = float(pred_match.sum()) / float(max(1, int(pred_edge.sum())))
    br = float(gt_match.sum()) / float(max(1, int(gt_edge.sum())))
    return 0.0 if (bp + br) <= 1e-9 else (2.0 * bp * br / (bp + br))


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    edge = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = 0.2 * out[edge] + 0.8 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_panel(crop_rgb: np.ndarray, gt: np.ndarray, results: dict[str, dict]) -> np.ndarray:
    tiles = []
    base = overlay(crop_rgb, gt, (0, 255, 0))
    tiles.append(base)
    for profile in PROFILES:
        pred = results[profile]["mask"]
        color = (255, 0, 0) if profile == MASK_COMPUTE_PROFILE_FULL else ((255, 215, 0) if profile == MASK_COMPUTE_PROFILE_STANDARD else (0, 255, 255))
        tile = overlay(crop_rgb, pred, color)
        tile = overlay(tile, gt, (0, 255, 0))
        tiles.append(tile)
    target_h = 360
    resized = []
    for tile in tiles:
        h, w = tile.shape[:2]
        scale = target_h / float(max(1, h))
        out_w = max(1, int(round(w * scale)))
        resized.append(cv2.resize(tile, (out_w, target_h), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(resized)


def collect_sections(root: Path) -> list[Path]:
    out = []
    for section_dir in sorted(root.iterdir()):
        if not section_dir.is_dir():
            continue
        if all((section_dir / name).exists() for name in ("crop_raw.png", "tissue_mask_final.png", "metadata.json")):
            try:
                meta = json.loads((section_dir / "metadata.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(meta.get("stain", "")).lower() == "gallyas":
                out.append(section_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--section", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel_dir = out / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    items = collect_sections(root)
    if args.section:
        wanted = set(args.section)
        sample = [p for p in items if p.name in wanted]
    else:
        random.seed(args.seed)
        sample = random.sample(items, min(args.sample_count, len(items)))

    rows: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = {f"{profile}:{metric}": [] for profile in PROFILES for metric in ("runtime_s", "dice", "precision", "recall", "fp_over_gt", "pred_over_gt", "bf16")}

    for section_dir in sample:
        crop_rgb = load_rgb(section_dir / "crop_raw.png")
        gt = load_mask(section_dir / "tissue_mask_final.png")
        section_results: dict[str, dict] = {}
        for profile in PROFILES:
            t0 = time.perf_counter()
            tissue, artifact, info = compute_auto_masks_resampled(
                crop_rgb,
                "gallyas",
                method=MASK_PRESET_M3_HYST_ENTRES_GUARD,
                compute_profile=profile,
            )
            runtime_s = time.perf_counter() - t0
            pred = ((tissue > 0) & ~(artifact > 0)).astype(np.uint8)
            metrics = {
                "runtime_s": runtime_s,
                "dice": dice(pred, gt),
                "precision": precision(pred, gt),
                "recall": recall(pred, gt),
                "fp_over_gt": fp_over_gt(pred, gt),
                "pred_over_gt": pred_over_gt(pred, gt),
                "bf16": boundary_f1(pred, gt, tol_px=16),
            }
            section_results[profile] = {
                "mask": pred,
                "compute_info": info,
                **metrics,
            }
            for name, value in metrics.items():
                aggregate[f"{profile}:{name}"].append(float(value))
                rows.append(
                    {
                        "section": section_dir.name,
                        "profile": profile,
                        **metrics,
                        "working_shape_hw": info["working_shape_hw"],
                        "working_scale": info["working_scale"],
                    }
                )

        panel = make_panel(crop_rgb, gt, section_results)
        Image.fromarray(panel).save(panel_dir / f"{section_dir.name}_panel.png")

    summary = {
        "sample_sections": [p.name for p in sample],
        "profiles": {},
    }
    for profile in PROFILES:
        summary["profiles"][profile] = {
            metric: float(np.mean(aggregate[f"{profile}:{metric}"])) if aggregate[f"{profile}:{metric}"] else None
            for metric in ("runtime_s", "dice", "precision", "recall", "fp_over_gt", "pred_over_gt", "bf16")
        }

    (out / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import csv

    with (out / "per_section_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "profile",
                "runtime_s",
                "dice",
                "precision",
                "recall",
                "fp_over_gt",
                "pred_over_gt",
                "bf16",
                "working_shape_hw",
                "working_scale",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Myelin Mask Compute Profile Probe",
        "",
        f"Root: `{root}`",
        f"Sample sections: {', '.join(p.name for p in sample)}",
        "",
        "| profile | mean runtime_s | mean dice | mean precision | mean recall | mean fp/gt | mean pred/gt | mean bf16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        item = summary["profiles"][profile]
        lines.append(
            f"| {profile} | {item['runtime_s']:.3f} | {item['dice']:.4f} | {item['precision']:.4f} | {item['recall']:.4f} | {item['fp_over_gt']:.4f} | {item['pred_over_gt']:.4f} | {item['bf16']:.4f} |"
        )
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
