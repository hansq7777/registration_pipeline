#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.tools.evaluate_myelin_mask_after_bbox_update import contour_metrics, region_metrics


def resize_mask(mask: np.ndarray, scale: float, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    small = cv2.resize(mask.astype(np.uint8) * 255, (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
    restored = cv2.resize(small.astype(np.uint8) * 255, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return restored


def _kernel_size(mask: np.ndarray) -> int:
    min_dim = min(mask.shape[:2])
    k = max(3, int(round(min_dim * 0.002)))
    return k if (k % 2 == 1) else (k + 1)


def morph(mask: np.ndarray, op: str) -> np.ndarray:
    kernel = np.ones((_kernel_size(mask), _kernel_size(mask)), np.uint8)
    if op == "shrink":
        out = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    elif op == "expand":
        out = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    else:
        raise ValueError(op)
    return out


def collect_masks(root: Path, limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    items = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        mask_path = path / "tissue_mask_final.png"
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        items.append((path.name, mask))
        if limit is not None and len(items) >= limit:
            break
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", nargs="*", type=float, default=[1.0, 0.75, 0.5, 0.33, 0.25])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    mask_root = Path(args.mask_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = collect_masks(mask_root, args.limit)
    rows: list[dict] = []

    for label, mask in masks:
        full_shrink = morph(mask, "shrink")
        full_expand = morph(mask, "expand")
        for scale in args.scales:
            t0 = time.perf_counter()
            restored = resize_mask(mask, scale, mask.shape[:2])
            fidelity_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            shrink_restored = morph(resize_mask(mask, scale, mask.shape[:2]), "shrink")
            shrink_sec = time.perf_counter() - t1

            t2 = time.perf_counter()
            expand_restored = morph(resize_mask(mask, scale, mask.shape[:2]), "expand")
            expand_sec = time.perf_counter() - t2

            reg = region_metrics(restored, mask)
            contour = contour_metrics(restored, mask)
            shrink_reg = region_metrics(shrink_restored, full_shrink)
            shrink_contour = contour_metrics(shrink_restored, full_shrink)
            expand_reg = region_metrics(expand_restored, full_expand)
            expand_contour = contour_metrics(expand_restored, full_expand)

            rows.append(
                {
                    "label": label,
                    "scale": scale,
                    "mask_dice": reg["dice"],
                    "mask_iou": reg["iou"],
                    "mask_boundary_f1_tol32": contour["boundary_f1_tol32"],
                    "mask_boundary_f1_tol64": contour["boundary_f1_tol64"],
                    "shrink_dice_vs_full": shrink_reg["dice"],
                    "shrink_boundary_f1_tol32": shrink_contour["boundary_f1_tol32"],
                    "expand_dice_vs_full": expand_reg["dice"],
                    "expand_boundary_f1_tol32": expand_contour["boundary_f1_tol32"],
                    "fidelity_sec": fidelity_sec,
                    "shrink_sec": shrink_sec,
                    "expand_sec": expand_sec,
                }
            )

    csv_path = output_dir / "resolution_fidelity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    aggregate: dict[str, dict] = {}
    for scale in args.scales:
        scale_rows = [r for r in rows if abs(float(r["scale"]) - float(scale)) < 1e-9]
        if not scale_rows:
            continue
        aggregate[str(scale)] = {
            "count": len(scale_rows),
            "mean_mask_dice": float(np.mean([r["mask_dice"] for r in scale_rows])),
            "mean_mask_iou": float(np.mean([r["mask_iou"] for r in scale_rows])),
            "mean_mask_boundary_f1_tol32": float(np.mean([r["mask_boundary_f1_tol32"] for r in scale_rows])),
            "mean_mask_boundary_f1_tol64": float(np.mean([r["mask_boundary_f1_tol64"] for r in scale_rows])),
            "mean_shrink_dice_vs_full": float(np.mean([r["shrink_dice_vs_full"] for r in scale_rows])),
            "mean_expand_dice_vs_full": float(np.mean([r["expand_dice_vs_full"] for r in scale_rows])),
            "mean_fidelity_sec": float(np.mean([r["fidelity_sec"] for r in scale_rows])),
            "mean_shrink_sec": float(np.mean([r["shrink_sec"] for r in scale_rows])),
            "mean_expand_sec": float(np.mean([r["expand_sec"] for r in scale_rows])),
        }

    summary = {
        "mask_root": str(mask_root),
        "aggregate": aggregate,
        "csv_path": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
