#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.tools.evaluate_myelin_support_ws0 import (
    dilate_mask,
    feature_stack,
    load_ws0_items,
    overlay_mask,
    predict_support,
    pseudo_support_labels,
    run_m3_from_support,
)
from histology.tools.search_myelin_mask_strategies import (
    local_recall,
    method_factory,
    region_metrics,
    region_slice_masks,
    leakage_metrics,
)


DEFAULT_SECTIONS = ["2504_5", "2503_144", "2507_66"]


def sample_training_data_light(
    items,
    *,
    rng: np.random.Generator,
    max_pos_per_item: int,
    max_neg_per_item: int,
):
    xs = []
    ys = []
    for item in items:
        feats = feature_stack(item.crop_rgb)
        pos, neg = pseudo_support_labels(item.gt_mask, item.artifact_mask)
        for mask, label, limit in ((pos, 1, max_pos_per_item), (neg, 0, max_neg_per_item)):
            yy, xx = np.where(mask)
            if yy.size == 0:
                continue
            take = min(limit, yy.size)
            idx = rng.choice(yy.size, size=take, replace=False)
            xs.append(feats[yy[idx], xx[idx], :])
            ys.append(np.full(take, label, dtype=np.uint8))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def reduced_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    rm = region_metrics(pred, gt)
    leak = leakage_metrics(pred, gt)
    regions = region_slice_masks(gt)
    return {
        **rm,
        **leak,
        "left_recall": local_recall(pred, regions["left"]),
        "right_recall": local_recall(pred, regions["right"]),
        "top_recall": local_recall(pred, regions["top"]),
        "boundary_recall": local_recall(pred, regions["boundary"]),
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    return {
        "count": len(rows),
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "mean_fp_over_gt_area": float(np.mean([r["fp_over_gt_area"] for r in rows])),
        "mean_pred_to_gt_area_ratio": float(np.mean([r["pred_to_gt_area_ratio"] for r in rows])),
        "mean_left_recall": float(np.mean([r["left_recall"] for r in rows])),
        "mean_right_recall": float(np.mean([r["right_recall"] for r in rows])),
        "mean_top_recall": float(np.mean([r["top_recall"] for r in rows])),
        "mean_boundary_recall": float(np.mean([r["boundary_recall"] for r in rows])),
        "probe_score": float(
            0.45 * np.mean([r["dice"] for r in rows])
            + 0.15 * np.mean([r["precision"] for r in rows])
            + 0.15 * np.mean([r["recall"] for r in rows])
            + 0.10 * np.mean([r["boundary_recall"] for r in rows])
            + 0.05 * np.mean([r["left_recall"] for r in rows])
            + 0.05 * np.mean([r["right_recall"] for r in rows])
            + 0.05 * np.mean([r["top_recall"] for r in rows])
            - 0.10 * np.mean([r["fp_over_gt_area"] for r in rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--max-pos-per-item", type=int, default=400)
    parser.add_argument("--max-neg-per-item", type=int, default=400)
    parser.add_argument("--save-qc", action="store_true")
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    qc_dir = outdir / "per_section"
    if args.save_qc:
        qc_dir.mkdir(exist_ok=True)

    selected = args.section or DEFAULT_SECTIONS
    items = load_ws0_items(gt_root, scale=float(args.scale))
    methods = method_factory()
    rows = []

    for label in selected:
        matched = [it for it in items if it.label == label]
        if not matched:
            continue
        item = matched[0]
        train_items = [it for it in items if it.slide_name != item.slide_name]
        rng = np.random.default_rng(sum(map(ord, label)))
        x_train, y_train = sample_training_data_light(
            train_items,
            rng=rng,
            max_pos_per_item=int(args.max_pos_per_item),
            max_neg_per_item=int(args.max_neg_per_item),
        )
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            class_weight="balanced",
            random_state=0,
        )
        clf.fit(x_train, y_train)

        crop = item.crop_rgb
        gt = item.gt_mask
        prob, support_rf = predict_support(clf, crop, thresh=0.50)
        base_support = methods["m2_candidate_union_v2"](crop)
        method_preds = {"m3_hyst_entres_guard_v1": methods["m3_hyst_entres_guard_v1"](crop)}
        if args.all_variants:
            method_preds["ws0_lr_replace_v1"] = run_m3_from_support(crop, support_rf)
            method_preds["ws0_lr_or_v1"] = run_m3_from_support(crop, support_rf | base_support)
        method_preds["ws0_lr_gate_v1"] = run_m3_from_support(crop, (prob >= 0.35) & dilate_mask(base_support, 0.02))

        if args.save_qc:
            sec_dir = qc_dir / label
            sec_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(sec_dir / "crop_rgb.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sec_dir / "gt_overlay.png"), cv2.cvtColor(overlay_mask(crop, gt, (0, 255, 0)), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sec_dir / "ws0_support_probability.png"), np.clip(prob * 255.0, 0, 255).astype(np.uint8))
            for method_name, pred in method_preds.items():
                overlay = overlay_mask(crop, pred, (255, 0, 0))
                cv2.imwrite(str(sec_dir / f"{method_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        for method_name, pred in method_preds.items():
            rows.append({
                "section": label,
                "slide_name": item.slide_name,
                "method": method_name,
                **reduced_metrics(pred, gt),
            })

    csv_path = outdir / "per_section_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for method_name in sorted({r["method"] for r in rows}):
        grouped[method_name] = summarize([r for r in rows if r["method"] == method_name])
    (outdir / "aggregate_metrics.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")

    ranking = sorted(grouped.items(), key=lambda kv: kv[1]["probe_score"], reverse=True)
    lines = [
        "# WS0 Light Probe",
        "",
        f"Scale: {float(args.scale):.2f}",
        f"Sections: {', '.join(selected)}",
        f"Training samples per item: pos={int(args.max_pos_per_item)} neg={int(args.max_neg_per_item)}",
        "",
        "## Ranking",
        "",
    ]
    for idx, (name, stats) in enumerate(ranking, start=1):
        lines.extend([
            f"{idx}. `{name}`",
            f"   - probe_score: `{stats['probe_score']:.4f}`",
            f"   - Dice: `{stats['mean_dice']:.4f}`",
            f"   - Precision: `{stats['mean_precision']:.4f}`",
            f"   - Recall: `{stats['mean_recall']:.4f}`",
            f"   - FP/GT: `{stats['mean_fp_over_gt_area']:.4f}`",
            f"   - left/right/top recall: `{stats['mean_left_recall']:.4f}` / `{stats['mean_right_recall']:.4f}` / `{stats['mean_top_recall']:.4f}`",
            "",
        ])
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(grouped, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
