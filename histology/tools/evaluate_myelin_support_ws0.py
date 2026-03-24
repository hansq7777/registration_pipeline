#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.tools.search_myelin_mask_strategies import (
    boundary_mask,
    collect_gt_crops,
    contour_metrics,
    entropy_score,
    fallback_if_too_small,
    finite_mean,
    hysteresis_support_reconstruct,
    leakage_metrics,
    local_recall,
    method_factory,
    region_metrics,
    region_slice_masks,
    residual_score,
)


@dataclass
class Ws0Item:
    label: str
    slide_name: str
    crop_rgb: np.ndarray
    gt_mask: np.ndarray
    artifact_mask: np.ndarray
    metadata: dict


def load_ws0_items(gt_root: Path, *, scale: float) -> list[Ws0Item]:
    items = []
    for item in collect_gt_crops(gt_root, scale=scale):
        sec_dir = gt_root / item.label
        artifact_path = sec_dir / "artifact_mask_final.png"
        if artifact_path.exists():
            artifact = np.asarray(Image.open(artifact_path).convert("L")) > 0
            if scale < 0.999:
                h, w = item.gt_mask.shape[:2]
                artifact = cv2.resize(
                    artifact.astype(np.uint8) * 255,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        else:
            artifact = np.zeros_like(item.gt_mask, dtype=bool)
        items.append(
            Ws0Item(
                label=item.label,
                slide_name=str(item.metadata["source_slide"]["name"]),
                crop_rgb=item.crop_rgb,
                gt_mask=item.gt_mask.astype(bool),
                artifact_mask=artifact.astype(bool),
                metadata=item.metadata,
            )
        )
    return items


def pseudo_support_labels(
    gt_mask: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    inner_frac: float = 0.015,
    outer_frac: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = gt_mask.shape[:2]
    r_in = max(1, int(round(min(h, w) * inner_frac)))
    r_out = max(1, int(round(min(h, w) * outer_frac)))
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1, 2 * r_in + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
    pos = cv2.erode(gt_mask.astype(np.uint8) * 255, k_in, iterations=1) > 0
    if not pos.any():
        pos = gt_mask.copy()
    dil = cv2.dilate(gt_mask.astype(np.uint8) * 255, k_out, iterations=1) > 0
    neg = (~dil) | artifact_mask.astype(bool)
    return pos.astype(bool), neg.astype(bool)


def feature_stack(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    resid = residual_score(crop_rgb)
    ent = entropy_score(crop_rgb, radius=5)
    nonwhite = (255 - crop_rgb.min(axis=2)).astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grayf = gray.astype(np.float32)
    mean = cv2.GaussianBlur(grayf, (0, 0), sigmaX=5, sigmaY=5)
    sqmean = cv2.GaussianBlur(grayf * grayf, (0, 0), sigmaX=5, sigmaY=5)
    local_std = np.sqrt(np.clip(sqmean - mean * mean, 0, None))
    local_std = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = (xx / max(1, w - 1)).astype(np.float32)
    y_norm = (yy / max(1, h - 1)).astype(np.float32)
    dist_border = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    dist_border = dist_border / max(1.0, min(h, w) / 2.0)

    feats = np.stack(
        [
            gray.astype(np.float32) / 255.0,
            resid.astype(np.float32) / 255.0,
            ent.astype(np.float32) / 255.0,
            nonwhite.astype(np.float32) / 255.0,
            grad.astype(np.float32) / 255.0,
            local_std.astype(np.float32) / 255.0,
            x_norm,
            y_norm,
            dist_border.astype(np.float32),
        ],
        axis=-1,
    )
    return feats


def sample_training_data(
    items: list[Ws0Item],
    *,
    rng: np.random.Generator,
    max_pos_per_item: int = 4000,
    max_neg_per_item: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
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
    if not xs:
        raise RuntimeError("No training samples collected for WS0.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def predict_support(
    model: HistGradientBoostingClassifier,
    crop_rgb: np.ndarray,
    *,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    feats = feature_stack(crop_rgb)
    flat = feats.reshape(-1, feats.shape[-1])
    prob = model.predict_proba(flat)[:, 1].reshape(crop_rgb.shape[:2])
    return prob.astype(np.float32), (prob >= thresh)


def dilate_mask(mask: np.ndarray, frac: float) -> np.ndarray:
    h, w = mask.shape[:2]
    r = max(1, int(round(min(h, w) * frac)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8) * 255, k, iterations=1) > 0


def run_m3_from_support(crop_rgb: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    methods = method_factory()
    structural_core = methods["crop_center_default2comp"](crop_rgb)
    fallback = methods["m2_hybrid_entres_tight_v1"](crop_rgb)
    recon = hysteresis_support_reconstruct(
        support_mask,
        residual_score(crop_rgb),
        structural_core,
        core_quantile=0.84,
        core_scale=1.00,
        overlap_frac=0.03,
        core_open_k=3,
        final_close_k=7,
    )
    return fallback_if_too_small(recon, fallback, min_frac_of_fallback=0.70).astype(bool)


def overlay_mask(crop_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = crop_rgb.copy()
    color_arr = np.asarray(color, dtype=np.uint8)
    m = mask.astype(bool)
    out[m] = np.clip(0.55 * out[m] + 0.45 * color_arr, 0, 255).astype(np.uint8)
    b = boundary_mask(m)
    out[b] = color_arr
    return out


def evaluate_prediction(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    rm = region_metrics(pred, gt)
    cm = contour_metrics(pred, gt)
    leak = leakage_metrics(pred, gt)
    local_regions = region_slice_masks(gt)
    return {
        **rm,
        **cm,
        **leak,
        "top_recall": local_recall(pred, local_regions["top"]),
        "middle_recall": local_recall(pred, local_regions["middle"]),
        "bottom_recall": local_recall(pred, local_regions["bottom"]),
        "left_recall": local_recall(pred, local_regions["left"]),
        "center_recall": local_recall(pred, local_regions["center"]),
        "right_recall": local_recall(pred, local_regions["right"]),
        "boundary_recall": local_recall(pred, local_regions["boundary"]),
        "core_recall": local_recall(pred, local_regions["core"]),
    }


def summarize(method_rows: list[dict]) -> dict[str, float]:
    mean_dice = float(np.mean([r["dice"] for r in method_rows]))
    mean_iou = float(np.mean([r["iou"] for r in method_rows]))
    mean_precision = float(np.mean([r["precision"] for r in method_rows]))
    mean_boundary_f1 = float(np.mean([r["boundary_f1_tol64"] for r in method_rows]))
    mean_leak = float(np.mean([r["fp_over_gt_area"] for r in method_rows]))
    mean_area_ratio = float(np.mean([r["pred_to_gt_area_ratio"] for r in method_rows]))
    hd95_vals = [r["hd95_px"] for r in method_rows]
    hd95_f = finite_mean(hd95_vals)
    hd95_term = 1.0 / (1.0 + ((hd95_f or 1e9) / 1000.0))
    composite = (
        0.35 * mean_dice
        + 0.15 * mean_iou
        + 0.15 * mean_precision
        + 0.20 * mean_boundary_f1
        + 0.10 * hd95_term
        - 0.10 * abs(mean_area_ratio - 1.0)
        - 0.10 * mean_leak
    )
    return {
        "count": len(method_rows),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_precision": mean_precision,
        "mean_recall": float(np.mean([r["recall"] for r in method_rows])),
        "mean_boundary_f1_tol32": float(np.mean([r["boundary_f1_tol32"] for r in method_rows])),
        "mean_boundary_f1_tol64": mean_boundary_f1,
        "mean_assd_px_finite": finite_mean([r["assd_px"] for r in method_rows]),
        "mean_hd95_px_finite": hd95_f,
        "mean_fp_over_gt_area": mean_leak,
        "mean_border_fp_over_gt_area": float(np.mean([r["border_fp_over_gt_area"] for r in method_rows])),
        "mean_pred_to_gt_area_ratio": mean_area_ratio,
        "mean_boundary_recall": float(np.mean([r["boundary_recall"] for r in method_rows])),
        "mean_core_recall": float(np.mean([r["core_recall"] for r in method_rows])),
        "composite_score": composite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--save-per-section-images", action="store_true")
    parser.add_argument("--max-pos-per-item", type=int, default=1200)
    parser.add_argument("--max-neg-per-item", type=int, default=1200)
    parser.add_argument("--gb-max-iter", type=int, default=120)
    parser.add_argument("--gb-max-depth", type=int, default=8)
    parser.add_argument("--test-slide", action="append", default=[])
    parser.add_argument("--test-section", action="append", default=[])
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_section_dir = output_dir / "per_section"
    if args.save_per_section_images:
        per_section_dir.mkdir(exist_ok=True)

    items = load_ws0_items(gt_root, scale=float(args.scale))
    slide_names = sorted({item.slide_name for item in items})
    if args.test_slide:
        selected = set(args.test_slide)
        slide_names = [name for name in slide_names if name in selected]
        if not slide_names:
            raise RuntimeError("No test slides matched --test-slide.")
    test_section_filter = set(args.test_section)
    methods = method_factory()
    rows: list[dict] = []
    grouped: dict[str, list[dict]] = {
        "m3_hyst_entres_guard_v1": [],
        "ws0_rf_replace_v1": [],
        "ws0_rf_or_v1": [],
        "ws0_rf_gate_v1": [],
    }

    for fold_idx, slide_name in enumerate(slide_names, start=1):
        train_items = [it for it in items if it.slide_name != slide_name]
        test_items = [it for it in items if it.slide_name == slide_name]
        if test_section_filter:
            test_items = [it for it in test_items if it.label in test_section_filter]
            if not test_items:
                continue
        print(f"[fold {fold_idx}/{len(slide_names)}] train={len(train_items)} test={slide_name}:{len(test_items)}", flush=True)

        rng = np.random.default_rng(1000 + fold_idx)
        x_train, y_train = sample_training_data(
            train_items,
            rng=rng,
            max_pos_per_item=int(args.max_pos_per_item),
            max_neg_per_item=int(args.max_neg_per_item),
        )
        model = HistGradientBoostingClassifier(
            max_iter=int(args.gb_max_iter),
            max_depth=int(args.gb_max_depth),
            min_samples_leaf=20,
            learning_rate=0.08,
            validation_fraction=None,
            random_state=fold_idx,
        )
        model.fit(x_train, y_train)

        for item in test_items:
            crop = item.crop_rgb
            gt = item.gt_mask
            prob, support_rf = predict_support(model, crop, thresh=0.50)
            base_support = methods["m2_candidate_union_v2"](crop)
            pred_baseline = methods["m3_hyst_entres_guard_v1"](crop)
            pred_replace = run_m3_from_support(crop, support_rf)
            pred_or = run_m3_from_support(crop, support_rf | base_support)
            pred_gate = run_m3_from_support(crop, (prob >= 0.35) & dilate_mask(base_support, 0.02))

            method_preds = {
                "m3_hyst_entres_guard_v1": pred_baseline,
                "ws0_rf_replace_v1": pred_replace,
                "ws0_rf_or_v1": pred_or,
                "ws0_rf_gate_v1": pred_gate,
            }

            if args.save_per_section_images:
                sec_dir = per_section_dir / item.label
                sec_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(sec_dir / "crop_rgb.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sec_dir / "gt_overlay.png"), cv2.cvtColor(overlay_mask(crop, gt, (0, 255, 0)), cv2.COLOR_RGB2BGR))
                prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(sec_dir / "ws0_support_probability.png"), prob_u8)
                cv2.imwrite(str(sec_dir / "ws0_support_binary.png"), (support_rf.astype(np.uint8) * 255))
                ordered = list(method_preds.keys())
                tile_h, tile_w = crop.shape[:2]
                header_h = 36
                panel = np.zeros((header_h + tile_h, tile_w * (2 + len(ordered)), 3), dtype=np.uint8)
                panel[:header_h, :, :] = 18
                panel[header_h:, :tile_w, :] = crop
                panel[header_h:, tile_w : 2 * tile_w, :] = overlay_mask(crop, gt, (0, 255, 0))
                cv2.putText(panel, "crop", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, "gt", (tile_w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                for idx_m, method_name in enumerate(ordered, start=2):
                    x0 = idx_m * tile_w
                    overlay = overlay_mask(crop, method_preds[method_name], (255, 0, 0))
                    panel[header_h:, x0 : x0 + tile_w, :] = overlay
                    cv2.putText(panel, method_name[:24], (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(sec_dir / f"{method_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sec_dir / f"{item.label}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            for method_name, pred in method_preds.items():
                metrics = evaluate_prediction(pred, gt)
                row = {
                    "section": item.label,
                    "slide_name": item.slide_name,
                    "method": method_name,
                    "fold_slide": slide_name,
                    **metrics,
                }
                rows.append(row)
                grouped[method_name].append(row)

    csv_path = output_dir / "per_section_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {name: summarize(method_rows) for name, method_rows in grouped.items()}
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    ranked = sorted(
        aggregate.items(),
        key=lambda kv: (kv[1]["composite_score"], kv[1]["mean_dice"], kv[1]["mean_boundary_f1_tol64"], -kv[1]["mean_fp_over_gt_area"]),
        reverse=True,
    )
    lines = [
        "# Myelin WS0 Weak-Supervision Support Baseline",
        "",
        f"GT sections evaluated: {len(items)}",
        f"Slide-level folds: {len(slide_names)}",
        f"Working scale: {float(args.scale):.3f}",
        f"Training samples per item: pos={int(args.max_pos_per_item)} neg={int(args.max_neg_per_item)}",
        f"HistGradientBoosting: max_iter={int(args.gb_max_iter)} max_depth={int(args.gb_max_depth)}",
        "",
        "## Ranking",
        "",
    ]
    for idx, (name, stats) in enumerate(ranked, start=1):
        lines.extend(
            [
                f"{idx}. `{name}`",
                f"   - composite: `{stats['composite_score']:.4f}`",
                f"   - Dice: `{stats['mean_dice']:.4f}`",
                f"   - BF64: `{stats['mean_boundary_f1_tol64']:.4f}`",
                f"   - HD95: `{stats['mean_hd95_px_finite']:.1f}`",
                f"   - FP/GT: `{stats['mean_fp_over_gt_area']:.4f}`",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
