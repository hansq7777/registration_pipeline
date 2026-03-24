#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.application.pair_workspace import parse_section_label
from histology.gui_mvp.hitl_gui.application.section_workspace import WorkspaceSection, list_workspace_sections


@dataclass(frozen=True)
class ThumbSection:
    ordinal: int
    label: str
    sec_num: int
    crop_path: Path


def _default_root(stain: str) -> Path:
    if stain == "myelin":
        preferred = Path("/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
        if preferred.exists():
            return preferred
        return Path(r"D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks")
    preferred = Path("/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")
    if preferred.exists():
        return preferred
    return Path(r"D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks")


def _collect_sections(root: Path, animal_id: int, stain: str) -> list[ThumbSection]:
    out: list[ThumbSection] = []
    sections = list_workspace_sections(root)
    filtered: list[tuple[int, WorkspaceSection]] = []
    for item in sections:
        parsed = parse_section_label(item.label)
        if parsed is None:
            continue
        animal, sec = parsed
        if animal != animal_id:
            continue
        filtered.append((sec, item))
    for idx, (sec, item) in enumerate(sorted(filtered), start=1):
        out.append(
            ThumbSection(
                ordinal=idx,
                label=item.label,
                sec_num=sec,
                crop_path=item.crop_path,
            )
        )
    return out


def _fit_with_padding(path: Path, thumb_w: int, thumb_h: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb_w, thumb_h), (18, 18, 18))
    off_x = (thumb_w - img.size[0]) // 2
    off_y = (thumb_h - img.size[1]) // 2
    canvas.paste(img, (off_x, off_y))
    return canvas


def _load_thumb_for_section(args: tuple[Path, int, int]) -> Image.Image:
    path, thumb_w, thumb_h = args
    return _fit_with_padding(path, thumb_w, thumb_h)


def _draw_side_grid(
    sections: list[ThumbSection],
    title: str,
    *,
    thumb_w: int,
    thumb_h: int,
    cols: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    caption_h = 42
    tile_w = thumb_w
    tile_h = thumb_h + caption_h
    rows = max(1, math.ceil(max(1, len(sections)) / cols))
    title_h = 40
    pad = 16
    panel_w = pad * 2 + cols * tile_w + (cols - 1) * pad
    panel_h = title_h + pad + rows * tile_h + max(0, rows - 1) * pad + pad
    panel = Image.new("RGB", (panel_w, panel_h), (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((pad, 10), title, fill=(240, 240, 240), font=font)

    thumbs: list[Image.Image] = []
    if sections:
        work = [(section.crop_path, thumb_w, thumb_h) for section in sections]
        max_workers = min(8, len(work))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            thumbs = list(ex.map(_load_thumb_for_section, work))

    for idx, section in enumerate(sections):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + pad)
        y = title_h + row * (tile_h + pad)
        thumb = thumbs[idx]
        panel.paste(thumb, (x, y))
        draw.rectangle((x, y, x + tile_w - 1, y + thumb_h - 1), outline=(110, 110, 110), width=1)
        draw.text(
            (x, y + thumb_h + 6),
            f"#{section.ordinal}  {section.label}",
            fill=(255, 220, 140),
            font=font,
        )
        draw.text(
            (x, y + thumb_h + 22),
            f"sec={section.sec_num}",
            fill=(210, 210, 210),
            font=font,
        )
    return panel


def export_pairing_thumbnail_qc(
    myelin_root: Path,
    nissl_root: Path,
    animal_id: int,
    output_dir: Path,
    *,
    items_per_page: int,
    cols: int,
    thumb_w: int,
    thumb_h: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default()

    myelin_sections = _collect_sections(myelin_root, animal_id, "myelin")
    nissl_sections = _collect_sections(nissl_root, animal_id, "nissl")

    page_count = max(
        1,
        math.ceil(max(len(myelin_sections), len(nissl_sections)) / max(1, items_per_page)),
    )

    manifest_rows: list[dict[str, object]] = []
    for side, sections in (("myelin", myelin_sections), ("nissl", nissl_sections)):
        for sec in sections:
            manifest_rows.append(
                {
                    "animal_id": animal_id,
                    "side": side,
                    "ordinal": sec.ordinal,
                    "label": sec.label,
                    "sec_num": sec.sec_num,
                    "crop_path": str(sec.crop_path),
                }
            )

    for page_idx in range(page_count):
        start = page_idx * items_per_page
        end = start + items_per_page
        my_panel = _draw_side_grid(
            myelin_sections[start:end],
            f"Myelin | animal {animal_id} | items {start + 1}-{min(end, len(myelin_sections))} / {len(myelin_sections)}",
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            cols=cols,
            font=font,
        )
        ni_panel = _draw_side_grid(
            nissl_sections[start:end],
            f"Nissl | animal {animal_id} | items {start + 1}-{min(end, len(nissl_sections))} / {len(nissl_sections)}",
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            cols=cols,
            font=font,
        )
        gap = 24
        canvas = Image.new(
            "RGB",
            (my_panel.size[0] + ni_panel.size[0] + gap, max(my_panel.size[1], ni_panel.size[1])),
            (12, 12, 12),
        )
        canvas.paste(my_panel, (0, 0))
        canvas.paste(ni_panel, (my_panel.size[0] + gap, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (16, canvas.size[1] - 18),
            "Use manifest.csv ordinal + label to report corrected cross-stain alignment.",
            fill=(180, 180, 180),
            font=font,
        )
        page_path = pages_dir / f"animal_{animal_id}_page_{page_idx + 1:02d}.png"
        canvas.save(page_path)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["animal_id", "side", "ordinal", "label", "sec_num", "crop_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = output_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"animal_id: {animal_id}",
                f"myelin_count: {len(myelin_sections)}",
                f"nissl_count: {len(nissl_sections)}",
                f"items_per_page: {items_per_page}",
                f"page_count: {page_count}",
                f"myelin_root: {myelin_root}",
                f"nissl_root: {nissl_root}",
                "",
                "Use manifest.csv to map ordinal numbers back to section labels.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal-id", type=int, required=True)
    parser.add_argument("--myelin-root", type=Path, default=_default_root("myelin"))
    parser.add_argument("--nissl-root", type=Path, default=_default_root("nissl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--items-per-page", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--thumb-height", type=int, default=160)
    args = parser.parse_args()
    export_pairing_thumbnail_qc(
        args.myelin_root,
        args.nissl_root,
        args.animal_id,
        args.output_dir,
        items_per_page=args.items_per_page,
        cols=args.cols,
        thumb_w=args.thumb_width,
        thumb_h=args.thumb_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
