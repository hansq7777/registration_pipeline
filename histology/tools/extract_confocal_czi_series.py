#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DEFAULT_BFTOOLS_DIR = Path("/mnt/c/work/Myelin_anno_tool/bftools/bftools")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-extract confocal CZI series into per-series OME-TIFF tiles using the "
            "same folder and filename structure as the known-good 2501_60_R_IL export."
        )
    )
    parser.add_argument("--czi", type=Path, nargs="+", help="One or more .czi files to extract")
    parser.add_argument("--root", type=Path, help="Directory containing .czi files to extract")
    parser.add_argument(
        "--bftools-dir",
        type=Path,
        default=DEFAULT_BFTOOLS_DIR,
        help="Directory containing bfconvert.bat and showinf.bat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional root for output folders; defaults to each CZI parent directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing OME-TIFF outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()
    if not args.czi and args.root is None:
        parser.error("provide either --czi or --root")
    return args


def _windows_path(path: Path) -> str:
    proc = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _cmd_bat_command(script_path: Path, *args: str) -> list[str]:
    return ["cmd.exe", "/c", _windows_path(script_path), *args]


def _run_command(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    print(" ".join(cmd))
    if dry_run:
        return None
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _extract_series_count(showinf_text: str, *, czi_path: Path) -> int:
    match = re.search(r"Series count\s*=\s*(\d+)", showinf_text)
    if not match:
        raise RuntimeError(f"Could not parse series count from showinf output for {czi_path.name}")
    return int(match.group(1))


def _collect_czi_paths(args: argparse.Namespace) -> list[Path]:
    czi_paths: list[Path] = []
    if args.czi:
        czi_paths.extend(Path(path) for path in args.czi)
    if args.root is not None:
        czi_paths.extend(sorted(Path(args.root).glob("*.czi")))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in czi_paths:
        canonical = Path(path)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def _write_text(path: Path, text: str, *, dry_run: bool) -> None:
    print(f"[WRITE] {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_text(path: Path, text: str, *, dry_run: bool) -> None:
    print(f"[APPEND] {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _extract_one_czi(
    czi_path: Path,
    *,
    bftools_dir: Path,
    output_root: Path | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not czi_path.exists():
        raise FileNotFoundError(f"Missing CZI: {czi_path}")
    showinf_bat = bftools_dir / "showinf.bat"
    bfconvert_bat = bftools_dir / "bfconvert.bat"
    if not showinf_bat.exists() or not bfconvert_bat.exists():
        raise FileNotFoundError(f"Missing Bio-Formats CLI in {bftools_dir}")

    target_root = Path(output_root) if output_root is not None else czi_path.parent
    out_dir = target_root / czi_path.stem
    log_path = out_dir / f"{czi_path.stem}_bfconvert.log"
    showinf_path = out_dir / f"{czi_path.stem}_showinf_autostitch_false.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run and overwrite and log_path.exists():
        log_path.unlink()

    showinf_cmd = _cmd_bat_command(
        showinf_bat,
        "-nopix",
        "-novalid",
        "-option",
        "zeissczi.autostitch",
        "false",
        _windows_path(czi_path),
    )
    showinf_run = _run_command(showinf_cmd, dry_run=False)
    showinf_text = showinf_run.stdout + showinf_run.stderr
    _write_text(showinf_path, showinf_text, dry_run=dry_run)
    series_count = _extract_series_count(showinf_text, czi_path=czi_path)
    series_digits = max(2, len(str(max(0, series_count - 1))))

    for series_index in range(series_count):
        out_name = f"{czi_path.stem}_S{series_index:0{series_digits}d}.ome.tif"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite and not dry_run:
            raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out_path}")
        if not dry_run and overwrite and out_path.exists():
            out_path.unlink()
        prefix = f"[CONVERT] series={series_index} -> {out_name}\n"
        bfconvert_cmd = _cmd_bat_command(
            bfconvert_bat,
            "-series",
            str(series_index),
            "-novalid",
            "-no-upgrade",
            "-option",
            "zeissczi.autostitch",
            "false",
            "-overwrite",
            _windows_path(czi_path),
            _windows_path(out_path),
        )
        bfconvert_run = _run_command(bfconvert_cmd, dry_run=dry_run)
        bfconvert_text = "" if bfconvert_run is None else (bfconvert_run.stdout + bfconvert_run.stderr)
        _append_text(log_path, prefix + bfconvert_text + ("\n" if not bfconvert_text.endswith("\n") else ""), dry_run=dry_run)


def main() -> int:
    args = _parse_args()
    czi_paths = _collect_czi_paths(args)
    if not czi_paths:
        raise SystemExit("No .czi files found to extract.")
    for czi_path in czi_paths:
        _extract_one_czi(
            czi_path,
            bftools_dir=Path(args.bftools_dir),
            output_root=None if args.output_root is None else Path(args.output_root),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
