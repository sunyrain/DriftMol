#!/usr/bin/env python3
"""Report or delete non-essential completed-run checkpoints.

The only automatically reclaimable checkpoint handled here is a run-local
``last.pt`` whose sibling directory already contains both a valid
``final_metrics.json`` and ``best.pt``. Active training output directories are
excluded from both dry-run reports and deletion.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "checkpoint_cleanup_candidates.md"
DEFAULT_DELETE_LOG = ROOT / "results" / "checkpoint_cleanup_deleted.md"


def human_size(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: str | Path, root: Path) -> Path:
    item = Path(path)
    if not item.is_absolute():
        item = root / item
    return item.resolve(strict=False)


def has_valid_final_metrics(run_dir: Path) -> bool:
    final_metrics = run_dir / "final_metrics.json"
    if not final_metrics.is_file():
        return False
    try:
        json.loads(final_metrics.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return True


def is_safe_last_checkpoint(last: Path, root: Path, active_run_dirs: set[Path]) -> bool:
    run_dir = last.parent.resolve(strict=False)
    if run_dir in active_run_dirs:
        return False
    if last.name != "last.pt" or last.is_symlink() or not last.is_file():
        return False
    best = last.parent / "best.pt"
    if best.is_symlink() or not best.is_file() or best.stat().st_size <= 0:
        return False
    if not has_valid_final_metrics(last.parent):
        return False
    try:
        last.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def load_output_dir_from_config(config_path: Path, root: Path) -> Path | None:
    config_file = resolve_path(config_path, root)
    if not config_file.is_file():
        return None
    try:
        config = yaml.safe_load(config_file.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(config, dict):
        return None
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        return None
    output_dir = experiment.get("output_dir")
    if not output_dir:
        return None
    return resolve_path(output_dir, root)


def extract_config_paths(command: str) -> list[Path]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    configs: list[Path] = []
    for idx, part in enumerate(parts):
        if part == "--config" and idx + 1 < len(parts):
            configs.append(Path(parts[idx + 1]))
        elif part.startswith("--config="):
            configs.append(Path(part.split("=", 1)[1]))
    return configs


def active_output_dirs_from_processes(root: Path) -> set[Path]:
    try:
        proc = subprocess.run(
            ["ps", "-ww", "-eo", "cmd"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    active_dirs: set[Path] = set()
    for command in proc.stdout.splitlines():
        for config_path in extract_config_paths(command):
            output_dir = load_output_dir_from_config(config_path, root)
            if output_dir is not None:
                active_dirs.add(output_dir)
    return active_dirs


def find_candidates(root: Path, active_run_dirs: set[Path] | None = None) -> list[tuple[int, Path]]:
    root = root.resolve(strict=False)
    active = active_run_dirs or set()
    candidates: list[tuple[int, Path]] = []
    for last in root.glob("outputs/**/last.pt"):
        if is_safe_last_checkpoint(last, root, active):
            candidates.append((last.stat().st_size, last))
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates


def write_report(
    path: Path,
    candidates: list[tuple[int, Path]],
    limit: int,
    active_run_dirs: set[Path] | None = None,
    deleted: list[tuple[int, Path]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(size for size, _ in candidates)
    shown = candidates[:limit] if limit > 0 else candidates
    deleted = deleted or []
    lines = [
        "# Checkpoint Cleanup Candidates",
        "",
        "Candidate rule: delete only completed-run `last.pt` files when sibling `best.pt` and valid `final_metrics.json` are present.",
        "Active training output directories are excluded.",
        "",
        f"Total reclaimable from listed completed-run `last.pt` files: {human_size(total)}",
        f"Candidate count: {len(candidates)}",
    ]
    if deleted:
        lines.extend(
            [
                f"Deleted in this run: {len(deleted)} files, {human_size(sum(size for size, _ in deleted))}",
                "",
            ]
        )
    else:
        lines.extend(["Dry run: no files were deleted.", ""])
    if active_run_dirs:
        lines.extend(["## Active Output Directories", ""])
        for active_dir in sorted(active_run_dirs, key=str):
            lines.append(f"- `{rel(active_dir)}`")
        lines.append("")
    lines.extend(["## Candidates", "", "| Size | File |", "|---:|---|"])
    for size, path_item in shown:
        lines.append(f"| {human_size(size)} | `{rel(path_item)}` |")
    if limit > 0 and len(candidates) > limit:
        lines.append(f"| ... | {len(candidates) - limit} more candidates omitted by display limit |")
    path.write_text("\n".join(lines) + "\n")


def delete_candidates(
    candidates: list[tuple[int, Path]],
    root: Path,
    active_run_dirs: set[Path],
) -> list[tuple[int, Path]]:
    deleted: list[tuple[int, Path]] = []
    for _, last in candidates:
        if not is_safe_last_checkpoint(last, root, active_run_dirs):
            continue
        size = last.stat().st_size
        last.unlink()
        deleted.append((size, last))
    return deleted


def write_delete_log(
    path: Path,
    deleted: list[tuple[int, Path]],
    active_run_dirs: set[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_lines = [
        "## Deletion Batch",
        "",
        f"Deleted files in batch: {len(deleted)}",
        f"Reclaimed in batch: {human_size(sum(size for size, _ in deleted))}",
        "",
        "### Deleted",
        "",
        "| Size | File |",
        "|---:|---|",
    ]
    for size, item in deleted:
        batch_lines.append(f"| {human_size(size)} | `{rel(item)}` |")
    batch_lines.extend(["", "### Active Output Directories Excluded", ""])
    for active_dir in sorted(active_run_dirs, key=str):
        batch_lines.append(f"- `{rel(active_dir)}`")

    if path.exists():
        existing = path.read_text().rstrip()
        if existing.startswith("# Deleted Checkpoint Audit Log"):
            path.write_text(existing + "\n\n" + "\n".join(batch_lines) + "\n")
            return

    lines = [
        "# Deleted Checkpoint Audit Log",
        "",
        "This log is append-only. Each batch is limited to completed-run `last.pt` checkpoints with sibling `best.pt` and valid `final_metrics.json`; active output directories are excluded.",
    ]
    lines.extend([""] + batch_lines)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--delete-log", default=str(DEFAULT_DELETE_LOG))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--delete", action="store_true", help="Delete the listed safe completed-run last.pt files.")
    args = parser.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    delete_log = Path(args.delete_log)
    if not delete_log.is_absolute():
        delete_log = ROOT / delete_log

    active_run_dirs = active_output_dirs_from_processes(ROOT)
    candidates = find_candidates(ROOT, active_run_dirs=active_run_dirs)
    deleted: list[tuple[int, Path]] = []
    if args.delete:
        deleted = delete_candidates(candidates, ROOT.resolve(strict=False), active_run_dirs)
        candidates = find_candidates(ROOT, active_run_dirs=active_run_dirs)
        write_delete_log(delete_log, deleted, active_run_dirs)
    write_report(out, candidates, args.limit, active_run_dirs=active_run_dirs, deleted=deleted)
    print(f"Wrote {rel(out)}")
    if args.delete:
        print(f"Wrote {rel(delete_log)}")
        print(f"Deleted: {len(deleted)}")
        print(f"Reclaimed: {human_size(sum(size for size, _ in deleted))}")
    print(f"Candidates: {len(candidates)}")
    print(f"Potential reclaim: {human_size(sum(size for size, _ in candidates))}")


if __name__ == "__main__":
    main()
