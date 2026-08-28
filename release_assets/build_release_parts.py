#!/usr/bin/env python3
"""Split GitHub-oversized project files into verifiable Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


GIT_LIMIT_BYTES = 100 * 1024 * 1024
DEFAULT_PART_BYTES = 1900 * 1024 * 1024
READ_BYTES = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def asset_stem(relative_path: Path) -> str:
    raw = relative_path.as_posix().replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    path_id = hashlib.sha256(relative_path.as_posix().encode()).hexdigest()[:10]
    return f"{safe}.{path_id}"


def iter_large_files(repo_root: Path) -> list[Path]:
    result: list[Path] = []
    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if ".git" in relative.parts or not path.is_file():
            continue
        if path.stat().st_size >= GIT_LIMIT_BYTES:
            result.append(path)
    return sorted(result, key=lambda item: item.relative_to(repo_root).as_posix())


def split_file(source: Path, relative: Path, output_dir: Path, part_bytes: int) -> dict:
    original_digest = hashlib.sha256()
    parts: list[dict] = []
    stem = asset_stem(relative)

    with source.open("rb") as input_handle:
        index = 0
        while True:
            first_chunk = input_handle.read(min(READ_BYTES, part_bytes))
            if not first_chunk:
                break

            part_name = f"{stem}.part-{index:03d}"
            part_path = output_dir / part_name
            part_digest = hashlib.sha256()
            written = 0

            with part_path.open("wb") as output_handle:
                chunk = first_chunk
                while chunk:
                    output_handle.write(chunk)
                    original_digest.update(chunk)
                    part_digest.update(chunk)
                    written += len(chunk)
                    remaining = part_bytes - written
                    if remaining == 0:
                        break
                    chunk = input_handle.read(min(READ_BYTES, remaining))

            parts.append(
                {
                    "name": part_name,
                    "size_bytes": written,
                    "sha256": part_digest.hexdigest(),
                }
            )
            print(f"created {part_name}: {written} bytes", flush=True)
            index += 1

    source_size = source.stat().st_size
    assert sum(part["size_bytes"] for part in parts) == source_size
    return {
        "path": relative.as_posix(),
        "size_bytes": source_size,
        "sha256": original_digest.hexdigest(),
        "parts": parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.part_bytes >= 2 * 1024 * 1024 * 1024:
        raise SystemExit("part size must stay below GitHub Free's 2 GiB per-file limit")

    large_files = iter_large_files(repo_root)
    print(f"large files: {len(large_files)}", flush=True)
    entries = [
        split_file(path, path.relative_to(repo_root), output_dir, args.part_bytes)
        for path in large_files
    ]
    manifest = {
        "schema": "opd-github-release-parts-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": "ilovecplusplus230/-OPD",
        "release_tag": "full-project-snapshot-2026-08-28",
        "git_limit_bytes": GIT_LIMIT_BYTES,
        "part_size_limit_bytes": args.part_bytes,
        "large_file_count": len(entries),
        "large_file_total_bytes": sum(entry["size_bytes"] for entry in entries),
        "part_count": sum(len(entry["parts"]) for entry in entries),
        "files": entries,
    }
    manifest_path = Path(__file__).with_name("LARGE_FILES_MANIFEST.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
