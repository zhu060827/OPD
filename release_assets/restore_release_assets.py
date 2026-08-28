#!/usr/bin/env python3
"""Download, verify, and restore oversized files from the GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote


READ_BYTES = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, size_bytes: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size_bytes and sha256_file(path) == sha256


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"downloading {destination.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "OPD-release-restorer"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, READ_BYTES)
    os.replace(temporary, destination)


def safe_destination(repo_root: Path, relative_path: str) -> Path:
    destination = (repo_root / relative_path).resolve()
    destination.relative_to(repo_root)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--assets-dir", type=Path, default=Path("release_parts"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(__file__).with_name("LARGE_FILES_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    repo_root = args.repo_root.resolve()
    assets_dir = args.assets_dir.resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    base_url = (
        f"https://github.com/{manifest['repository']}/releases/download/"
        f"{quote(manifest['release_tag'])}"
    )

    for entry in manifest["files"]:
        destination = safe_destination(repo_root, entry["path"])
        if verified(destination, entry["size_bytes"], entry["sha256"]):
            print(f"already valid: {entry['path']}", flush=True)
            continue
        if args.verify_only:
            raise SystemExit(f"missing or invalid: {entry['path']}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        for part in entry["parts"]:
            part_path = assets_dir / part["name"]
            if not verified(part_path, part["size_bytes"], part["sha256"]):
                if not args.download:
                    raise SystemExit(f"missing or invalid part: {part_path}")
                download(f"{base_url}/{quote(part['name'])}", part_path)
                if not verified(part_path, part["size_bytes"], part["sha256"]):
                    raise SystemExit(f"download verification failed: {part_path}")

        temporary = destination.with_suffix(destination.suffix + ".restore")
        digest = hashlib.sha256()
        written = 0
        with temporary.open("wb") as output:
            for part in entry["parts"]:
                with (assets_dir / part["name"]).open("rb") as input_handle:
                    while chunk := input_handle.read(READ_BYTES):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
        if written != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise SystemExit(f"restored file verification failed: {entry['path']}")
        os.replace(temporary, destination)
        print(f"restored: {entry['path']}", flush=True)

    print("all oversized files are present and verified")


if __name__ == "__main__":
    main()
