#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/BytedTsinghua-SIA/Open-MOPD.git"
PINNED_COMMIT="4809a96cf85a869106ff0ff3f37d0a51e12010ae"
TARGET="${1:-third_party/Open-MOPD}"

if [[ ! -d "${TARGET}/.git" ]]; then
    git clone "${REPO_URL}" "${TARGET}"
fi

git -C "${TARGET}" fetch origin "${PINNED_COMMIT}"
git -C "${TARGET}" checkout --detach "${PINNED_COMMIT}"
actual="$(git -C "${TARGET}" rev-parse HEAD)"
[[ "${actual}" == "${PINNED_COMMIT}" ]] || {
    echo "Open-MOPD commit mismatch: ${actual}" >&2
    exit 2
}
echo "Pinned Open-MOPD is ready at ${TARGET} (${actual})"
