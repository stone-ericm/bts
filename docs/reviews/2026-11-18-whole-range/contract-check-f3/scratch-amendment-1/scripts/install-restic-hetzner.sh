#!/usr/bin/env bash
# Install a pinned, checksum-verified restic binary to ~/.local/bin (no root).
#
# Part of audit F5 (operational-state backup). Idempotent: re-running with the
# same pin is a no-op if the installed binary already reports that version.
# Bump RESTIC_VERSION deliberately; the checksum comes from the release's own
# SHA256SUMS file, so a moved/tampered artifact fails hard.
#
# Usage: bash scripts/install-restic-hetzner.sh

set -euo pipefail

RESTIC_VERSION="0.19.1"
ARCH="linux_amd64"
DEST="$HOME/.local/bin/restic"
BASE_URL="https://github.com/restic/restic/releases/download/v${RESTIC_VERSION}"
ASSET="restic_${RESTIC_VERSION}_${ARCH}.bz2"

if [ -x "$DEST" ] && "$DEST" version 2>/dev/null | grep -q "restic ${RESTIC_VERSION}"; then
    echo "restic ${RESTIC_VERSION} already installed at $DEST"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ${ASSET}..."
curl -fsSL -o "$TMP/$ASSET" "$BASE_URL/$ASSET"
curl -fsSL -o "$TMP/SHA256SUMS" "$BASE_URL/SHA256SUMS"

echo "Verifying checksum..."
(cd "$TMP" && grep " ${ASSET}\$" SHA256SUMS | sha256sum -c -)

bunzip2 -f "$TMP/$ASSET"
mkdir -p "$(dirname "$DEST")"
install -m 0755 "$TMP/restic_${RESTIC_VERSION}_${ARCH}" "$DEST"

"$DEST" version
echo "Installed to $DEST"
