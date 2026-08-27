#!/usr/bin/env bash
# Restore search indexes onto the fast local NVMe.
#
# /datadisk is local NVMe: ~14x the sequential throughput of the managed disks and far
# better random IOPS, which is what Lucene needs. It is also ephemeral - wiped when the
# VM deallocates - so the durable copy stays on the slow disk and is synced across at
# boot. Rebuilding instead would cost hours; this costs minutes, and nothing when the
# index is already present.
#
# Pairs are "source|destination". Add more as new indexes appear.

set -uo pipefail

FAST_ROOT="${FAST_ROOT:-/datadisk}"

PAIRS=(
    "$HOME/openalex-data/index|$FAST_ROOT/openalex-data/index"
    "$HOME/sciencepcm-data/index|$FAST_ROOT/sciencepcm-data/index"
)

if [[ ! -d "$FAST_ROOT" ]]; then
    echo "$FAST_ROOT does not exist; nothing to restore." >&2
    exit 0
fi

if [[ ! -w "$FAST_ROOT" ]]; then
    echo "$FAST_ROOT is not writable by $(id -un). Run: sudo chown \"\$(id -u):\$(id -g)\" $FAST_ROOT" >&2
    exit 1
fi

for pair in "${PAIRS[@]}"; do
    source="${pair%%|*}"
    destination="${pair##*|}"

    if [[ ! -d "$source" ]]; then
        echo "skip   $source (not present)"
        continue
    fi

    mkdir -p "$destination"
    echo "sync   $source -> $destination"

    # Trailing slashes matter: copy the CONTENTS, not the directory into itself.
    # --delete keeps the fast copy an exact mirror after an index rebuild.
    rsync -a --delete --info=stats2 "$source/" "$destination/" || {
        echo "rsync failed for $source" >&2
        exit 1
    }
done

echo "indexes ready under $FAST_ROOT"
