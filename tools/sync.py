"""Sync the files the A100 box needs from nerds21 into RemoteBlobStore.

Idempotent: each entry is skipped when the cloud directory already holds every
local file name, so re-running costs a directory listing and nothing else.

nerds21 stays the durable archive. The Azure box is a 99-day working copy, so
anything expensive to recompute must live here before it is deleted there -
in particular passages-2023-2025, which is derived from 55 GB of PMC XML that
exists only on nerds21.

    python tools/sync.py --check     # report, transfer nothing
    python tools/sync.py             # sync what is missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifacts import DEFAULT_SERVER, client_hash  # noqa: E402

NERDS21 = Path(r"\\nerds21\sciencepcm")

# Anything under __temp is derived and disposable; sync.ps1 rebuilds what is missing.
TEMP = NERDS21 / "mcpserver" / "__temp"


class Entry:
    def __init__(self, local: Path, cloud: str, why: str, optional: bool = False):
        self.local = local
        self.cloud = cloud
        self.why = why
        self.optional = optional


MANIFEST = [
    Entry(
        NERDS21 / "dataset" / "OpenAlex-neuroscience-abstracts" / "data",
        "sciencepcm/abstracts",
        "5.3M abstracts - the retrieval benchmark tier",
    ),
    Entry(
        TEMP / "passages-2019-2025",
        "sciencepcm/passages-2019-2025",
        "REGENERABLE ONLY ON NERDS21 - needs the 62.8 GB of PMC/bioRxiv/medRxiv XML",
    ),
    Entry(
        NERDS21 / "dataset" / "Questions-neuroscience",
        "sciencepcm/questions",
        "BioASQ + v0.2 evaluation questions",
    ),
    Entry(
        TEMP / "models",
        "sciencepcm/models",
        "exported MedCPT ONNX; only needed if HuggingFace is unreachable from the A100 box",
        optional=True,
    ),
    Entry(
        NERDS21 / "dataset" / "OpenAlex-neuroscience" / "data",
        "sciencepcm/openalex-works",
        "19 GB of full OpenAlex records; only needed for metadata beyond the abstracts projection",
        optional=True,
    ),
]


def local_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def cloud_names(cloud, path: str) -> set[str] | None:
    """Returns the file names present, or None when the directory does not exist."""
    status = cloud.Directory(cloudDirectory=path)
    if status is None:
        return None

    files = getattr(status, "files", None)
    if files is None and isinstance(status, dict):
        files = status.get("files")
    if files is None:
        return set()

    names = set()
    for item in files:
        if isinstance(item, str):
            names.add(Path(item).name)
        else:
            value = getattr(item, "name", None) or getattr(item, "Name", None)
            if value:
                names.add(Path(str(value)).name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Report status and transfer nothing.")
    parser.add_argument("--include-optional", action="store_true", help="Include the 19 GB OpenAlex records.")
    parser.add_argument("--only", action="append", help="Sync only these cloud paths. Repeatable.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    args = parser.parse_args()

    entries = [e for e in MANIFEST if args.include_optional or not e.optional]
    if args.only:
        entries = [e for e in entries if e.cloud in args.only]

    from RemoteBlobStore.Remote.RemoteBlobServer import RemoteBlobServer
    from RemoteBlobStore.Remote.Runners.CloudMachine import CloudMachine
    from RemoteBlobStore.Remote.Runners.UploadTask import BlobFilter, TagRule

    server = RemoteBlobServer()
    cloud = CloudMachine(port=server.port, clientHash=client_hash(), serverUrl=args.server)

    uploaded = 0
    skipped = 0
    missing = 0

    try:
        for entry in entries:
            print(f"\n=== {entry.cloud}")
            print(f"    {entry.why}")

            if not entry.local.exists():
                print(f"    MISSING LOCALLY: {entry.local}")
                missing += 1
                continue

            files = local_files(entry.local)
            size_gb = sum(f.stat().st_size for f in files) / 1024**3
            print(f"    local: {len(files):,} files, {size_gb:,.2f} GB")

            remote = cloud_names(cloud, entry.cloud)
            if remote is None:
                print("    cloud: directory does not exist")
            else:
                outstanding = {f.name for f in files} - remote
                if not outstanding:
                    print(f"    cloud: all {len(files):,} files present - skipping")
                    skipped += 1
                    continue
                print(f"    cloud: {len(remote):,} files present, {len(outstanding):,} missing")

            if args.check:
                print("    --check set, not uploading")
                continue

            print(f"    uploading {entry.local} ...")
            cloud.Upload(
                localDirectory=str(entry.local),
                cloudDirectory=entry.cloud,
                tagRules=[
                    TagRule(
                        key="project",
                        value="sciencepcm",
                        kvFilter=BlobFilter(ftype="pattern", filterDefinition="*"),
                    )
                ],
                publicRules=[],
                recursiveUpload=True,
            )
            uploaded += 1
            print("    done")
    finally:
        server.Dispose()

    print(f"\nuploaded {uploaded}, already synced {skipped}, missing locally {missing}")
    if missing:
        print("Some sources were missing. Do not delete anything on nerds21 until that is resolved.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
