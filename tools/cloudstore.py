"""All cloud transfer for this project, through RemoteBlobStore.

    pip install git+https://github.com/markusmobius/newsprinceton-pythoncloud

Three levels, used for different things:

  sync                  Uploads the fixed set of directories nerds21 owns, skipping
                        anything the cloud already has. Driven by MANIFEST below.

  push-dir / pull-dir   Plain recursive directory transfer, for anything not in the
                        manifest.

  push / pull           Config-versioned single artifacts. Keyed by a sha256 over the
                        pipeline Stage list, which is built from ingest-report.json and
                        embed-report.json so the key cannot drift from what produced the
                        data. Used for vectors and indexes, where serving the wrong
                        version is silent rather than loud.

Auth comes from CLOUDPDS_CLIENT_HASH or legopds_clienthash.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_SERVER = "https://www.legopds.projectratio.net:6008"

NERDS21 = Path(r"\\nerds21\sciencepcm")

# Everything uploaded lives under __temp. Partly because it is all derived and
# disposable, and partly because the uploader opens source files read-write to
# fingerprint them, which fails on the read-only dataset directories.
TEMP = NERDS21 / "mcpserver" / "__temp"


class Entry:
    def __init__(self, local: Path, cloud: str, why: str, optional: bool = False):
        self.local = local
        self.cloud = cloud
        self.why = why
        self.optional = optional


MANIFEST = [
    Entry(
        TEMP / "abstracts",
        "sciencepcm/abstracts",
        "5.3M abstracts - the retrieval benchmark tier",
    ),
    Entry(
        TEMP / "passages-2019-2025",
        "sciencepcm/passages-2019-2025",
        "REGENERABLE ONLY ON NERDS21 - needs the 62.8 GB of PMC/bioRxiv/medRxiv XML",
    ),
    Entry(
        TEMP / "questions",
        "sciencepcm/questions",
        "BioASQ + v0.2 evaluation questions",
    ),
    Entry(
        TEMP / "openalex-works",
        "sciencepcm/openalex-works",
        "19 GB of full OpenAlex records; only needed for metadata beyond the abstracts projection",
        optional=True,
    ),
]


# ----------------------------------------------------------------- plumbing


def client_hash() -> str:
    for name in ("CLOUDPDS_CLIENT_HASH", "legopds_clienthash"):
        value = os.getenv(name)
        if value:
            return value
    raise SystemExit("Set CLOUDPDS_CLIENT_HASH (or legopds_clienthash) before using this tool.")


def open_cloud(args):
    from RemoteBlobStore.Remote.RemoteBlobServer import RemoteBlobServer
    from RemoteBlobStore.Remote.Runners.CloudMachine import CloudMachine

    server = RemoteBlobServer()
    cloud = CloudMachine(port=server.port, clientHash=client_hash(), serverUrl=args.server)
    return server, cloud


def open_machine(args):
    from RemoteBlobStore.DataVersionMachine import DataVersionMachine

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    return DataVersionMachine(
        clientHash=client_hash(),
        serverUrl=args.server,
        cacheFolder=str(args.cache),
    )


def tag_rules():
    from RemoteBlobStore.Remote.Runners.UploadTask import BlobFilter, TagRule

    return [
        TagRule(
            key="project",
            value="sciencepcm",
            kvFilter=BlobFilter(ftype="pattern", filterDefinition="*"),
        )
    ]


def cloud_names(cloud, path: str) -> set[str] | None:
    """File names present in a cloud directory, or None when it does not exist."""
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


def git_version(repo: Path | None = None) -> str:
    """Code version for a Stage. Artifacts should change when the code that made them does."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo or Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_stages(report_paths: list[Path], code_version: str):
    """Turn ingest-report.json / embed-report.json into the Stage list.

    Reading the reports rather than accepting hand-written config is deliberate: the
    key cannot then drift from what actually produced the data.
    """
    from RemoteBlobStore.DataVersionMachine import Stage

    stages = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.replace("-report", "")

        if name == "embed":
            config = {
                "model": report.get("model", {}),
                "input": report.get("input"),
                "schema": report.get("schema"),
                "include_title": report.get("include_title"),
                "dimensions": report.get("dimensions"),
                "vectors": report.get("vectors_written"),
                "vector_format": report.get("vector_format"),
            }
        elif name == "ingest":
            config = {
                "inputs": report.get("inputs"),
                "year_min": report.get("year_min"),
                "year_max": report.get("year_max"),
                "chunking": report.get("chunking"),
                "articles": report.get("counts", {}).get("articles_written"),
                "chunks": report.get("counts", {}).get("chunks_written"),
            }
        else:
            config = report

        stages.append(Stage(name, code_version, config))

    if not stages:
        raise SystemExit("At least one --report is required.")
    return stages


# ----------------------------------------------------------------- commands


def cmd_sync(args) -> int:
    entries = [e for e in MANIFEST if args.include_optional or not e.optional]
    if args.only:
        entries = [e for e in entries if e.cloud in args.only]

    server, cloud = open_cloud(args)
    uploaded = skipped = missing = 0

    try:
        for entry in entries:
            print(f"\n=== {entry.cloud}")
            print(f"    {entry.why}")

            if not entry.local.exists():
                print(f"    MISSING LOCALLY: {entry.local}")
                missing += 1
                continue

            files = sorted(p for p in entry.local.rglob("*") if p.is_file())
            size_gb = sum(f.stat().st_size for f in files) / 1024**3
            print(f"    local: {len(files):,} files, {size_gb:,.2f} GB")

            remote = cloud_names(cloud, entry.cloud)
            if remote is None:
                print("    cloud: directory does not exist")
            elif args.force:
                print(f"    cloud: {len(remote):,} files present, --force set")
            else:
                outstanding = {f.name for f in files} - remote
                if not outstanding:
                    print(f"    cloud: all {len(files):,} files present - skipping")
                    print("           names only are compared; use --force after re-ingesting")
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
                tagRules=tag_rules(),
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


def cmd_push_dir(args) -> int:
    server, cloud = open_cloud(args)
    try:
        print(f"Uploading {args.local} -> {args.cloud}")
        cloud.Upload(
            localDirectory=str(args.local),
            cloudDirectory=args.cloud,
            tagRules=tag_rules(),
            publicRules=[],
            recursiveUpload=not args.flat,
        )
        print("upload complete")
    finally:
        server.Dispose()
    return 0


def cmd_pull_dir(args) -> int:
    from RemoteBlobStore.Remote.Runners.DownloadTask import DirectorySet, DownloadTask

    server, cloud = open_cloud(args)
    try:
        Path(args.local).mkdir(parents=True, exist_ok=True)
        path = args.cloud if args.cloud.endswith("/") else args.cloud + "/"
        print(f"Downloading {path} -> {args.local}")
        cloud.Download(downloads=DownloadTask(str(args.local), DirectorySet(paths=[path])))
        print("download complete")
    finally:
        server.Dispose()
    return 0


def cmd_list(args) -> int:
    server, cloud = open_cloud(args)
    try:
        status = cloud.Directory(cloudDirectory=args.cloud)
        if status is None:
            print(f"{args.cloud}: does not exist")
            return 1
        print(json.dumps(status, default=lambda x: x.__dict__, indent=2))
    finally:
        server.Dispose()
    return 0


def cmd_push(args) -> int:
    machine = open_machine(args)
    try:
        stages = build_stages(args.report, args.code_version or git_version())
        print(f"stages: {machine.getConfigJson(stages)}")

        existing = machine.loadVersion(cloudPath=args.cloud, stages=stages, debug=args.debug)
        if existing and not args.force:
            print(f"Already stored for this configuration: {existing}")
            return 0

        # saveVersion requires the temp file's basename to equal the config hash.
        temp = machine.getTempFile(stages)
        source = Path(args.local)

        print(f"Packing {source} -> {temp}")
        mode = "w:gz" if args.compress else "w"
        with tarfile.open(temp, mode) as archive:
            for item in sorted(source.rglob("*")):
                if item.is_file():
                    archive.add(item, arcname=str(item.relative_to(source)))

        size_gb = Path(temp).stat().st_size / 1024**3
        print(f"Uploading {size_gb:,.2f} GB to {args.cloud}")
        machine.saveVersion(
            tempFileName=temp,
            cloudPath=args.cloud,
            stages=stages,
            logs=[{"tool": "cloudstore.py", "source": str(source)} for _ in stages],
            debug=args.debug,
        )
        print("saved")
    finally:
        machine.Dispose()
    return 0


def cmd_pull(args) -> int:
    machine = open_machine(args)
    try:
        stages = build_stages(args.report, args.code_version or git_version())
        print(f"stages: {machine.getConfigJson(stages)}")

        local = machine.loadVersion(cloudPath=args.cloud, stages=stages, debug=args.debug)
        if local is None:
            print("No artifact stored for this configuration.")
            return 1

        destination = Path(args.out)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Expanding {local} -> {destination}")
        with tarfile.open(local) as archive:
            archive.extractall(destination, filter="data")
        print("done")
    finally:
        machine.Dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=os.getenv("CLOUDPDS_SERVER", DEFAULT_SERVER))
    parser.add_argument("--cache", type=Path, default=Path(os.getenv("CLOUDPDS_CACHE", ".artifact-cache")))
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Upload the nerds21 manifest, skipping what is already there.")
    sync.add_argument("--check", action="store_true", help="Report status, transfer nothing.")
    sync.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even when the names all match. Needed after re-ingesting, because "
        "a rebuilt corpus reuses the same shard filenames with different contents.",
    )
    sync.add_argument("--include-optional", action="store_true")
    sync.add_argument("--only", action="append", help="Sync only these cloud paths. Repeatable.")
    sync.set_defaults(func=cmd_sync)

    push_dir = sub.add_parser("push-dir", help="Recursive directory upload.")
    push_dir.add_argument("--local", type=Path, required=True)
    push_dir.add_argument("--cloud", required=True)
    push_dir.add_argument("--flat", action="store_true", help="Do not recurse.")
    push_dir.set_defaults(func=cmd_push_dir)

    pull_dir = sub.add_parser("pull-dir", help="Recursive directory download.")
    pull_dir.add_argument("--cloud", required=True)
    pull_dir.add_argument("--local", type=Path, required=True, help="Local root directory.")
    pull_dir.set_defaults(func=cmd_pull_dir)

    listing = sub.add_parser("list", help="Show a cloud directory.")
    listing.add_argument("--cloud", required=True)
    listing.set_defaults(func=cmd_list)

    push = sub.add_parser("push", help="Config-versioned artifact upload.")
    push.add_argument("--local", type=Path, required=True, help="Directory to archive.")
    push.add_argument("--cloud", required=True)
    push.add_argument("--report", type=Path, action="append", required=True,
                      help="ingest-report.json / embed-report.json. Repeatable, in pipeline order.")
    push.add_argument("--code-version", default=None, help="Defaults to the short git SHA.")
    push.add_argument("--compress", action="store_true", help="gzip the tar. Float vectors barely compress.")
    push.add_argument("--force", action="store_true")
    push.add_argument("--debug", action="store_true", help="Local cache only, no network.")
    push.set_defaults(func=cmd_push)

    pull = sub.add_parser("pull", help="Config-versioned artifact download.")
    pull.add_argument("--cloud", required=True)
    pull.add_argument("--out", type=Path, required=True)
    pull.add_argument("--report", type=Path, action="append", required=True)
    pull.add_argument("--code-version", default=None)
    pull.add_argument("--debug", action="store_true")
    pull.set_defaults(func=cmd_pull)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
