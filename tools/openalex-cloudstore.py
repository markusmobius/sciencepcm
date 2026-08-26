"""Transfer the OpenAlex abstract digest through RemoteBlobStore.

Auth comes from CLOUDPDS_CLIENT_HASH or legopds_clienthash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_SERVER = os.getenv("CLOUDPDS_SERVER_URL", "https://www.legopds.projectratio.net:6008")
DEFAULT_CLOUD = "openalex/abstracts"


def client_hash() -> str:
    for name in ("CLOUDPDS_CLIENT_HASH", "legopds_clienthash"):
        if value := os.getenv(name):
            return value
    raise SystemExit("Set CLOUDPDS_CLIENT_HASH (or legopds_clienthash).")


def open_cloud(args):
    from RemoteBlobStore.Remote.RemoteBlobServer import RemoteBlobServer
    from RemoteBlobStore.Remote.Runners.CloudMachine import CloudMachine

    server = RemoteBlobServer()
    cloud = CloudMachine(port=server.port, clientHash=client_hash(), serverUrl=args.server)
    return server, cloud


def tag_rules():
    from RemoteBlobStore.Remote.Runners.UploadTask import BlobFilter, TagRule

    return [
        TagRule(
            key="project",
            value="openalex",
            kvFilter=BlobFilter(ftype="pattern", filterDefinition="*"),
        )
    ]


def remote_file_names(cloud, path: str) -> set[str] | None:
    status = cloud.Directory(cloudDirectory=path)
    if status is None:
        return None
    files = getattr(status, "files", None)
    if files is None and isinstance(status, dict):
        files = status.get("files")
    names = set()
    for item in files or []:
        value = item if isinstance(item, str) else getattr(item, "name", None) or getattr(item, "Name", None)
        if value:
            names.add(Path(str(value)).name)
    return names


def sync(args) -> int:
    source = args.local.resolve()
    if not (source / "openalex-ingest-report.json").exists():
        raise SystemExit(f"No openalex-ingest-report.json in {source}; refusing to upload an incomplete digest.")

    files = sorted(path for path in source.rglob("*") if path.is_file())
    size_gb = sum(path.stat().st_size for path in files) / 1024**3
    server, cloud = open_cloud(args)
    try:
        remote = remote_file_names(cloud, args.cloud)
        outstanding = set(path.name for path in files) - (remote or set())
        print(f"local : {len(files):,} files, {size_gb:,.2f} GB at {source}")
        print(f"cloud : {args.cloud} ({0 if remote is None else len(remote):,} files present)")
        if remote is not None and not outstanding and not args.force:
            print("already synced by file name; use --force after rebuilding the digest")
            return 0
        if args.check:
            print(f"check only: {len(outstanding):,} file names are absent")
            return 0

        cloud.Upload(
            localDirectory=str(source),
            cloudDirectory=args.cloud,
            tagRules=tag_rules(),
            publicRules=[],
            recursiveUpload=True,
        )
        print("upload complete")
        return 0
    finally:
        server.Dispose()


def pull(args) -> int:
    from RemoteBlobStore.Remote.Runners.DownloadTask import DirectorySet, DownloadTask

    args.local.mkdir(parents=True, exist_ok=True)
    server, cloud = open_cloud(args)
    try:
        path = args.cloud.rstrip("/") + "/"
        print(f"downloading {path} -> {args.local}")
        cloud.Download(downloads=DownloadTask(str(args.local), DirectorySet(paths=[path])))
        print("download complete")
        return 0
    finally:
        server.Dispose()


def list_cloud(args) -> int:
    server, cloud = open_cloud(args)
    try:
        status = cloud.Directory(cloudDirectory=args.cloud)
        if status is None:
            print(f"{args.cloud}: does not exist")
            return 1
        print(json.dumps(status, default=lambda value: value.__dict__, indent=2))
        return 0
    finally:
        server.Dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.getenv("CLOUDPDS_SERVER", DEFAULT_SERVER))
    parser.add_argument("--cloud", default=DEFAULT_CLOUD)
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("sync", help="Upload the nerds21 OpenAlex digest.")
    upload.add_argument("--local", type=Path, required=True)
    upload.add_argument("--check", action="store_true")
    upload.add_argument("--force", action="store_true")
    upload.set_defaults(handler=sync)

    download = commands.add_parser("pull", help="Download the digest on the A100 machine.")
    download.add_argument("--local", type=Path, required=True)
    download.set_defaults(handler=pull)

    listing = commands.add_parser("list", help="List the cloud digest without downloading it.")
    listing.set_defaults(handler=list_cloud)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())