"""
Container-friendly training entrypoint for the MAFIA TD3 controller.

Features:
- Runs one or many seeds (wraps scripts/run_multi_seed_experiment.py::run_one)
- Packages run artifacts and metrics into a tar.gz archive
- Optionally uploads the archive to a remote location (local path or rclone remote)

Example:
    python train.py --num-seeds 1 --base-seed 2025 --upload-uri /mnt/drive/checkpoints
"""

import argparse
import base64
import datetime
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from scripts.run_multi_seed_experiment import aggregate, run_one


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD3+MAFIA and optionally upload checkpoints/artifacts."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=os.environ.get("MAFIA_SEEDS", ""),
        help="Comma-separated list of seeds to run (overrides num/base).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=_env_int("MAFIA_NUM_SEEDS", 1),
        help="Number of runs when --seeds is empty.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=_env_int("MAFIA_BASE_SEED", 2025),
        help="Base seed; seeds are base_seed + i when --seeds empty.",
    )
    parser.add_argument(
        "--upload-uri",
        type=str,
        default=os.environ.get("MAFIA_UPLOAD_URI")
        or os.environ.get("UPLOAD_URI"),
        help=(
            "Destination for the packaged archive. "
            "Can be a local path (host-mounted) or an rclone remote like gdrive:folder."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=os.environ.get("MAFIA_ARTIFACT_DIR", "./artifacts"),
        help="Local folder to store packaged artifacts before upload.",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default=os.environ.get("MAFIA_ARCHIVE_NAME", ""),
        help="Optional archive name without extension. Defaults to timestamp-based.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep the packaged archive after successful upload.",
    )
    parser.add_argument(
        "--res-root",
        type=str,
        default=os.environ.get("MAFIA_RES_ROOT"),
        help="Override output root for results/checkpoints (mirrors MAFIA_RES_ROOT env).",
    )
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(s) for s in args.seeds.split(",") if s.strip()]
    return [args.base_seed + i for i in range(args.num_seeds)]


def ensure_rclone_config():
    encoded = os.environ.get("RCLONE_CONFIG_BASE64")
    if not encoded:
        return
    config_dir = Path.home() / ".config" / "rclone"
    config_dir.mkdir(parents=True, exist_ok=True)
    conf_path = config_dir / "rclone.conf"
    conf_path.write_text(base64.b64decode(encoded).decode("utf-8"))


def upload_artifact(archive_path: Path, dest_uri: str) -> Optional[str]:
    if not dest_uri:
        return None

    # Use rclone when a remote-like URI is supplied
    rclone_prefixes = (
        "rclone:",
        "gdrive:",
        "drive:",
        "s3:",
        "s3://",
        "gs://",
        "gcs://",
        "dropbox:",
        "onedrive:",
    )
    should_use_rclone = dest_uri.startswith(rclone_prefixes)
    if should_use_rclone:
        ensure_rclone_config()
        remote_target = dest_uri
        if dest_uri.startswith("rclone:"):
            remote_target = dest_uri[len("rclone:") :]
        cmd = ["rclone", "copy", str(archive_path), remote_target]
        print(f"[upload] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return f"{remote_target.rstrip('/')}/{archive_path.name}"

    # Fallback: copy to local/host-mounted path
    dest_dir = Path(dest_uri)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / archive_path.name
    shutil.copy2(archive_path, target)
    return str(target)


def to_native(val):
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val


def package_runs(
    res_dirs: Sequence[str],
    run_metrics: List[dict],
    summary_df,
    artifact_dir: Path,
    archive_name: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not archive_name:
        archive_name = datetime.datetime.utcnow().strftime("mafia-train-%Y%m%d-%H%M%S")
    archive_path = artifact_dir / f"{archive_name}.tar.gz"
    summary_path = artifact_dir / f"{archive_name}-summary.json"

    summary_records = []
    if summary_df is not None:
        summary_records = (
            summary_df.reset_index().rename(columns={"index": "metric"}).to_dict(orient="records")
        )
    payload = {
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "runs": [{k: to_native(v) for k, v in m.items()} for m in run_metrics],
        "summary": summary_records,
        "res_dirs": list(res_dirs),
    }
    summary_path.write_text(json.dumps(payload, indent=2))

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(summary_path, arcname="summary.json")
        for res_dir in res_dirs:
            tar.add(res_dir, arcname=f"runs/{Path(res_dir).name}")
    return archive_path


def main():
    args = parse_args()
    seeds = resolve_seeds(args)
    if not seeds:
        raise SystemExit("No seeds specified.")

    if args.res_root:
        os.environ["MAFIA_RES_ROOT"] = args.res_root

    print(f"[train] Running seeds: {seeds}")
    results = []
    res_dirs: List[str] = []
    for seed in seeds:
        metrics = run_one(seed)
        results.append(metrics)
        res_dirs.append(metrics["res_dir"])

    _, summary_df = aggregate(results)
    archive_path = package_runs(
        res_dirs=res_dirs,
        run_metrics=results,
        summary_df=summary_df,
        artifact_dir=Path(args.artifact_dir),
        archive_name=args.archive_name,
    )
    print(f"[train] Packaged artifacts -> {archive_path}")

    uploaded_to = None
    if args.upload_uri:
        uploaded_to = upload_artifact(archive_path, args.upload_uri)
        print(f"[train] Uploaded archive to: {uploaded_to}")
        if not args.keep_local:
            archive_path.unlink(missing_ok=True)

    print("[train] Done.")
    if uploaded_to:
        print(f"[train] Remote location: {uploaded_to}")
    else:
        print(f"[train] Local archive: {archive_path}")
        print("        (mount a host or drive path via -v to persist the archive)")


if __name__ == "__main__":
    main()
