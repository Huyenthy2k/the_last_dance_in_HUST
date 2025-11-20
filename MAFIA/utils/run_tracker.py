#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for tracking run/checkpoint metadata outside TensorBoard.
Creates and maintains a manifest JSON per run so resume/metrics status
can be inspected even when training restarts multiple times.
"""
import json
import os
import time
from typing import Dict, List, Optional

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
MAX_HISTORY = 20  # keep last N events per list to avoid unbounded growth


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_manifest(manifest_path: str) -> Optional[Dict]:
    if not manifest_path:
        return None
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[run_tracker] Warning: failed to read {manifest_path}: {e}", flush=True)
        return None


def _save_manifest(manifest_path: str, manifest: Dict) -> None:
    if not manifest_path:
        return
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


def ensure_manifest(manifest_path: str, run_id: Optional[str], total_epochs: Optional[int]) -> Dict:
    """
    Ensure a manifest file exists and basic keys are populated.
    Returns the manifest dictionary (freshly written if needed).
    """
    if not manifest_path:
        return {}
    manifest = _load_manifest(manifest_path) or {}
    manifest.setdefault("version", MANIFEST_VERSION)
    manifest.setdefault("created_at", _now_ts())
    if run_id:
        manifest["run_id"] = run_id
    if total_epochs is not None:
        manifest["total_epochs"] = int(total_epochs)
    manifest.setdefault("completed_epochs", 0)
    manifest.setdefault("last_checkpoint", None)
    manifest.setdefault("last_metrics_epoch", None)
    manifest.setdefault("resume_events", [])
    manifest.setdefault("checkpoint_history", [])
    manifest.setdefault("metrics_history", [])
    manifest.setdefault("notes", "")
    _save_manifest(manifest_path, manifest)
    return manifest


def _trim_list(items: List[Dict]) -> List[Dict]:
    if len(items) <= MAX_HISTORY:
        return items
    return items[-MAX_HISTORY:]


def record_resume_event(manifest_path: str, checkpoint_info_path: str, checkpoint_info: Dict, reason: str = "auto") -> None:
    if not manifest_path:
        return
    manifest = ensure_manifest(
        manifest_path,
        run_id=os.path.basename(os.path.dirname(manifest_path)),
        total_epochs=None,
    )
    event = {
        "timestamp": _now_ts(),
        "reason": reason,
        "checkpoint_info_path": checkpoint_info_path,
        "checkpoint_type": checkpoint_info.get("type"),
        "epoch": checkpoint_info.get("epoch"),
        "timesteps": checkpoint_info.get("timesteps"),
    }
    manifest["resume_events"].append(event)
    manifest["resume_events"] = _trim_list(manifest["resume_events"])
    manifest["last_resume"] = event
    _save_manifest(manifest_path, manifest)


def record_checkpoint_save(manifest_path: str, checkpoint_name: str, checkpoint_info: Dict) -> None:
    if not manifest_path:
        return
    manifest = ensure_manifest(
        manifest_path,
        run_id=os.path.basename(os.path.dirname(manifest_path)),
        total_epochs=None,
    )
    record = {
        "timestamp": _now_ts(),
        "name": checkpoint_name,
        "type": checkpoint_info.get("type"),
        "epoch": checkpoint_info.get("epoch"),
        "timesteps": checkpoint_info.get("timesteps"),
    }
    manifest["last_checkpoint"] = record
    manifest["checkpoint_history"].append(record)
    manifest["checkpoint_history"] = _trim_list(manifest["checkpoint_history"])
    epoch_val = checkpoint_info.get("epoch")
    if isinstance(epoch_val, int):
        manifest["completed_epochs"] = max(manifest.get("completed_epochs", 0), epoch_val)
    _save_manifest(manifest_path, manifest)


def record_metrics_update(manifest_path: str, epoch: int, phases: List[str]) -> None:
    if not manifest_path:
        return
    manifest = ensure_manifest(
        manifest_path,
        run_id=os.path.basename(os.path.dirname(manifest_path)),
        total_epochs=None,
    )
    entry = {
        "timestamp": _now_ts(),
        "epoch": int(epoch),
        "phases": phases,
    }
    manifest["last_metrics_epoch"] = int(epoch)
    manifest["metrics_history"].append(entry)
    manifest["metrics_history"] = _trim_list(manifest["metrics_history"])
    completed = manifest.get("completed_epochs", 0)
    if isinstance(epoch, int):
        manifest["completed_epochs"] = max(completed, epoch)
    _save_manifest(manifest_path, manifest)


def print_manifest_summary(manifest_path: str, heading: Optional[str] = None) -> None:
    if not manifest_path:
        return
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return
    if heading:
        print(f"\n{'='*100}", flush=True)
        print(f"{heading:^100}", flush=True)
        print(f"{'='*100}", flush=True)
    run_id = manifest.get("run_id", os.path.basename(os.path.dirname(manifest_path)))
    total_epochs = manifest.get("total_epochs", "?")
    completed = manifest.get("completed_epochs", 0)
    last_metrics = manifest.get("last_metrics_epoch")
    last_checkpoint = manifest.get("last_checkpoint")
    last_resume = manifest.get("last_resume")
    print(f"[RUN] ID: {run_id} | Total epochs: {total_epochs} | Completed: {completed}", flush=True)
    if last_metrics is not None:
        print(f"[RUN] Last metrics epoch: {last_metrics}", flush=True)
    if last_checkpoint:
        print(f"[RUN] Last checkpoint: {last_checkpoint.get('name')} (epoch {last_checkpoint.get('epoch')}, type={last_checkpoint.get('type')})", flush=True)
    if last_resume:
        print(f"[RUN] Last resume: epoch {last_resume.get('epoch')} via {last_resume.get('checkpoint_type')} ({last_resume.get('reason')})", flush=True)
    print(f"{'-'*100}", flush=True)
