"""Pure helpers for presenting training jobs."""

from __future__ import annotations

import json


def job_response(row) -> dict:
    progress = json.loads(row["progress_json"]) if row["progress_json"] else {}
    return {
        "id": row["id"],
        "name": row["name"],
        "profile_id": row["profile_id"],
        "dataset_config_id": row["dataset_config_id"],
        "status": row["status"],
        "queue_position": row["queue_position"],
        "current_stage": row["current_stage"],
        "stages": json.loads(row["stages_json"]),
        "progress": {
            "epoch": progress.get("epoch"),
            "total_epochs": progress.get("total_epochs"),
            "step": progress.get("step"),
            "total_steps": progress.get("total_steps"),
            "percent": progress.get("percent"),
        },
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }
