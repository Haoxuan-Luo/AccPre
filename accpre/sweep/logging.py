"""Structured logging for predictor sweeps.

Default: per-run JSONL + a single-line sweep-level CSV/JSONL line.
Optional: Weights & Biases if `wandb` is installed AND `--wandb_project`
is passed; otherwise the wandb helpers are no-ops.

Schema (one dict per logged event):

  {
    "event":   "train_epoch" | "train_final" | "eval_final" | "run_summary",
    "t":       float,        # wall time since run start
    "run_id":  str,
    "family":  str, "dataset": str,
    "lr": float, "epochs": int, "batch_size": int, "seed": int,
    ...metric fields...
  }

The JSONL lives at `<run_dir>/metrics.jsonl` and is appended to.
A single-line `<run_dir>/summary.json` holds the run-final summary.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Lightweight sweep-run logger.

    Writes one JSONL event per call to `log(event, **fields)` and keeps
    a final `summary.json` in the run directory. If wandb is available
    and a project is given, every log call is also forwarded to wandb.
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        base_fields: Dict[str, Any],
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
    ) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.run_id = run_id
        self.base_fields = dict(base_fields)
        self._start = time.time()
        self._jsonl_path = run_dir / "metrics.jsonl"
        # Fresh log every run.
        self._jsonl_path.write_text("")
        self._wandb = None
        if wandb_project:
            try:
                import wandb  # type: ignore
                self._wandb = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name or run_id,
                    config=self.base_fields,
                    dir=str(run_dir),
                    reinit=True,
                )
            except Exception as e:   # noqa: BLE001
                print(f"[logger] wandb disabled (import/init failed: {e})")
                self._wandb = None

    def log(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "t": time.time() - self._start,
            "run_id": self.run_id,
            **self.base_fields,
            **fields,
        }
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self._wandb is not None:
            try:
                # wandb.log does not accept "event" / string fields as
                # metrics; pass numeric fields only, plus a step marker.
                numeric = {
                    k: v for k, v in fields.items()
                    if isinstance(v, (int, float)) and v is not None
                }
                numeric["_event"] = event
                self._wandb.log(numeric)
            except Exception:  # noqa: BLE001
                pass

    def write_summary(self, summary: Dict[str, Any]) -> None:
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
            self._wandb = None


def append_sweep_row(sweep_dir: Path, row: Dict[str, Any]) -> None:
    """Append one run-summary row to the sweep-level JSONL index."""
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
