from __future__ import annotations

from pathlib import Path
from typing import Any


class MlflowTracker:
    def __init__(
        self,
        enabled: bool,
        tracking_uri: str | None,
        experiment_name: str | None,
        run_name: str | None,
    ) -> None:
        self.enabled = enabled
        self._mlflow = None
        self._active = False
        if not enabled:
            return
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            print(f"[mlflow] disabled (import failed): {exc}")
            self.enabled = False
            return
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        self._active = True
        print(
            f"[mlflow] enabled uri={mlflow.get_tracking_uri()} "
            f"experiment={experiment_name or 'Default'} run_name={run_name}"
        )

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        safe = {}
        for k, v in params.items():
            if v is None:
                continue
            safe[str(k)] = str(v) if isinstance(v, (list, dict, tuple, set)) else v
        self._mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        if step is None:
            self._mlflow.log_metrics(metrics)
        else:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        p = Path(path)
        if p.exists():
            self._mlflow.log_artifact(str(p))

    def log_artifacts(self, dir_path: str | Path) -> None:
        if not self._active:
            return
        p = Path(dir_path)
        if p.exists():
            self._mlflow.log_artifacts(str(p))

    def set_tag(self, key: str, value: Any) -> None:
        if not self._active:
            return
        self._mlflow.set_tag(str(key), str(value))

    def end(self, status: str = "FINISHED") -> None:
        if not self._active:
            return
        self._mlflow.end_run(status=status)
        self._active = False

