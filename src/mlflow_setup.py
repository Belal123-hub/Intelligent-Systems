"""MLflow configuration shared by every experiment in the course.

Storage is an SQLite database `mlflow.db` in the project root, plus an
artifact store `mlartifacts/` next to it. Neither is committed to git
(see .gitignore). This matches Lab 0 §2.5 and Rules §4.4.
"""

from pathlib import Path


def configure_mlflow(project_root: Path) -> None:
    import mlflow

    db_path = project_root / "mlflow.db"
    artifact_dir = project_root / "mlartifacts"
    artifact_dir.mkdir(exist_ok=True)

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("lr0")