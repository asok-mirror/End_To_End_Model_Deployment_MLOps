from core import config, utils
from pathlib import Path
import mlflow
import shutil


def promote_model_to_serving(run_id: str = open(Path(config.MODEL_DIR, "run_id.txt")).read()) -> None:
    """Copy the model file from mlflow artifacts to serving

    Args:
        run_id (str, optional): mlflow run id to fetch model. Defaults to open(Path(config.MODEL_DIR, "run_id.txt")).read()
    """
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]

    model_path = Path(artifact_uri, config.MODEL_DIR)
    serving_model_path = Path(config.BASE_DIR, config.MODEL_DIR)

    shutil.copy2(model_path, serving_model_path, follow_symlinks=True)

if __name__ == "__main__":
    promote_model_to_serving()