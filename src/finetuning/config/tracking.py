from finetuning.config.base import BaseConfig


class TrackingConfig(BaseConfig):
    """Config for tracking experiment results in MLFlow"""

    mlflow_server_uri: str
    "MLflow server URI. Can be local path."
    experiment_name: str
    "Experiment name that is used for MLFlow tracking"
    run_id: str | None = None
    "Run id, to resume logging to previousely started run"
    run_name: str | None = None
    "Run name, to give meaningful name to the run to be displayed in MLFlow UI. Used only when run_id is unspecified."
