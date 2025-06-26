from typing import Optional

from pydantic import Field

from finetuning.config.base import BaseConfig


class TrackingConfig(BaseConfig):
    """Config for tracking experiment results in MLFlow"""

    mlflow_server_uri: str = Field(description="MLflow server URI. Can be local path.")
    experiment_name: str = Field(description="Experiment name that is used for MLFlow tracking.")
    run_id: Optional[str] = Field(default=None, description="Run id, to resume logging to previously started run.")
    run_name: Optional[str] = Field(
        default=None,
        description="Run name, to give meaningful name to the run to be displayed in MLFlow UI. Used only when run_id is unspecified.",
    )
