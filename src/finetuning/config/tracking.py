from typing import Optional

from pydantic import Field

from finetuning.config.base import BaseConfig


class TrackingConfig(BaseConfig):
    """Config for tracking experiment results in MLFlow"""

    mlflow_server_uri: str = Field(description="MLflow server URI. Can be local path.")
    experiment_name: str = Field(description="Experiment name that is used for MLFlow tracking.")
