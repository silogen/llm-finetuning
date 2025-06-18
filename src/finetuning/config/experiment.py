import math
import warnings
from textwrap import dedent
from typing import Literal

from pydantic import Field
from transformers.utils import logging

from finetuning.config.base import BaseConfig
from finetuning.config.data import ChatTemplateName, ChatTrainValidConfig, DataInputType
from finetuning.config.hf_integration import (
    BatchsizeConfig,
    DPOConfig,
    GenericPeftConfig,
    NoPeftConfig,
    PretrainedPeftConfig,
    SFTArguments,
    SilogenDPOConfig,
    SilogenTrainingArguments,
)
from finetuning.config.quantization import BnBQuantizationConfig, NoQuantizationConfig, QuantizationType
from finetuning.config.run import RunConfig
from finetuning.config.tracking import TrackingConfig
from finetuning.data.data_types import data_type_by_method
from finetuning.utils.distributed import is_fsdp
from finetuning.utils.training import resolve_batchsize_and_accumulation

logger = logging.get_logger(__file__)


class FinetuningTrackingConfig(TrackingConfig):
    """Settings that define how run details are logged"""

    hf_mlflow_log_artifacts: str = Field(default="False", description="Whether to store model artifacts in MLFlow.")


class Overrides(BaseConfig):
    """Override options

    These implement dynamic scaling for the learning rate.
    """

    lr_multiplier: float = Field(
        default=1.0, description="Multiplier applied to the learning rate in the training_args"
    )
    lr_batch_size_scaling: Literal["none", "sqrt", "linear"] = Field(
        default="none",
        description=dedent(
            """Scales the learning rate in the training_args by a factor derived from the total training batch size. \
            'none': No scaling. \
            'sqrt': Multiplies learning rate by square root of batch size (a classic scaling rule). \
            'linear': Multiplies learning rate by the batch size (a more modern scaling rule).
            """
        ),
    )


class ExperimentConfig(BaseConfig):
    """A full experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["sft", "dpo"]
    data_conf: ChatTrainValidConfig = Field(description="The data input config")
    training_args: SilogenTrainingArguments = Field(description="Transformer TrainingArguments with some restrictions")
    overrides: Overrides = Field(default=Overrides(), description="Override options to simplify the config interface")
    batchsize_conf: BatchsizeConfig = Field(description="Batch size configuration")
    peft_conf: NoPeftConfig | PretrainedPeftConfig | GenericPeftConfig = Field(description="Adapter configuration")
    run_conf: RunConfig = Field(description="Model related configuration")
    tracking: FinetuningTrackingConfig | None = Field(default=None, description="MLFlow tracking configuration")
    quant_conf: NoQuantizationConfig | BnBQuantizationConfig = Field(
        default=NoQuantizationConfig(), description="Quantization configuration"
    )

    def resolve_training_args(self):
        """Resolves the training args with the batch_size configuration and the HuggingFace post_init"""
        per_device_train_batch_size, gradient_accumulation = resolve_batchsize_and_accumulation(
            self.batchsize_conf.total_train_batch_size, self.batchsize_conf.max_per_device_train_batch_size
        )
        per_device_eval_batch_size = (
            per_device_train_batch_size
            if self.batchsize_conf.per_device_eval_batch_size is None
            else self.batchsize_conf.per_device_eval_batch_size
        )
        self.training_args.per_device_train_batch_size = per_device_train_batch_size
        self.training_args.per_device_eval_batch_size = per_device_eval_batch_size
        self.training_args.gradient_accumulation_steps = gradient_accumulation
        if self.training_args.deepspeed:
            self.training_args.deepspeed.update(
                {
                    "train_micro_batch_size_per_gpu": per_device_train_batch_size,
                    "train_batch_size": self.batchsize_conf.total_train_batch_size,
                }
            )
        self.training_args.resolve_arguments()

    def __apply_overrides(self):
        original_lr = self.training_args.learning_rate
        if self.overrides.lr_multiplier != 1.0:
            MSG = f"Multiplying learning rate by multiplier {self.overrides.lr_multiplier}"
            logger.info(MSG)
            self.training_args.learning_rate *= self.overrides.lr_multiplier
        if self.overrides.lr_batch_size_scaling == "sqrt":
            MSG = "Scaling learning rate with the square root of the batch size"
            logger.info(MSG)
            self.training_args.learning_rate *= math.sqrt(self.batchsize_conf.total_train_batch_size)
        elif self.overrides.lr_batch_size_scaling == "linear":
            MSG = "Scaling learning rate linearly by the batch size"
            logger.info(MSG)
            self.training_args.learning_rate *= self.batchsize_conf.total_train_batch_size
        if self.training_args.learning_rate != original_lr:
            MSG = f"Learning rate changed from {original_lr} to {self.training_args.learning_rate}"
            logger.info(MSG)

    def model_post_init(self, __context):
        self.__apply_overrides()
        if self.training_args.gradient_checkpointing and self.run_conf.model_args.use_cache:
            # These are mutually incompatible
            logger.warning(
                "Setting run_conf.model_args.use_cache=False, because training_args.gradient_checkpointing=True"
            )
            self.run_conf.model_args.use_cache = False

        if (
            self.quant_conf.quantization_type == QuantizationType.BITSANDBYTES
            and is_fsdp
            and self.quant_conf.bnb_4bit_quant_storage is None
        ):
            # This is required for FSDP + QLoRA:
            dtype = "bfloat16" if self.training_args.bf16 else "float16" if self.training_args.fp16 else "float32"
            logger.warning(
                "Because of FSDP + Quantized basemodel, setting quant_conf.bnb_4bit_quant_storage to "
                f"inferred compute_dtype {dtype}."
            )
            self.quant_conf.bnb_4bit_quant_storage = dtype

        data_type_name = data_type_by_method[self.method].__name__
        if self.data_conf.training_data.data_type != data_type_name:
            logger.warning(
                f"Setting data_conf.training_data.data_type to {data_type_name} to match training method {self.method}"
            )
            self.data_conf.training_data.data_type = data_type_name

        if (
            self.data_conf.validation_data.type != DataInputType.NONE
            and self.data_conf.validation_data.data_type != data_type_name
        ):
            logger.warning(
                f"Setting data_conf.validation_data.data_type to {data_type_name} to match training method {self.method}"
            )
            self.data_conf.validation_data.data_type = data_type_name


class SFTExperimentConfig(ExperimentConfig):
    """A full SFT experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["sft"] = "sft"
    sft_args: SFTArguments = Field(description="SFT specific arguments")

    def model_post_init(self, __context):
        super().model_post_init(__context)

        if self.data_conf.chat_template_name != ChatTemplateName.CHAT_ML and self.sft_args.train_on_completions_only:
            warnings.warn(
                "You are training on completions only, but not using the Chat-ML chat template. "
                "This might not work correctly."
            )


class DPOExperimentConfig(ExperimentConfig):
    """A full DPO experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["dpo"] = "dpo"
    # We have to override training_args because TRL's DPOTrainer takes a subclass of TrainingArguments as args:
    # We used to have the regular training_args and a separate dpo_args subconfig in this class, similar to
    # SFTExperimentConfig, which has training_args + sft_args.
    training_args: SilogenDPOConfig = Field(description="TRL DPOTrainerArguments with some restrictions")
