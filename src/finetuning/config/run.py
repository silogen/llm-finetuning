import os
from textwrap import dedent
from typing import Dict, Literal, Optional

import torch
from accelerate import DistributedType, PartialState
from pydantic import ConfigDict, Field, field_serializer, field_validator, model_validator

from finetuning.config.base import BaseConfig


class ModelArguments(BaseConfig):
    """These are passed to AutoModelForCausalLM.from_pretrained

    See parameter docstrings and help at:
    https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    See below in "Parameters for big model inference" too, it affects training too. Also note that this link takes you
    to the transformers main branch version - be sure to compare with the installed version of transformers (that keeps
    changing over time, and it is difficult to keep this docstring up to date, so we wanted to link to the latest here).

    Some important parameters to consider are:

    - device_map :
        A map that specifies where each submodule should go. It doesn’t need to be refined to each parameter/buffer
        name, once a given module name is inside, every submodule of it will be sent to the same device. If we only pass
        the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which the model will be allocated,
        the device map will map the entire model to this device. Passing device_map = 0 means put the whole model on GPU
        0.
    - attn_implementation :
        The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of
        the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using
        Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is
        otherwise the manual "eager" implementation.

    NOTE:
        This does not include quantization_config. Quantization config is specified separately.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    silogen_extra_args: Dict[str, object] = Field(
        default_factory=dict,
        description="Don't specify directly - this gathers additional args passed to the model",
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def handle_silogen_extra_args(cls, values):
        """This gathers any additional args passed to the model that are not explicitly defined in the config, and puts them in silogen_extra_args. This is useful for passing on HF-specific args that we don't want to explicitly define in our config."""
        if "silogen_extra_args" in values:
            raise ValueError(
                "silogen_extra_args should not be passed directly, it is reserved for gathering extra args passed to the model. Please remove it from your config."
            )
        known_keys = set(cls.model_fields.keys())
        silogen_extra_args = {k: v for k, v in values.items() if k not in known_keys}
        values["silogen_extra_args"] = silogen_extra_args
        return values

    # The datatype to use for model parameters
    dtype: Literal["auto"] | str | torch.dtype = "auto"

    @classmethod
    def _str_to_dtype(cls, x: str) -> Literal["auto"] | torch.dtype:
        """Validator for converting string to proper torch.dtype, while also handling 'auto'"""
        if x == "auto":
            return x
        elif isinstance(x, str):
            return getattr(torch, x)
        elif isinstance(x, torch.dtype):
            return x
        else:
            raise ValueError(f"Invalid dtype value: {x}")

    @field_validator("dtype", mode="before")
    @classmethod
    def _str_to_dtype_validator(cls, x: str) -> Literal["auto"] | torch.dtype:
        return cls._str_to_dtype(x)

    @field_serializer("dtype", when_used="json")
    @classmethod
    def _dtype_to_str(cls, x: str | torch.dtype) -> str:
        """Serializer for converting torch.dtype to string, while also handling 'auto'"""
        if str(x) == "auto":
            return x
        else:
            return str(x)[len("torch.") :]  # Remove the "torch." prefix

    pretrained_model_name_or_path: str | os.PathLike | None = Field(
        default=None,
        description=dedent(
            """\
            Can be either:
            - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
            - A path to a *directory* containing model weights saved using `~PreTrainedModel.save_pretrained`.
            - A path or url to a *tensorflow index checkpoint file*.
            - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format.
            - `None` if you are both providing the configuration and state dictionary."""
        ),
    )
    config: Optional[str | os.PathLike] = Field(
        default=None,
        description=dedent(
            """\
            Configuration for the model to use instead of an automatically loaded configuration.
            Can be either an instance of a class derived from `PretrainedConfig`, or a string/path valid as input to `PretrainedConfig.from_pretrained`."""
        ),
    )
    cache_dir: Optional[str | os.PathLike] = Field(
        default=None,
        description="Path to a directory in which a downloaded pretrained model configuration should be cached.",
    )
    from_tf: bool = Field(
        default=False,
        description="Load the model weights from a TensorFlow checkpoint save file.",
    )
    from_flax: bool = Field(
        default=False,
        description="Load the model weights from a Flax checkpoint save file.",
    )
    ignore_mismatched_sizes: bool = Field(
        default=False,
        description="Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model.",
    )
    force_download: bool = Field(
        default=False,
        description="Whether or not to force the (re-)download of the model weights and configuration files.",
    )
    proxies: Optional[Dict[str, str]] = Field(
        default=None,
        description="A dictionary of proxy servers to use by protocol or endpoint.",
    )
    output_loading_info: bool = Field(
        default=False,
        description="Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.",
    )
    local_files_only: bool = Field(
        default=False,
        description="Whether or not to only look at local files (i.e., do not try to download the model).",
    )
    token: str | bool | None = Field(
        default=None,
        description="The token to use as HTTP bearer authorization for remote files.",
    )
    revision: str = Field(
        default="main",
        description="The specific model version to use. It can be a branch name, a tag name, or a commit id.",
    )
    attn_implementation: Optional[str] = Field(
        default=None,
        description=dedent(
            """\
            The attention implementation to use in the model. Can be any of 'eager', 'sdpa', 'flash_attention_2', or 'flash_attention_3'.
            Accepts HF kernel references in the form: <namespace>/<repo_name>[@<revision>][:<kernel_name>]"""
        ),
    )
    device_map: str | Dict[str, int | str | torch.device] | int | torch.device | None = Field(
        default=None,
        description="A map that specifies where each submodule should go.",
    )
    max_memory: Optional[Dict] = Field(
        default=None,
        description="A dictionary device identifier to maximum memory if using `device_map`.",
    )
    tp_plan: Optional[str] = Field(
        default=None,
        description="A torch tensor parallel plan. Currently only accepts 'auto'.",
    )
    tp_size: Optional[str] = Field(
        default=None,
        description="A torch tensor parallel degree. If not provided would default to world size.",
    )
    offload_folder: str | os.PathLike | None = Field(
        default=None,
        description="If the `device_map` contains any value 'disk', the folder where we will offload weights.",
    )
    offload_buffers: bool = Field(
        default=False,
        description="Whether or not to offload the buffers with the model parameters.",
    )
    subfolder: str = Field(
        default="",
        description="In case the relevant files are located inside a subfolder of the model repo on huggingface.co.",
    )
    variant: Optional[str] = Field(
        default=None,
        description="If specified load weights from `variant` filename, e.g. pytorch_model.<variant>.bin.",
    )
    use_safetensors: Optional[bool] = Field(
        default=None,
        description="Whether or not to use `safetensors` checkpoints.",
    )
    weights_only: bool = Field(
        default=True,
        description="Indicates whether unpickler should be restricted to loading only tensors and primitive types.",
    )
    key_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers architecture, but was not converted accordingly.",
    )

    def model_post_init(self, __context):
        accelerate_state = PartialState()

        # Handle legacy torch_dtype key for backwards compatibility:
        if "torch_dtype" in self.silogen_extra_args:
            # First check if dtype was also specified:

            self.dtype = self._str_to_dtype(self.silogen_extra_args.pop("torch_dtype"))

        # Deepspeed sets the device_map internally so device_map is automatically set only when Deepspeed is not active.
        if accelerate_state.distributed_type != DistributedType.DEEPSPEED:
            if self.device_map is None:  # If not provided, infer from the environment
                if accelerate_state.distributed_type != DistributedType.NO and accelerate_state.num_processes > 1:
                    # It's a multi-process setup where we want each training process to use a single GPU.
                    self.device_map = {"": accelerate_state.local_process_index}
                else:
                    self.device_map = "auto"

    def get_model_load_kwargs(self):
        """Returns a dictionary that is ready to be passed to AutoModelForCausalLM.from_pretrained as **kwargs"""
        return {**self.model_dump(exclude_unset=True), **self.silogen_extra_args}


class RunConfig(BaseConfig):
    """Experiment running configuration"""

    model: str = Field(
        default="/local_resources/basemodel",
        description="Local path to model to be fine-tuned. Normally this should be /local_resources/basemodel",
    )
    model_args: ModelArguments = ModelArguments()
    tokenizer: None | str = Field(
        default=None,
        description=dedent("Model HuggingFace ID, or path, or None to use the one associated with the model"),
    )
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Use the Fast version of the tokenizer. The 'slow' version may be compatible with more features.",
    )
    resume_from_checkpoint: bool | str = Field(
        default=False,
        description=dedent(
            """\
        Normally should be set to 'auto' to continue if a checkpoint exists.\
        Can set to True to always try to continue, False to never try, or a path to load from a specific path."""
        ),
    )
    final_checkpoint_name: str = Field(
        default="checkpoint-final", description="Name of final checkpoint. Should be left as default"
    )
    determinism: Literal["no", "half", "full"] = Field(
        default="no",
        description=dedent(
            """\
            Set the level of determinism in implementations. Deterministic implementations are not always available,\
            and when they are, they are usually slower than their non-deterministic counterparts. Recommended for\
            debugging only.\
            'no': No determinism.\
            'half': Prefer deterministic implementations.\
            'full': Only fully deterministic implementations, error out on operations that only have non-deterministic\
                    implementations."""
        ),
    )

    def model_post_init(self, __context):
        # Assign the model name/path for loading the tokenizer if not explicitly supplied
        if self.tokenizer is None:
            self.tokenizer = self.model
