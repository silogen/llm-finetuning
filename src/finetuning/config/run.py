from textwrap import dedent
from typing import Dict, Literal, Optional

import torch
from accelerate import DistributedType, PartialState
from pydantic import ConfigDict, Field, field_serializer, field_validator

from finetuning.config.base import BaseConfig


class ModelArguments(BaseConfig):
    """These are passed to AutoModelForCausalLM.from_pretrained

    See parameter docstrings and help at:
    https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    See below in "Parameters for big model inference" too, it affects training too. Also note that this link takes you
    to the transformers main branch version - be sure to compare with the installed version of transformers (that keeps
    changing over time, and it is difficult to keep this doctstring up to date, so we wanted to link to the latest here).

    Some important parameters to consider are:

    device_map :
        A map that specifies where each submodule should go. It doesn’t need to be refined to each parameter/buffer
        name, once a given module name is inside, every submodule of it will be sent to the same device. If we only pass
        the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which the model will be allocated,
        the device map will map the entire model to this device. Passing device_map = 0 means put the whole model on GPU
        0.
    attn_implementation :
        The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of
        the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using
        Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is
        otherwise the manual "eager" implementation.

    NOTE:
        This does not include quantization_config. Quantization config is specified separately.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The datatype to use for model parameters
    torch_dtype: Literal["auto"] | torch.dtype = "auto"

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def _str_to_dtype(cls, x: str) -> Literal["auto"] | torch.dtype:
        """Validator for converting string to proper torch.dtype, while also handling 'auto'"""
        if x == "auto":
            return x
        else:
            return getattr(torch, x)

    @field_serializer("torch_dtype", when_used="json")
    @classmethod
    def _dtype_to_str(cls, x: str | torch.dtype) -> str:
        """Serializer for converting torch.dtype to string, while also handling 'auto'"""
        if str(x) == "auto":
            return x
        else:
            return str(x)[len("torch.") :]  # Remove the "torch." prefix

    # Custom device map so that you can manually override the choices that HuggingFace would make.
    # This can also be a string to specify "auto", "balanced_low_0", or "sequential"
    device_map: Dict[str, int | str] | str | None = None
    max_memory: Optional[Dict[str, str]] = None
    low_cpu_mem_usage: bool = False
    # Note: this can be set to "sdpa", "flash_attention_2", "eager"
    attn_implementation: Optional[str] = None
    offload_folder: Optional[str] = None
    offload_state_dict: Optional[bool] = None  # Default is True if offloading (otherwise no effect)
    offload_buffers: Optional[bool] = None

    # Saves generated hidden states to speed up generation
    # see: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
    # use_cache is mutually exclusive with gradient_checkpointing
    use_cache: bool = True

    # HF HUB arguments:
    cache_dir: Optional[str] = None
    force_download: bool = False
    local_files_only: bool = False
    proxies: Optional[Dict[str, str]] = None
    resume_download: bool = False
    revision: str = "main"
    code_revision: str = "main"
    subfolder: Optional[str] = None
    token: Optional[str] = None
    use_safetensors: Optional[bool] = None
    variant: Optional[str] = None
    # Warning: if set to True, allows execution of downloaded remote code
    trust_remote_code: bool = False

    def model_post_init(self, __context):
        accelerate_state = PartialState()

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
        return self.model_dump(exclude_unset=True)


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
        Normally should be set to 'auto' to continue if a checkpoint exists.
        Can set to True to always try to continue, False to never try, or a path to load from a specific path."""
        ),
    )
    final_checkpoint_name: str = Field(
        default="checkpoint-final", description="Name of final checkpoint. Should be left as default"
    )
    determinism: Literal["no", "half", "full"] = Field(
        default = "no",
        description = dedent("""\
            Set the level of determinism in implementations. Deterministic implementations are not always available,
            and when they are, they are usually slower than their non-deterministic counterparts. Recommended for
            debugging only.
            'no': No determinism.
            'half': Prefer deterministic implementations. 
            'full': Only fully deterministic implementations, error out on operations that only have non-deterministic 
                    implementations.""")
    )

    def model_post_init(self, __context):
        # Assign the model name/path for loading the tokenizer if not explicitly supplied
        if self.tokenizer is None:
            self.tokenizer = self.model
