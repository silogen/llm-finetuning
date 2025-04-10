from enum import Enum
from typing import Any, Dict, Literal

import transformers

from finetuning.config.base import BaseConfig
from finetuning.config.data import DatasetDefinition
from finetuning.config.inference import InferenceModelConfig
from finetuning.config.quantization import BnBQuantizationConfig, NoQuantizationConfig


class PromptType(str, Enum):
    OPEN = "open-input"
    EVALDATA = "eval-data"


class PromptConfig(BaseConfig):
    """Base configuration for prompting information (dataset to base on or empty start)"""

    type: PromptType


class OpenPromptConfig(PromptConfig):
    """Generate based on the prompt given in the Config"""

    type: Literal[PromptType.OPEN]
    input: str = ""  # The initial tokens to start with (Note, tokenizer generally adds '<s>'
    num_samples: int = 1  # The number of samples of conversations to produce.


class EvalDataPromptConfig(PromptConfig):
    type: Literal[PromptType.EVALDATA]
    data: DatasetDefinition


class GenerationConfig(BaseConfig):
    """An experiment config for easy generation from checkpoints with HuggingFace tools"""

    model_conf: InferenceModelConfig
    prompt_conf: OpenPromptConfig | EvalDataPromptConfig
    hf_gen_params: Dict[str, Any] = {}
    quant_conf: NoQuantizationConfig | BnBQuantizationConfig = NoQuantizationConfig()
