from enum import Enum
from typing import List, Literal

from pydantic import Field

from finetuning.config.base import BaseConfig


class ChatTemplateName(str, Enum):
    """Chat template to use."""

    MISTRAL_WITH_SYSTEM = "mistral-with-system"
    CHAT_ML = "chat-ml"
    PORO = "poro"
    KEEP_ORIGINAL = "keep-original"
    SIMPLIFIED_LLAMA31 = "simplified-llama31"


class DataInputType(str, Enum):
    """
    In PRECOMPUTE_WEIGHTED_MIX, the resulting interleaved dataset is fully precomputed. This means that we wait until
    every single sample in every dataset gets picked. The upside is that the resulting dataset can store and cache
    things like tokenization etc. (no need to continuously run expensive data processsing on-the-fly) and we can use
    things such as group_by_length (which requires random-access and helps to reduce padding). The downside is that the
    resulting dataset can be very large, so any full dataset mapping and filtering operations etc. can take a while.

    In CONCATENATION, the resulting dataset is fully precomputed, but this is fast because each dataset is just
    concatenated, unlike in the interleaving processes.

    AUTO_SPLIT is used for taking a split of the training data for validation.

    The NONE DataInputType is meant for skipping validation.
    """

    CONCATENATION = "CONCATENATION"
    PRECOMPUTE_WEIGHTED_MIX = "PRECOMPUTE_WEIGHTED_MIX"
    AUTO_SPLIT = "AUTO_SPLIT"
    NONE = "NONE"


class DatasetDefinition(BaseConfig):
    """Define how to load a dataset"""

    path: str = Field(description="Local path to a JSONL file in the finetuning data format")


class WeightedDatasetDefinition(DatasetDefinition):
    """Define a dataset, with a weight for sampling"""

    sampling_weight: float = 1.0


class DataInput(BaseConfig):
    """Base config for data input"""

    type: DataInputType
    data_type: str = Field(
        default="ChatConversation",
        description="Generally, the data_type is automatically set based on the experiment config method.",
    )


class ConcatenationDataInput(DataInput):
    """A simple list of datasets

    These are simply concatenated, the same as sampling all with equal weight.

    The datasets themselves need to be in the finetuning supported JSONL formats.
    For SFT this means lines:

    {"messages": [{"content": "string", "role": "string"}]}

    For DPO this means lines of:
    {
       "prompt_messages": [{"content": "string", "role": "string"}],
       "chosen_messages": [{"content": "string", "role": "string"}],
        "rejected_messages": [{"content": "string", "role": "string"}]
    }
    """

    type: Literal[DataInputType.CONCATENATION]
    datasets: List[DatasetDefinition] = Field(min_length=1)


class WeightedMixDataInput(DataInput):
    """A list of datasets where each is sampled by a certain weight

    These datasets are interleaved based on the sampling weights. The resulting dataset is fully precomputed, upto
    the point where every single sample in every dataset gets picked. This means that with small sampling weights,
    it can take a lot of draws to see every sample from a dataset and so the resulting dataset can be very large.

    The datasets themselves need to be in the finetuning supported JSONL formats.
    For SFT this means lines:

    {"messages": [{"content": "string", "role": "string"}]}

    For DPO this means lines of:
    {
       "prompt_messages": [{"content": "string", "role": "string"}],
       "chosen_messages": [{"content": "string", "role": "string"}],
        "rejected_messages": [{"content": "string", "role": "string"}]
    }
    """

    type: Literal[DataInputType.PRECOMPUTE_WEIGHTED_MIX]
    datasets: List[WeightedDatasetDefinition] = Field(min_length=1)
    seed: int = Field(default=19851243, description="Seed for the random number generator for interleaving draws")


class AutoSplitDataInput(DataInput):
    """Automatic validation split from the training data"""

    type: Literal[DataInputType.AUTO_SPLIT]
    ratio: float = Field(default=0.2, description="Ratio of the training data to use for validation")
    seed: int = Field(default=1289525893, description="Seed for the random number generator for splitting")


class NoneDataInput(DataInput):
    """A special type for not using data e.g. in validation"""

    type: Literal[DataInputType.NONE]


class MissingPadTokenStrategy(str, Enum):
    """Specifies the available missing pad token strategies.

    We've shown in a small set of experiments that repurposing EOS can start to hurt performance
    while the other options seem to work equally well.

    Repurposing EOS is the default in many online sources, but it is actually a bad idea if we want to predict
    EOS, as all the pad_token_ids get ignored in loss computation, and thus the model does not learn to predict
    the end of the text. However, for models that have additional tokens for end of message, end of turn, etc.
    this is not so dangerous.

    Repurposing BOS is similar to repurposing EOS, but since we do not need to predict BOS, this may be more sensible.

    Repurposing UNK can work with tokenizers that never produce UNKs in normal data (e.g. Mistral tokenizers should have
    a byte fall-back so that everything can be tokenized).

    UNK_CONVERT_TO_EOS uses a hack where the unk_token_id is initially used for padding, but in the collation phase the
    input-side UNKs (padding) gets set to EOS, so that the input-side padding looks like EOS. On the output-side, the
    UNKs (padding) still gets ignored. NOTE: This will leave the tokenizer's pad_token_id set to the unk_token_id; so
    any subsequent use of the model where padding is involved should somehow explicitly set the pad_token_id again.
    """

    EOS_REPURPOSE = "eos-repurpose"
    BOS_REPURPOSE = "bos-repurpose"
    UNK_REPURPOSE = "unk-repurpose"
    UNK_CONVERT_TO_EOS = "unk-convert-to-eos"


class ChatTrainValidConfig(BaseConfig):
    """Training time data configuration

    Always defines some DataInput for training data and can include validation DataInput, though a trivial NoneDataInput
    is also allowed for the validation side.

    Additionally includes chat template and padding configurations, as those are part of the data input pipeline.
    """

    training_data: ConcatenationDataInput | WeightedMixDataInput
    validation_data: ConcatenationDataInput | AutoSplitDataInput | NoneDataInput
    chat_template_name: ChatTemplateName = ChatTemplateName.MISTRAL_WITH_SYSTEM
    padding_side: str = Field(default="right", description="Padding side, right is usually right.")
    missing_pad_token_strategy: MissingPadTokenStrategy = Field(
        default=MissingPadTokenStrategy.BOS_REPURPOSE,
        description="See the MissingPadTokenStrategys for descriptions of the options",
    )
