import json
import os

import pytest
from transformers import AutoTokenizer

from finetuning.config import SFTExperimentConfig
from finetuning.sft import subsetup_sft_training_data


def test_sft_data_truncation(tmp_path):
    # 1. Create a dummy jsonl file with a long example
    data_path = tmp_path / "test_data.jsonl"
    long_text = "word " * 100
    example = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": long_text}]}
    with open(data_path, "w") as f:
        f.write(json.dumps(example) + "\n")

    # 2. Define the config with truncation
    yaml_config = f"""
method: sft
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: "{data_path}"
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 4
  max_per_device_train_batch_size: 1
training_args:
  output_dir: "{tmp_path}/output"
sft_args:
  max_seq_length: 20
  length_handling: truncate
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: "hf-internal-testing/tiny-random-LlamaForCausalLM"
"""
    exp_conf = SFTExperimentConfig.from_yaml(yaml_config)

    # 3. Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    # Need to set chat template for tokenizer if not already present or if we want to ensure ChatML
    # The tiny-random-Llama might not have a chat template.
    # SFT logic usually uses get_chat_template if needed.
    from finetuning.data import get_chat_template

    chat_template = get_chat_template(exp_conf.data_conf.chat_template_name)
    tokenizer.chat_template = chat_template.jinjastr
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Run data setup
    train_dataset = subsetup_sft_training_data(exp_conf, tokenizer)

    # 5. Assertions
    assert len(train_dataset) == 1
    input_ids = train_dataset[0]["input_ids"]
    assert len(input_ids) <= exp_conf.sft_args.max_seq_length
    assert train_dataset[0]["length"] == len(input_ids)
