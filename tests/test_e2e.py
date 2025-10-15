"""This file contains tests that run the full E2E setups and train for a few updates

NOTE: These tests should move to our E2E testing framework once we have the necessary things in place (workload engine,
etc.) - see this ticket: https://silogen.atlassian.net/browse/SDX-344

TIPS:
- Inject something like this in the tests at the start to inspect the intermediate results:
    import pathlib
    tmpdir = pathlib.Path("sft_tests")
    tmpdir.mkdir()
- Note: To test DPO, we first run SFT
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import finetuning.cli
from finetuning.data.tokenizer import subsetup_tokenizer

sftdata = [
    {
        "dataset": "test-1",
        "id": "test-1-1",
        "messages": [
            {"role": "system", "content": "A B C"},
            {"role": "user", "content": "A B C"},
            {"role": "assistant", "content": "A B C"},
        ],
    },
]

dpodata = [
    {
        "dataset": "test-1",
        "id": "test-1-1",
        "prompt_messages": [
            {"role": "system", "content": "A B C"},
            {"role": "user", "content": "A B C"},
        ],
        "chosen_messages": [
            {"role": "assistant", "content": "A B C"},
        ],
        "rejected_messages": [
            {"role": "assistant", "content": "D E F"},
        ],
    },
]


def test_sft_gpt_lora(tmpdir):
    # 0. Config definition:
    config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "LORA"
  task_type: "CAUSAL_LM"
  peft_kwargs:
    r: 4
    lora_alpha: 4
    lora_dropout: 0.1
    target_modules:
      - c_attn
    fan_in_fan_out: true
run_conf:
  model: hf-internal-testing/tiny-random-gpt2
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: 91c0fe31d692dd8448d9bc06e8d1877345009e3b
  resume_from_checkpoint: False
  determinism: "full"
"""
    num_steps = 15
    # 1. setup
    # Set paths
    train_data_path = tmpdir / "data.json"
    ckpt_dir = tmpdir / "outputs"
    config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    config_filled_in = config.format(
        output_dir=str(ckpt_dir), train_data_path=str(train_data_path), num_steps=str(num_steps)
    )
    with open(config_path, "w") as fo:
        fo.write(config_filled_in)

    # 2. Run
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(config_path)])

    # 3. Inspect results!
    with open(ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should still be around -log(1/1002)~6.9, (1002 is the number of units)
    assert np.isclose(trainer_state["log_history"][0]["loss"], 6.9, atol=0.5)
    # In current setup, loss goes at least below 3.0 by 15 steps
    assert trainer_state["log_history"][-1]["loss"] < 3.0

    # LoRA setup should lead to adapter checkpoint
    assert (ckpt_dir / "checkpoint-final" / "adapter_model.safetensors").isfile()
    # And since the basemodel vocab was modified, there is a new basemodel checkpoint too
    assert (ckpt_dir / "checkpoint-new-basemodel" / "model.safetensors").isfile()


def test_sft_gpt(tmpdir):
    # 0. Config definition:
    config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: "PRECOMPUTE_WEIGHTED_MIX"
    datasets:
      - path: {train_data_path}
  validation_data:
    type: "CONCATENATION"
    datasets:
      - path: {train_data_path}
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 1.0
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "steps"
  eval_steps: 10
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: hf-internal-testing/tiny-random-gpt2
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: 91c0fe31d692dd8448d9bc06e8d1877345009e3b
  resume_from_checkpoint: False
  determinism: "half"
"""
    num_steps = 15
    # 1. setup
    # Set paths

    train_data_path = tmpdir / "data.json"
    ckpt_dir = tmpdir / "outputs"
    config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    config_filled_in = config.format(
        output_dir=str(ckpt_dir), train_data_path=str(train_data_path), num_steps=str(num_steps)
    )
    with open(config_path, "w") as fo:
        fo.write(config_filled_in)

    # 2. Run
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(config_path)])

    # 3. Inspect results!
    with open(ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())

    # At the start, loss should still be around -log(1/1002)~6.9, (1002 is the number of units)
    #  see: http://karpathy.github.io/2019/04/25/recipe/  -> verify loss @ init
    assert np.isclose(trainer_state["log_history"][0]["loss"], 6.9, atol=0.5)
    # In current setup, loss goes at least below 4.0 by 15 steps
    assert trainer_state["log_history"][-1]["loss"] < 4.0


def test_sft_gemma2_lora(tmpdir):
    # 0. Config definition:
    config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "LORA"
  task_type: "CAUSAL_LM"
  peft_kwargs:
    r: 4
    lora_alpha: 4
    lora_dropout: 0.1
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
run_conf:
  model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: de7c11b6c25d26ddd1bf4324fcf479b61d18e440
  resume_from_checkpoint: False
"""
    num_steps = 15
    # 1. setup
    # Set paths
    train_data_path = tmpdir / "data.json"
    ckpt_dir = tmpdir / "outputs"
    config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    config_filled_in = config.format(
        output_dir=str(ckpt_dir), train_data_path=str(train_data_path), num_steps=str(num_steps)
    )
    with open(config_path, "w") as fo:
        fo.write(config_filled_in)

    # 2. Run
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(config_path)])

    # 3. Inspect results!
    with open(ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should still be around -log(1/256000)~12.4529327234617, (Even the tiny Gemma2 has 256000 units after Chat-ML)
    assert np.isclose(trainer_state["log_history"][0]["loss"], 12.45, atol=0.5)
    # In current setup, loss goes at least below 5.0 by 15 steps
    assert trainer_state["log_history"][-1]["loss"] < 5.0
    # LoRA setup should lead to adapter checkpoint
    assert (ckpt_dir / "checkpoint-final" / "adapter_model.safetensors").isfile()
    # And since the basemodel vocab was modified, there is a new basemodel checkpoint too
    assert (ckpt_dir / "checkpoint-new-basemodel" / "model.safetensors").isfile()


def intercept_tokenizer_call(*args, overwrite_chat_template=False, **kwargs):
    return subsetup_tokenizer(*args, overwrite_chat_template=True, **kwargs)


@patch("finetuning.dpo.subsetup_tokenizer", intercept_tokenizer_call)
def test_dpo_gemma2(tmpdir):
    # 0. DPO Config definition:
    dpo_config = """\
data_conf:
  chat_template_name: "mistral-with-system"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
  remove_unused_columns: false
  beta: 0.01
  loss_type: "sigmoid"
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: de7c11b6c25d26ddd1bf4324fcf479b61d18e440
  resume_from_checkpoint: False
"""
    num_steps = 15
    # 1. setup
    # Set paths
    dpotmpdir = tmpdir / "dpo"
    dpotmpdir.mkdir()
    dpo_train_data_path = dpotmpdir / "data.json"
    dpo_ckpt_dir = dpotmpdir / "outputs"
    dpo_config_path = dpotmpdir / "config.yaml"
    # Save config and data to disk
    with open(dpo_train_data_path, "w") as fo:
        fo.write(json.dumps(dpodata))
    dpo_config_filled_in = dpo_config.format(
        output_dir=str(dpo_ckpt_dir),
        train_data_path=str(dpo_train_data_path),
        num_steps=str(num_steps),
    )
    with open(dpo_config_path, "w") as fo:
        fo.write(dpo_config_filled_in)

    # 2. Run DPO
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "dpo", str(dpo_config_path)])

    # 3. Inspect DPO results!
    with open(dpo_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # The margins start close to zero
    assert np.isclose(trainer_state["log_history"][0]["rewards/margins"], 0.0, atol=0.03)
    # But they rise to >0.07
    assert trainer_state["log_history"][-1]["rewards/margins"] > 0.07


def test_sft_and_dpo_gemma2(tmpdir):
    # 0. Config definition:
    sft_config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: de7c11b6c25d26ddd1bf4324fcf479b61d18e440
  resume_from_checkpoint: False
"""
    num_steps = 15
    # 1. setup
    # Set paths
    sft_train_data_path = tmpdir / "data.json"
    sft_ckpt_dir = tmpdir / "outputs"
    sft_config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(sft_train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    sft_config_filled_in = sft_config.format(
        output_dir=str(sft_ckpt_dir), train_data_path=str(sft_train_data_path), num_steps=str(num_steps)
    )
    with open(sft_config_path, "w") as fo:
        fo.write(sft_config_filled_in)

    # 2. Run SFT
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(sft_config_path)])

    # # 3. Inspect SFT results!
    with open(sft_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should still be around -log(1/256000)~12.4529327234617, (Even the tiny Gemma2 has 256000 units after Chat-ML)
    assert np.isclose(trainer_state["log_history"][0]["loss"], 12.45, atol=0.5)
    # In current setup, loss goes at least below 4.5 by 15 steps
    assert trainer_state["log_history"][-1]["loss"] < 4.5

    # ############################# DPO STARTS HERE ################################ #
    # 4. DPO Config definition:
    dpo_config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
  remove_unused_columns: false
  beta: 0.01
  loss_type: "sigmoid"
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: {sft_checkpoint}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
"""
    # note: potential to redefine
    num_steps = 15
    # 5. setup
    # Set paths
    dpotmpdir = tmpdir / "dpo"
    dpotmpdir.mkdir()
    dpo_train_data_path = dpotmpdir / "data.json"
    dpo_ckpt_dir = dpotmpdir / "outputs"
    dpo_config_path = dpotmpdir / "config.yaml"
    # Save config and data to disk
    with open(dpo_train_data_path, "w") as fo:
        fo.write(json.dumps(dpodata))
    dpo_config_filled_in = dpo_config.format(
        output_dir=str(dpo_ckpt_dir),
        train_data_path=str(dpo_train_data_path),
        num_steps=str(num_steps),
        sft_checkpoint=str(sft_ckpt_dir / "checkpoint-final"),
    )
    with open(dpo_config_path, "w") as fo:
        fo.write(dpo_config_filled_in)

    # 6. Run DPO
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "dpo", str(dpo_config_path)])

    # 7. Inspect DPO results!
    with open(dpo_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # The margins start close to zero
    assert np.isclose(trainer_state["log_history"][0]["rewards/margins"], 0.0, atol=0.03)
    # But they rise to >0.07
    assert trainer_state["log_history"][-1]["rewards/margins"] > 0.07


def test_sft_and_dpo_lora_gemma2(tmpdir):
    # 0. Config definition:
    sft_config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_first_step: true
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
  model_args:
    attn_implementation: "eager"
    use_cache: False
    revision: de7c11b6c25d26ddd1bf4324fcf479b61d18e440
  resume_from_checkpoint: False
"""
    num_steps = 15
    # 1. setup
    # Set paths
    sft_train_data_path = tmpdir / "data.json"
    sft_ckpt_dir = tmpdir / "outputs"
    sft_config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(sft_train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    sft_config_filled_in = sft_config.format(
        output_dir=str(sft_ckpt_dir), train_data_path=str(sft_train_data_path), num_steps=str(num_steps)
    )
    with open(sft_config_path, "w") as fo:
        fo.write(sft_config_filled_in)

    # 2. Run SFT
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(sft_config_path)])

    # 3. Inspect SFT results!
    with open(sft_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should still be around -log(1/256000)~12.4529327234617, (Even the tiny Gemma2 has 256000 units after Chat-ML)
    assert np.isclose(trainer_state["log_history"][0]["loss"], 12.45, atol=0.5)
    # In current setup, loss goes at least below 4.5 by 15 steps
    assert trainer_state["log_history"][-1]["loss"] < 4.5

    # ############################# DPO STARTS HERE ################################ #
    # 4. DPO Config definition:
    dpo_config = """\
data_conf:
  chat_template_name: "chat-ml"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_first_step: true
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
  remove_unused_columns: false
  beta: 0.1
  loss_type: "sigmoid"
peft_conf:
  peft_type: "LORA"
  task_type: "CAUSAL_LM"
  peft_kwargs:
    r: 4
    lora_alpha: 4
    lora_dropout: 0.0
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
run_conf:
  model: {sft_checkpoint}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
"""
    # note: potential to redefine
    num_steps = 20
    # 5. setup
    # Set paths
    dpotmpdir = tmpdir / "dpo"
    dpotmpdir.mkdir()
    dpo_train_data_path = dpotmpdir / "data.json"
    dpo_ckpt_dir = dpotmpdir / "outputs"
    dpo_config_path = dpotmpdir / "config.yaml"
    # Save config and data to disk
    with open(dpo_train_data_path, "w") as fo:
        fo.write(json.dumps(dpodata))
    dpo_config_filled_in = dpo_config.format(
        output_dir=str(dpo_ckpt_dir),
        train_data_path=str(dpo_train_data_path),
        num_steps=str(num_steps),
        sft_checkpoint=str(sft_ckpt_dir / "checkpoint-final"),
    )
    with open(dpo_config_path, "w") as fo:
        fo.write(dpo_config_filled_in)

    # 6. Run DPO
    finetuning.cli.finetuning_main_cli(argv=["--num-preprocess-workers", "1", "dpo", str(dpo_config_path)])

    # 7. Inspect DPO results!
    with open(dpo_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # The margins start close to zero
    assert np.isclose(trainer_state["log_history"][0]["rewards/margins"], 0.0, atol=0.02)
    # But they rise to >0.03
    assert trainer_state["log_history"][-1]["rewards/margins"] > 0.03

    # LoRA setup should lead to adapter checkpoint
    assert (dpo_ckpt_dir / "checkpoint-final" / "adapter_model.safetensors").isfile()


def test_generate(tmpdir):
    # 0. Config definition:
    generate_config = """\
model_conf:
  model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
  model_args:
    attn_implementation: "eager"
    use_cache: False
    device_map: cpu
    revision: de7c11b6c25d26ddd1bf4324fcf479b61d18e440
prompt_conf:
  type: "open-input"
  input: "Hi my name is "
hf_gen_params:
  do_sample: false
  max_new_tokens: 4
"""
    with open(tmpdir / "gen_conf.yaml", "w") as fo:
        fo.write(generate_config)
    finetuning.cli.generation_cli([str(tmpdir / "gen_conf.yaml"), str(tmpdir / "generated_output.yaml")])
    with open(tmpdir / "generated_output.yaml") as fin:
        list_of_outputs = yaml.safe_load(fin)
    assert list_of_outputs[0]["answer"] == "amientosamientosérômeérôme"  # This is what tiny-random-Gemma2 replies.
