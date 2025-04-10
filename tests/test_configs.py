import peft
import pytest
import yaml
from pydantic import BaseModel, ValidationError

from finetuning.config import DPOExperimentConfig, GenericPeftConfig, SFTExperimentConfig, SilogenTrainingArguments
from finetuning.config.base import BaseConfig, DisallowedInput


def test_validation():
    class TestConfig(BaseConfig):
        foo: str

    # Valid:
    assert TestConfig(**{"foo": "A string"}).foo == "A string"
    # Wrong key specified:
    with pytest.raises(ValidationError):
        TestConfig(**{"bar": "A string"})
    # Empty input:
    with pytest.raises(ValidationError):
        TestConfig()
    # Additional input:
    with pytest.raises(ValidationError):
        TestConfig(**{"foo": "A string", "bar": "Another string"})

    class NestedConfig(BaseConfig):
        test: TestConfig
        foo: str

    # Valid:
    assert NestedConfig(**{"test": {"foo": "A string"}, "foo": "A string"}).foo == "A string"
    # Nested structure is also enforced:
    with pytest.raises(ValidationError):
        NestedConfig(**{"test": "A string", "foo": "A string"})

    class DefaultConfig(BaseConfig):
        foo: str = "A string"

    # Default works:
    assert DefaultConfig().foo == "A string"
    # Can be changed:
    assert DefaultConfig(**{"foo": "Another string"}).foo == "Another string"


def test_yaml_instantiation(tmp_path):
    class TestConfig(BaseConfig):
        foo: str
        bar: int = 1

    class NestedConfig(BaseConfig):
        test: TestConfig
        fizz: int

    # Valid YAML:
    yamlstr = """
    test:
        foo: "A string"
    fizz: 1
    """
    assert NestedConfig.from_yaml(yamlstr).test.foo == "A string"
    assert NestedConfig.from_yaml(yamlstr).fizz == 1
    # Invalid YAML:
    yamlstr = """
    test:
        foo: "A string"
        buzz: 3
    fizz: 1
    """
    with pytest.raises(ValidationError):
        NestedConfig.from_yaml(yamlstr)

    # Back and forth:
    yamlstr = """
    test:
        foo: "A string"
    fizz: 1
    """
    assert NestedConfig.from_yaml(yaml.dump(NestedConfig.from_yaml(yamlstr).model_dump())).fizz == 1


def test_disallowed_input():
    class TestModel(BaseModel):
        foo: str
        bar: DisallowedInput = "Forbidden Fruit"

    # If not specified, all good:
    assert TestModel(foo="hi").foo == "hi"
    assert TestModel(foo="hi").bar == "Forbidden Fruit"

    # But User Input is not allowed
    with pytest.raises(ValidationError):
        TestModel(foo="hi", bar="world")

    # Even if it matches the default:
    with pytest.raises(ValidationError):
        TestModel(foo="hi", bar="Forbidden Fruit")


def test_hf_training_args_config():
    args = SilogenTrainingArguments(output_dir="Dummy dir")

    # We can use the
    args = SilogenTrainingArguments(output_dir="Dummy dir")
    print(args.model_dump())

    # Back and forth (note: since per_device_train_batch_size is disallowed, we must use exclude_unset=True):
    args = SilogenTrainingArguments(output_dir="Dummy dir")
    recreated = SilogenTrainingArguments(**(args.model_dump(exclude_unset=True)))
    assert recreated.output_dir == "Dummy dir"
    assert recreated.model_dump() == args.model_dump()


def test_peft_config():
    config = GenericPeftConfig(peft_type="LORA", task_type="CAUSAL_LM")
    assert isinstance(config.get_peft_config(), peft.LoraConfig)
    config = GenericPeftConfig(
        peft_type="LORA", task_type="CAUSAL_LM", peft_kwargs={"r": 32, "target_modules": ["v_proj"]}
    )
    assert isinstance(config.get_peft_config(), peft.LoraConfig)


def test_full_sft_config():
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      chat_template_name: "chat-ml"
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: true
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      overwrite_output_dir: true
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
    sft_args:
      max_seq_length: 2048
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    tracking:
      mlflow_server_uri: "file:///mlruns"
      experiment_name: default
      hf_mlflow_log_artifacts: "False"
    """
    # This just tests the loading - any failure will raise a validation error
    config = SFTExperimentConfig.from_yaml(yamlstr)


def test_full_config_resolve_training_arguments():
    # This test is for the special resolve_training_args() call that is done when ready to start an experiment
    # with a full experiment config.
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      chat_template_name: "chat-ml"
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: false
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      overwrite_output_dir: true
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
    sft_args:
      max_seq_length: 2048
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    tracking:
      mlflow_server_uri: "file:///mlruns"
      experiment_name: default
      hf_mlflow_log_artifacts: "False"
    """
    config = SFTExperimentConfig.from_yaml(yamlstr)
    assert config.training_args.per_device_train_batch_size == -1
    config.resolve_training_args()
    assert config.training_args.per_device_train_batch_size == 4
    assert config.training_args.gradient_accumulation_steps == 128
    # We can find the HuggingFace extra keys, too:
    config.training_args.distributed_state


def test_dpo_config():
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      chat_template_name: "chat-ml"
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: true
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      overwrite_output_dir: true
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
      beta: 0.01
      loss_type: "sigmoid"
      max_prompt_length: 123
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    """
    # This just tests the loading - any failure will raise a validation error
    config = DPOExperimentConfig.from_yaml(yamlstr)
