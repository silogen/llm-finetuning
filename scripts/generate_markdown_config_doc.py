#!/usr/bin/env python3

import argparse

import jsonschema_markdown

import finetuning

parser = argparse.ArgumentParser("Documentation maker")
parser.add_argument(
    "--method", default="sft", choices=["sft", "dpo"], help="Choose the type of documentation to generate"
)
args = parser.parse_args()

# This script generates a markdown document from the pydantic model of the finetuning config,
# by converting it to a json schema and then to markdown.

# The output is used for the silogen-finetuning-engine documentation in the silogen/ai-workloads repository.

# NB: some of the naming in the output uses JSON conventions, which might be misleading.
# E.g., "number" and "null" rather than "float" and "None". This should perhaps be modified.

if args.method == "sft":
    schema = finetuning.config.SFTExperimentConfig.model_json_schema()
else:
    schema = finetuning.config.DPOExperimentConfig.model_json_schema()
markdown = jsonschema_markdown.generate(schema, hide_empty_columns=True, footer=False)

# Replace the introduction with a custom one
intro = """# Finetuning config structure and parameters for {method}

This document describes the structure of the {method} finetuning configuration, and the parameters and values that can be defined there.

See the finetuning config section [this config file]({valid_file}) for an example of a valid configuration.
See the various sub-configs for their options. Additional properties are not allowed.

**Top-level properties:**

""".format(
    method=args.method.upper(),
    valid_file=(
        "overrides/llama-31-tiny-random-deepspeed-values.yaml"
        if args.method == "sft"
        else "overrides/tiny-llama-dpo-full-param.yaml"
    ),
)
markdown = (
    intro + markdown[markdown.index("| Property | Type | Required | Possible values | Default | Description |") :]
)

# Remove parts related to HuggingFace TrainingArguments and its sub-classes
to_remove = [
    "DebugOption",
    "FSDPOption",
    "HubStrategy",
    "IntervalStrategy",
    "OptimizerNames",
    "SaveStrategy",
    "SchedulerType",
    "FDivergenceType",
]
with open("requirements.txt") as f:
    requirements_content = f.read()
    transformers_version = None
    trl_version = None

    for line in requirements_content.splitlines():
        line = line.strip()
        if line.startswith("transformers[tokenizers]=="):
            transformers_version = line.split("==")[1].split("#")[0].strip()
        elif line.startswith("trl=="):
            trl_version = line.split("==")[1].split("#")[0].strip()
    if transformers_version is None:
        raise ValueError("Could not find transformers exact version in requirements.txt")
    if trl_version is None:
        raise ValueError("Could not find trl exact version in requirements.txt")
# Custom section to use for SilogenTrainingArguments and SilogenDPOConfig
silogen_training_args_section = """SilogenTrainingArguments

HuggingFace TrainingArguments as Config with additional SiloGen conventions

The list of training arguments is best available online (the version might not be up-to-date here):
https://huggingface.co/docs/transformers/v{transformers_version}/en/main_classes/trainer#transformers.TrainingArguments

The TrainingArguments object does a lot of things besides specifying the training configuaration options (e.g. it
has computed properties like true training batch size etc.)
""".format(
    transformers_version=transformers_version
)
silogen_dpo_config_section = """SilogenDPOConfig

HuggingFace TRL DPOConfig as Config with additional SiloGen conventions

The list of training arguments is best available online (the version might not be up-to-date here):
https://huggingface.co/docs/transformers/v{transformers_version}/en/main_classes/trainer#transformers.TrainingArguments

Additionally, the DPOConfig has arguments specific to DPO training, which can be found here:
https://huggingface.co/docs/trl/v{trl_version}/en/dpo_trainer#trl.DPOConfig

The object does a lot of things besides specifying the training configuaration options (e.g. it
has computed properties like true training batch size etc.)
""".format(
    transformers_version=transformers_version, trl_version=trl_version
)
sections = markdown.split("\n## ")
sections = [
    silogen_training_args_section if section.startswith("SilogenTrainingArguments") else section for section in sections
]
sections = [silogen_dpo_config_section if section.startswith("SilogenDPOConfig") else section for section in sections]
markdown = "\n## ".join(section for section in sections if section.split(maxsplit=1)[0] not in to_remove)

# Remove the repeated "Additional properties are not allowed" mentions
markdown = markdown.replace("> ⚠️ Additional properties are not allowed.\n\n", "")

# Make sure default "False" is shown for boolean parameters
markdown = markdown.replace("`boolean` |  | boolean |  |", "`boolean` |  | boolean | `False` |")

with open(f"config_doc_{args.method}.md", "w") as f:
    f.write(markdown)
