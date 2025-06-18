import jsonschema_markdown

import finetuning

# This script generates a markdown document from the pydantic model of the finetuning config,
# by converting it to a json schema and then to markdown.

# The output is used for the silogen-finetuning-engine documentation in the silogen/ai-workloads repository.

# NB: some of the naming in the output uses JSON conventions, which might be misleading.
# E.g., "number" and "null" rather than "float" and "None". This should perhaps be modified.

schema = finetuning.config.SFTExperimentConfig.model_json_schema()
markdown = jsonschema_markdown.generate(schema, hide_empty_columns=True, footer=False)

# Replace the introduction with a custom one
intro = """# Finetuning config structure and parameters

This document describes the structure of the finetuning configuration, and the parameters and values that can be defined there.

See the finetuning config section [this config file](overrides/llama-31-tiny-random-deepspeed-values.yaml) for an example of a valid configuration.
See the various sub-configs for their options. Additional properties are not allowed.

**Top-level properties:**

"""
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
]
# Custom section to use for SilogenTrainingArguments
silogen_training_args_section = """SilogenTrainingArguments

HuggingFace TrainingArguments as Config with additional SiloGen conventions

The list of training arguments is best available online (the version might not be up-to-date here):
https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

The TrainingArguments object does a lot of things besides specifying the training configuaration options (e.g. it
has computed properties like true training batch size etc.)
"""
sections = markdown.split("\n## ")
sections = [
    silogen_training_args_section if section.startswith("SilogenTrainingArguments") else section for section in sections
]
markdown = "\n## ".join(section for section in sections if section.split(maxsplit=1)[0] not in to_remove)

# Remove the repeated "Additional properties are not allowed" mentions
markdown = markdown.replace("> ⚠️ Additional properties are not allowed.\n\n", "")

# Make sure default "False" is shown for boolean parameters
markdown = markdown.replace("`boolean` |  | boolean |  |", "`boolean` |  | boolean | `False` |")

with open("ft_config_doc.md", "w") as f:
    f.write(markdown)
