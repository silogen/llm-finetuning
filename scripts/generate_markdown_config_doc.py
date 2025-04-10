import jsonschema_markdown

import finetuning

# This script generates a markdown document from the pydantic model of the finetuning config,
# by converting it to a json schema and then to markdown.

# The output is used as the basis for the silogen-finetuning-engine documentation in the silogen/ai-workloads repository.
# Some manual editing is done to make it more readable before publishing:
#  - add additional descriptions
#  - fix bad formatting

# NB: some of the naming in the output uses JSON conventions, which might be misleading.
# E.g., "number" and "null" rather than "float" and "None". This should perhaps be modified.
# Also a default value of False for a boolean parameter shows up as empty in the markdown.

schema = finetuning.config.SFTExperimentConfig.model_json_schema()
markdown = jsonschema_markdown.generate(schema, hide_empty_columns=True, footer=False)

# Remove parts related to HuggingFace TrainingArguments and its sub-classes
to_remove = [
    "ChatTemplateName",
    "DebugOption",
    "FSDPOption",
    "HubStrategy",
    "IntervalStrategy",
    "OptimizerNames",
    "SaveStrategy",
    "SchedulerType",
    "SilogenTrainingArguments",
]
sections = markdown.split("\n## ")
markdown = "\n## ".join(section for section in sections if section.split(maxsplit=1)[0] not in to_remove)

# Remove the repeated "Additional properties are not allowed" mentions
markdown = markdown.replace("> ⚠️ Additional properties are not allowed.\n\n", "")

with open("ft_config_doc.md", "w") as f:
    f.write(markdown)
