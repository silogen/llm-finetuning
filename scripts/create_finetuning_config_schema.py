#!/usr/bin/env python3
import json
from textwrap import indent

from pydantic import Field

from finetuning.config import DPOExperimentConfig, SFTExperimentConfig
from finetuning.config.base import BaseConfig


# This is used so that we can discriminate between SFT and DPO experiments.
# In requests, the finetuning config is anyway embedded under a specific key, not "flat"
# like it's passed on to the finetuning program itself.
class MasterConfig(BaseConfig):
    finetuning_config: SFTExperimentConfig | DPOExperimentConfig = Field(discriminator="method")


schema = MasterConfig.model_json_schema()

# Make total_train_batch_size optional for schema.json (will be set to num GPUS)
del schema["$defs"]["BatchsizeConfig"]["properties"]["total_train_batch_size"]["type"]
schema["$defs"]["BatchsizeConfig"]["properties"]["total_train_batch_size"]["anyOf"] = [
    {"type": "integer"},
    {"type": "null"},
]

# Remove const from torch_dtype (it's not a constant - this is a surprising pydantic output)
del schema["$defs"]["ModelArguments"]["properties"]["torch_dtype"]["const"]

print("NOTE: only $defs, add these to the schema.json in the appropriate place")
print("---")
for line in indent(json.dumps({"$defs": schema["$defs"]}, indent=2), 2 * " ").splitlines()[1:-2]:
    print(line)
