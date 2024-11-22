# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

# Get a list of all directories in /mnt/data
directories = list(Path("/mnt/data").glob("**/config.json"))
results = {}
for dir_name in directories:
    dir_name = str(Path(dir_name).parent)
    model_name = str(Path(dir_name).parts[3]).replace("model--", "").replace("--", "/")
    print(model_name)
    try:
        # Try to load the model using transformers' AutoModel and save its module names
        with init_empty_weights():
            config = AutoConfig.from_pretrained(
                dir_name, torch_dtype=torch.float16, device_map="auto"
            )
            model = AutoModel.from_config(config)

        modules = (
            []
        )  # We use an ordered dict so that the order of the modules is preserved when we convert it to JSON
        for name, _ in model.named_modules():
            modules.append(
                name
            )  # The values are all 'None' because they don't have any significance in this context and I wanted to make them null
        results[model_name] = modules
    except Exception as e:
        print("Error with directory {}:".format(dir_name), str(e))

# Save the results to a JSON file
with open("model_modules.json", "w") as fp:
    json.dump(results, fp, sort_keys=True, indent=2)
