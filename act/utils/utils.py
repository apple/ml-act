# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import logging
import os
import re
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_yaml(path: Path) -> t.Union[t.List, t.Dict]:
    # Adding float resolver that includes "1e-3" like floats. Otherwise they are loaded as strings.
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path, "r") as infile:
        return yaml.load(infile, Loader=loader)


def load_json(path: Path) -> t.Union[t.List, t.Dict]:
    with open(path, "r") as infile:
        return json.load(infile)


def setup_wandb(cfg: DictConfig) -> wandb.apis.public.Run:
    if cfg.wandb.mode == "disabled":
        return None
    import wandb

    if Path(".wandb.yaml").exists():
        wandb_config = load_yaml(".wandb.yaml")
        os.environ["WANDB_API_KEY"] = wandb_config["WANDB_API_KEY"]
        os.environ["WANDB_BASE_URL"] = wandb_config["WANDB_BASE_URL"]
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(
            config=cfg_dict,
            **cfg.wandb,
        )
    else:
        raise FileNotFoundError(
            "Cannot find '.wandb.yaml'. You must set it if you want to use WandB, with content:\n"
            "WANDB_API_KEY: your_api_key\n"
            "WANDB_BASE_URL: your_base_url"
        )
    return run


def seed_all(seed=42):
    """Set all random seeds to a given value for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def is_module_name_in_regex(module_name: str, regex_list: t.List[str]) -> t.List[str]:
    """Returns True if module_name matches any of the regular expressions in the list

        The intended behavior is the following:

        - module_name="foo.bar", regex=".*bar" --> ["foo.bar"]
        - module_name="foo.bar", regex=".*bar:0" --> ["foo.bar:0"]
        - module_name="foo.bar", regex=".*bar:1" --> ["foo.bar:1"]
        - module_name="foo.bar:0", regex=".*bar" --> ["foo.bar:0"]
        - module_name="foo.bar:1", regex=".*bar" --> ["foo.bar:1"]
        - module_name="foo.bar:0", regex=".*bar:0" --> ["foo.bar:0"]
        - module_name="foo.bar:0", regex=".*bar:1" --> []

        The goal is to signal on which tensors we should create hooks.
        If any of the returned tensor names does not really exist, the flow will fail at hook creation/save.
        This might happen specially for the case:
         - module_name="foo.bar", regex=".*bar:1" --> ["foo.bar:1"]

    Args:
        module_name (str): name of pytorch module to find
        regex_list (t.List[str]): list with regex expressions

    Returns:
        list: the list of module names that match the expression
    """

    ret = []
    for regex in regex_list:
        # Just for the weird case that module_name has :num. Unlikely with current torch api.
        # In such case, we match the base part of the modulename if no specific :num is requested through regex.
        module_name_base = module_name
        if re.fullmatch(r".*(:[0-9]+)", module_name) is not None and not ":" in regex:
            module_name_base = module_name.split(":")[0]
        elif re.fullmatch(r".*(:[0-9]+)", module_name) is None and ":" in regex:
            # In case module_name does not contain :num but regex does, remove :num from regex
            regex, regex_num_tensor = regex.split(":")
            module_name = module_name + f":{regex_num_tensor}"
        match = re.fullmatch(regex, module_name_base)
        if match is not None:
            ret.append(module_name)
    ret = list(set(ret))
    return ret


def log_image_folder_wandb(
    folder: Path, limit: int = 100
) -> t.Generator[t.Tuple[np.ndarray, str, str], None, None]:
    """
    Process all png images in a given folder and log them to wandb.

    Args:
        cfg (DictConfig): The configuration object.
        folder (Path): The root folder containing the image files.
        limit (int, optional): Maximum number of images to process from each parent directory. Defaults to 100.

    Returns:
        None
    """
    image_paths_dict = defaultdict(lambda: defaultdict(list))
    image_paths = folder.glob("**/*.png")
    for path in image_paths:
        id = path.stem
        parent = str(path.parent.parent)
        image_paths_dict[parent][id].append(path)
    for parent in image_paths_dict:
        for id in list(sorted(image_paths_dict[parent]))[:limit]:
            images = [
                np.asarray(Image.open(str(p)))
                for p in sorted(image_paths_dict[parent][id])
            ]
            images = np.concatenate(images, axis=1)
            description = str(Path(parent).relative_to(folder))
            images = wandb.Image(images, caption=str(id))
            wandb.log({description: images})


def log_wandb(*args, **kwargs):
    """Log data to Weights & Biases (W&B) platform based on the provided configuration.

    Args:
        cfg (DictConfig): The experiment's configuration object containing logging parameters such as mode and any additional metadata.
        *args: Variable length positional arguments of types allowed for direct logging into W&B. Only dictionary arguments are supported.
        **kwargs: Keyworded variable-length arguments with values to be logged in W&B. If a value is a pandas DataFrame, it will be converted to a wandb.Table and logged as such.

    Returns:
        None if logging has been disabled (cfg['mode'] == "disabled"). Otherwise, the `wandb.Run` object used for logging.

    Raises:
        ValueError: If any argument other than dictionaries is passed in *args or if a value that is not a pandas DataFrame is provided in **kwargs.

    Note:
        - This function requires the `wandb` and `pandas` libraries to be installed.
        - The 'mode' parameter in cfg dictates whether logging should proceed. If 'disabled', this function immediately returns None.
        - If called without any arguments or with only positional dictionaries, each dictionary will be logged as is into W&B using `wandb.log()`.
        - For keyworded arguments where the value is a pandas DataFrame, it converts the dataframe to a wandb.Table and logs it in W&B.
    """
    import wandb

    for arg in args:
        if isinstance(arg, dict):
            wandb.log(arg)
        else:
            raise ValueError(
                f"Only dictionary args are allowed. For dataframes, use kwargs."
            )
    for k, v in kwargs.items():
        if isinstance(v, pd.DataFrame):
            table = wandb.Table(dataframe=v)
            wandb.log({k: table})
        else:
            wandb.log({k: v})
