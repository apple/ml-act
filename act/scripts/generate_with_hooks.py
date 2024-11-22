# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Loads a model and a dataset and extracts intermediate responses
import functools
import logging
import os
import typing as t
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tqdm
from omegaconf import DictConfig
from transformers import pipeline, set_seed

from act.models import get_model
from act.models.model_with_hooks import ModelWithHooks
from act.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# Taken from ActAdd (https://colab.research.google.com/drive/1X2ZfC4y8Jx8FbkR7m-bLi8Ifrq-8MPTO#scrollTo=uDRWo4_xOH3A&line=11&uniqifier=1)
SAMPLING_KWARGS = dict(temperature=1.0, top_p=0.3, repetition_penalty=1.2)

from torch.utils.data import DataLoader, Dataset


# Custom Dataset class
class TextDataset(Dataset):
    """A PyTorch Dataset class for loading text data from a file.

    This class reads text sentences from a given file, cleans them up by stripping whitespace,
    and allows for limiting the number of sentences loaded.

    Attributes:
      file_path (str): The path to the file containing the text sentences.
      max_sentences (int, optional): The maximum number of sentences to load. If None, all sentences are loaded.

    """

    def __init__(self, file_path: str, max_sentences: int = None):
        """Initializes TextDataset.

        Args:
          file_path (str): The path to the text file containing sentences.
          max_sentences (int, optional): The maximum number of sentences to load. If None, all sentences are loaded.

        """
        # Read the file and store sentences
        with open(file_path, "r") as f:
            self.sentences = f.readlines()

        if max_sentences and max_sentences < len(self.sentences):
            self.sentences = self.sentences[:max_sentences]

        # Clean up the sentences by stripping leading/trailing whitespace
        self.sentences = [
            sentence.strip() for sentence in self.sentences if sentence.strip()
        ]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Return the sentence at the given index
        return self.sentences[idx]


# Helper function to create DataLoader
def create_dataloader(
    file_path, batch_size=1, max_sentences=None, shuffle=False, num_workers=0
):
    dataset = TextDataset(file_path, max_sentences=max_sentences)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


def print_generated_sentences(output: t.List[t.Dict[str, str]]) -> None:
    for o in output:
        print("-" * 80)
        print(o[0]["generated_text"])


def generate(cfg: DictConfig) -> None:
    """Generates text using a pretrained language model with optional interventions.

    This function generates text using a pretrained language model specified in the
    configuration (`cfg`). It allows for applying interventions during the generation
    process, controlled by parameters defined in `cfg.intervention_params`. The generated
    text is then logged and optionally saved to a CSV file.

    Args:
        cfg (DictConfig): A Hydra configuration object containing all necessary
            parameters for text generation and intervention. See the
            `configs/text_generation.yaml` example for details.

    Raises:
        ValueError: If `cfg.prompt` ends with ".txt" but the file does not exist.

    Returns:
        None
    """
    if cfg.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif cfg.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)

    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = cfg.results_dir / output_path
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    model, tokenizer = get_model(
        cache_dir=cfg.cache_dir,
        device=cfg.device,
        model_task="text-generation",
        **cfg.model_params,
    )
    # Create hooked model
    model_hooks = ModelWithHooks(
        module=model,
    )

    results = []
    for strength in np.linspace(
        cfg.min_strength, cfg.max_strength, cfg.strength_sample_size
    ):
        model_hooks.remove_hooks()
        hook_params = dict(cfg.intervention_params.hook_params)
        hook_params["strength"] = strength
        model_hooks.load_hooks_from_folder(
            folder=Path(cfg.intervention_params.state_path),
            module_names=cfg.model_params.module_names,
            hook_type=cfg.intervention_params.name,
            **hook_params,
        )

        # Generate without hooks
        generator = pipeline(
            "text-generation",
            model=model_hooks.module,
            tokenizer=tokenizer,
        )

        generate_fn = functools.partial(
            generator,
            max_new_tokens=cfg.new_seq_len,
            do_sample=True,
            **SAMPLING_KWARGS,
        )

        # Register hooks
        model_hooks.register_hooks()

        # Generate with hooks
        set_seed(cfg.seed)
        batch_size = min(cfg.batch_size, cfg.num_sentences)
        if cfg.prompt.endswith(".txt"):
            prompt_loader = create_dataloader(
                cfg.prompt,
                batch_size=batch_size,
                max_sentences=cfg.num_sentences,
            )
        else:
            assert (
                len(cfg.prompt) > 0
            ), "This script does not handle empty prompts for now."
            batch_sizes = (
                [
                    len(batch_indices)
                    for batch_indices in np.array_split(
                        np.arange(cfg.num_sentences), cfg.num_sentences / batch_size
                    )
                ]
                if cfg.num_sentences >= batch_size
                else [
                    cfg.num_sentences,
                ]
            )
            prompt_loader = [[cfg.prompt] * bs for bs in batch_sizes]

        decoded_hook = []
        for prompts in tqdm.tqdm(prompt_loader, desc=f"Generation {strength:0.2f}"):
            gen = generate_fn(prompts, num_return_sequences=1)
            decoded_hook.extend(gen)

        print("\n")
        logger.info("With hook")
        logger.info("=========")
        print_generated_sentences(decoded_hook[:10])
        model_hooks.remove_hooks()

        for d in decoded_hook:
            gen_without_prompt = d[0]["generated_text"].replace(cfg.prompt, "")
            results.append([strength, cfg.prompt, gen_without_prompt])

    if output_path is not None:
        df = pd.DataFrame(data=results, columns=["strength", "prompt", "generation"])
        df.to_csv(output_path / "text_generation.csv")

    if cfg.wandb.mode != "disabled":
        utils.log_wandb(text_generation=df)


@hydra.main(config_path="../configs", config_name="text_generation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
