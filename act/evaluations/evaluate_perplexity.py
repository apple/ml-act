# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import os
import typing as t
from pathlib import Path
from sys import platform

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from act.models import get_model
from act.utils import utils
from act.utils.perplexity import measure_perplexity

if platform == "darwin":
    # Xavi: MacOS, remove dynamo errors, or test_perplexity will fail.
    # Xavi: This happened for transformers==4.44.2
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True


logging.getLogger().setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


@torch.inference_mode()
def evaluate(cfg: DictConfig) -> t.Dict[str, float]:
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir, output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # Set random seed
    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Setup device and distributed learning
    if cfg.device in ["cuda", None] and torch.cuda.is_available():
        cfg.device = "cuda"
    elif cfg.device == "cuda":
        raise (RuntimeError("Cuda not available"))
    elif cfg.device is None:
        cfg.device = "cpu"

    # Models and Tokenizers
    module, tokenizer = get_model(
        model_path=cfg.perplexity_model_path,
        cache_dir=cfg.data_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        rand_weights=False,
        seq_len=cfg.seq_len,
        model_task="text-generation",
    )
    # module = torch.compile(module)

    # Trying , and ; as delimiters.
    try:
        df = pd.read_csv(cfg.data_path, index_col=0)
    except:
        try:
            df = pd.read_csv(cfg.data_path, delimiter=";", index_col=0)
        except Exception as exc:
            raise RuntimeError(exc)

    sentences = df[cfg.column_sentences[0]].values.tolist()
    sentences = [s.replace("<s> ", "") for s in sentences]
    if len(cfg.column_sentences) > 1:
        prompts = df[cfg.column_sentences[1]].values.tolist()
        prompts = [s.replace("<s> ", "").strip() for s in prompts]
    else:
        prompts = None

    logging.info(
        f"Computing PPL with {cfg.perplexity_model_path} on {len(sentences)} sentences."
    )

    ppl = measure_perplexity(
        continuations=sentences,
        prompts=prompts,
        model=module,
        tokenizer=tokenizer,
        device=cfg.device,
        batch_size=cfg.batch_size,
        autoregressive=(
            cfg.intervention_params.hook_params.intervention_position == "last"
        ),
    )

    # Add PPL column!
    model_name = Path(cfg.perplexity_model_path).name
    col = f"ppl_{model_name}"
    if col in df.columns:
        col = col + "-v2"
    df[col] = ppl
    logging.info(f"Average ppl_{model_name}: {df[col].mean()}")

    logging.info(df)
    if output_path is not None:
        output_file = Path(output_path) / "model_perplexity.csv"
        df.to_csv(output_file)
        logging.info(output_file)
    if cfg.wandb.mode != "disabled":
        utils.log_wandb(model_perplexity=df)

    return df


@hydra.main(
    config_path="../act/configs", config_name="text_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
