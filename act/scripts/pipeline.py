# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
This script defines a pipeline for training and evaluating generative models with interventions.
It handles loading configurations, managing responses, and learning interventions on specified modules.

The pipeline leverages Hydra for configuration management and provides both "atonce" and "incr" modes for intervention learning. 
"atonce" mode learns interventions on all modules simultaneously, while "incr" mode allows for incremental learning, leveraging previously learned interventions when training on subsequent modules.

"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from act.evaluations import (
    calculate_clip_score,
    evaluate_0shot,
    evaluate_eleuther,
    evaluate_perplexity,
    evaluate_toxicity,
)
from act.scripts import generate_with_hooks, generate_with_hooks_diffusion
from act.scripts.learn_intervention import InterventionsManager, learn_intervention
from act.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _rtp(cfg: DictConfig):
    intervention_state_path = InterventionsManager.get_output_path(cfg.interventions)
    cfg.rtp.intervention_params.state_path = intervention_state_path
    if cfg.rtp.fast:
        cfg.rtp.ppl_sentences = 3
        cfg.rtp.rtp_sentences = 3

    evaluate_toxicity.measure_toxicity(cfg.rtp)


def _text_generation(cfg: DictConfig):
    intervention_state_path = InterventionsManager.get_output_path(cfg.interventions)
    cfg.text_generation.intervention_params.state_path = intervention_state_path
    if cfg.text_generation.fast:
        cfg.text_generation.new_seq_len = 10
        cfg.text_generation.num_sentences = 3
        cfg.text_generation.max_strength = 1
        cfg.text_generation.strength_sample_size = 2

    generate_with_hooks.generate(cfg.text_generation)


def _text_to_image_generation(cfg: DictConfig):
    intervention_state_path = InterventionsManager.get_output_path(cfg.interventions)
    cfg.text_to_image_generation.intervention_params.state_path = (
        intervention_state_path
    )

    generate_with_hooks_diffusion.generate(cfg.text_to_image_generation)


def _zero_shot(cfg: DictConfig):
    evaluate_0shot.evaluate(cfg.zero_shot)


def _mmlu(cfg: DictConfig):
    intervention_state_path = InterventionsManager.get_output_path(cfg.interventions)
    cfg.mmlu.intervention_params.state_path = intervention_state_path
    if cfg.mmlu.fast:
        cfg.mmlu.limit = 10
        cfg.mmlu.bootstrap_iters = 2
    evaluate_eleuther.run_eleuther_eval(cfg.mmlu)


def _model_perplexity(cfg: DictConfig):
    if cfg.model_perplexity.fast:
        cfg.model_perplexity.perplexity_model_path = "EleutherAI/pythia-70m"
    evaluate_perplexity.evaluate(cfg.model_perplexity)


def _clip_score(cfg: DictConfig):
    calculate_clip_score.calculate_clip_score(cfg.clip_score)


EVAL_REGISTRY = {
    "rtp": _rtp,
    "text-generation": _text_generation,
    "zero_shot": _zero_shot,
    "mmlu": _mmlu,
    "model_perplexity": _model_perplexity,
    "text-to-image-generation": _text_to_image_generation,
    "clip_score": _clip_score,
}


@hydra.main(config_path="../configs", config_name="text_generation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)

    wandb_run = utils.setup_wandb(cfg)

    # Learn intervention first, which includes computing responses.
    learn_intervention(cfg)

    # Now evaluate the intervention.
    for eval in cfg.evaluation:
        logger.info(f"Running {eval}")
        EVAL_REGISTRY[eval](cfg)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
