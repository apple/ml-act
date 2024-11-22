# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import pickle
import typing as t
from pathlib import Path

import hydra
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from omegaconf import DictConfig, OmegaConf

# Local imports
from act.models import get_model
from act.models.model_with_hooks import ModelWithHooks
from act.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_results(results: t.Dict, output_path: str) -> None:
    """
    Saves results dictionary as pickle named after args used for evaluation run in cfg.output_dir (which is created if it does not exist yet)
    Args:
        results: nested dictionary containing the results to be saved
        args: args used for evaluation run

    Returns: None

    """
    out_file = Path(output_path) / "eleuther.pkl"
    logger.info(f"Saving eleuther eval results to {out_file}")
    with out_file.open("wb") as fp:
        pickle.dump(results, fp)


def evaluate(cfg: DictConfig) -> dict:
    """

    Args:
        args: the argument namespace from the argparser

    Returns: a results object which is the standard output of the lm-eval-harness

    """
    module, tokenizer = get_model(
        cfg.model_params.model_path,
        cfg.cache_dir,
        cfg.dtype,
        cfg.device,
        model_task="text-generation",
        seq_len=128,
    )

    assert cfg.model_params.module_names is not None, logging.error(
        f"Intervention specified as {cfg.intervention_params.name}, but no module names passed (passed {cfg.model_params.module_names})"
    )

    # Create hooked model
    model_with_hooks = ModelWithHooks(
        module=module,
    )
    model_with_hooks.load_hooks_from_folder(
        folder=Path(cfg.intervention_params.state_path),
        module_names=cfg.model_params.module_names,
        hook_type=cfg.intervention_params.name,
        **cfg.intervention_params.hook_params,
    )
    model_with_hooks.register_hooks()
    model = model_with_hooks.module

    # Convert into an "evaluable" HF model
    lm = HFLM(model, tokenizer=tokenizer)

    # Run evaluation
    results = evaluator.simple_evaluate(
        lm,
        tasks=list(cfg.tasks),
        num_fewshot=cfg.num_fewshot,
        limit=cfg.limit,  # can set a limit for quicker testing
        bootstrap_iters=cfg.bootstrap_iters,  # for statistical significance estimation
        random_seed=cfg.rs,
        numpy_random_seed=cfg.nrs,
        torch_random_seed=cfg.trs,
        device=cfg.device,
        batch_size=cfg.batch_size,
        cache_requests=True,
    )

    return results


def main(cfg: DictConfig) -> None:
    run_eleuther_eval(cfg)


def run_eleuther_eval(cfg: DictConfig) -> None:
    # Run actual evaluations
    results = evaluate(cfg)

    # Save complete results dict as pickle
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir) / output_path
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    save_results(results, output_path)
    if cfg.wandb.mode != "disabled":
        overall_results: t.Dict = results["results"]["mmlu"]
        overall_results = {"mmlu-" + k: v for k, v in overall_results.items()}
        utils.log_wandb(overall_results)

    # Pop created samples for simplified output printing in console and separate logging in wandb
    # TODO: Save samples?
    samples = results.pop("samples")

    # Console printing of summarized results
    logger.info(
        f"{cfg.model_params.model_path}, limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}"
    )
    logger.info(make_table(results))
    if "groups" in results:
        logger.info(make_table(results, "groups"))


@hydra.main(
    config_path="../act/configs", config_name="text_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    run_eleuther_eval(cfg)


if __name__ == "__main__":
    main()
