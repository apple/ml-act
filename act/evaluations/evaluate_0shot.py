# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import functools
import logging
import os
import typing as t
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import tqdm

from act.models import get_model
from act.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"
from omegaconf import DictConfig


def compute_topk_softmax(
    logits: torch.Tensor, protected_tokens: t.List[int], topk: int = 10
) -> t.List[float]:
    """
    Computes the softmax'd values for `protected_tokens` "against" the topk tokens predicted.
    If `protected_tokens` are not in the topk, they are included. So the final set of tokens
    used for softmax can have a size between [topk, topk+len(protected_tokens)].

    :param logits: Logits tensor, of size (num_tokens, vocabulary). Only the first token will be used!
    :param protected_tokens: List of indices to the protected tokens.
    :param topk: Number of top tokens to use for softmax.

    :return:
    """
    device = logits.device
    # Get the indices to the top-l tokens according to logits.
    top_logits_idx = (
        (torch.argsort(logits[0], dim=0, descending=True)[:topk])
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    # Make a set
    top_logits_idx = set(top_logits_idx)
    # Add protected tokens if needed
    top_logits_idx.update(set(protected_tokens))
    # Back to list
    top_logits_idx = list(top_logits_idx)
    # And back to tensor
    top_logits_idx_torch = torch.tensor(top_logits_idx).to(device)
    # Get the values corresponding to the top-k tokens
    top_logits = logits[0, top_logits_idx_torch]
    # Compute softmax
    top_logits_sm = torch.softmax(top_logits, 0)
    # Take the indices in `top_logits_idx` values corresponding to the protected tokens
    answer_idx_in_top = [top_logits_idx.index(pt) for pt in protected_tokens]
    # Get the softmaxed values corresponding to the protected tokens
    logits_answers_sm = [top_logits_sm[idx].item() for idx in answer_idx_in_top]
    return logits_answers_sm


def build_llm_input(
    mode: str,
    system_question: str,
    sentence: str,
    sentence2: t.Optional[str] = None,
    prompt: t.Optional[str] = None,
    prepend: t.Optional[t.List[str]] = None,
) -> t.List[t.Dict]:
    messages = []
    s1 = sentence if prepend is None else f"{prepend[0]} {sentence}"
    s2 = sentence2 or None
    if prepend is not None and s2 is not None:
        s2 = f"{prepend[1]} {s2}"

    if mode == "sentence":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {"role": "user", "content": s1},
        ]
    elif mode == "2sentences":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"{s1}\n{s2}",
            },
        ]
    elif mode == "sentence_prompt":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nContinuation: {s1}",
            },
        ]
    elif mode == "2sentences_prompt":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\n{s1}\n{s2}",
            },
        ]
    return messages


def read_csv(data_path: Path) -> pd.DataFrame:
    # Trying , and ; as delimiters.
    try:
        df = pd.read_csv(data_path, index_col=0)
    except:
        try:
            df = pd.read_csv(data_path, delimiter=";", index_col=0)
        except Exception as exc:
            raise RuntimeError(exc)
    # Hack for user study csvs, remove NaN in the "id" column (there are explanation cells).
    if id in df.columns:
        df = df[~df.id.isna()]
    return df


@torch.inference_mode()
def evaluate(cfg: DictConfig) -> None:
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir) / output_path
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

    model, tokenizer = get_model(
        cfg.model_path,
        cache_dir=cfg.cache_dir,
        dtype=cfg.dtype,
        device=cfg.device,
        model_task="text-generation",
        seq_len=cfg.seq_len,
    )

    model = torch.compile(model)
    print(model)

    answers_token_ids = [tokenizer.encode(a)[0] for a in cfg.system_answers]
    print(f"Answer tokens: {answers_token_ids}")

    # Trying , and ; as delimiters.
    df = read_csv(cfg.data_path)
    df2 = read_csv(cfg.data_path2) if cfg.data_path2 is not None else df

    # Get sentences (and prompts) from data df.
    if cfg.col_sentence2 is None and cfg.col_prompt is None:
        mode = "sentence"
    if cfg.col_sentence2 is None and cfg.col_prompt is not None:
        mode = "sentence_prompt"
    if cfg.col_sentence2 is not None and cfg.col_prompt is None:
        mode = "2sentences"
    if cfg.col_sentence2 is not None and cfg.col_prompt is not None:
        mode = "2sentences_prompt"

    sentences = df[cfg.col_sentence1].to_numpy()
    sentences2, prompts = None, None
    if "prompt" in mode:
        prompts = df[cfg.col_prompt].to_numpy()
    if "2sentences" in mode:
        sentences2 = df2[cfg.col_sentence2].to_numpy()

    # Remove strange characters appearing
    # fixme: Why do we get these <s>?
    sentences = [s.replace("<s>", "") for s in sentences]
    if sentences2 is not None:
        sentences2 = [s.replace("<s>", "") for s in sentences2]

    answers_str = ", ".join(cfg.system_answers)
    system_prompts = [
        (
            sp.strip()
            + f"\nYou can only output one of these answers: ({answers_str}). Do not provide explanations, elaborations, or any other information, just ({answers_str})."
        )
        for sp in cfg.system_prompt
    ]

    terminators = []
    if hasattr(tokenizer, "eos_id"):
        terminators += [
            tokenizer.eos_id,
        ]
    if hasattr(tokenizer, "special_tokens"):
        terminators += [
            tokenizer.special_tokens["<|eot_id|>"],
            tokenizer.special_tokens["<|end_of_text|>"],
        ]
    print(f"Terminators: {terminators}")

    generator = functools.partial(
        model.generate,
        eos_token_id=terminators,
        do_sample=False,
        # temperature=0.6,
        # top_p=0.9,
        # output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        # suppress_tokens=terminators,
    )

    results_per_system_prompt = []
    for sp, system_question in enumerate(system_prompts):
        logger.info(f"SYSTEM_QUESTION:\n{system_question}")

        results = []
        logger.info(f"Running on {len(sentences)} sentences.")
        for i, sentence in enumerate(tqdm.tqdm(sentences, desc="Answering")):
            sentence2 = sentences2[i] if sentences2 is not None else None
            prompt = prompts[i] if prompts is not None else None
            messages = build_llm_input(
                mode=mode,
                system_question=system_question,
                sentence=sentence,
                sentence2=sentence2,
                prompt=prompt,
                prepend=cfg.prepend,
            )
            if i == 0:
                logger.info(mode)
                logger.info(messages)

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(cfg.device)

            # Generate!
            with torch.no_grad():
                outputs = generator(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=1,
                )

            response = outputs["sequences"][0, input_ids.shape[-1] :]
            decoded = tokenizer.decode(response.tolist())
            decoded = decoded.replace("<|eot_id|>", "")

            # outputs["logits"] is a tuple of N tokens being generated.
            # Each element is a [1, vocab] shaped array.
            logits_answers = [
                outputs["logits"][0][0, a_id].item() for a_id in answers_token_ids
            ]

            # Also compute softmax of logits among top K tokens
            topk = 10
            logits_answers_sm = compute_topk_softmax(
                logits=outputs["logits"][0],
                protected_tokens=answers_token_ids,
                topk=topk,
            )

            # Collect results
            results.append(
                [
                    decoded,
                ]
                + logits_answers
                + logits_answers_sm
            )
        q_str = f"q{sp}"
        results_df = pd.DataFrame(
            data=results,
            columns=[  # cfg.columns +
                f"{q_str}_llm_answer",
            ]
            + [f"{q_str}_{sa}" for sa in cfg.system_answers]
            + [f"{q_str}_{a}" + f"_sm_{topk}" for a in cfg.system_answers],
        )

        if len(cfg.system_answers) == 2:
            c0 = results_df[f"{q_str}_{cfg.system_answers[0]}"]
            c1 = results_df[f"{q_str}_{cfg.system_answers[1]}"]
            diff = np.array(c1 > c0).astype(np.int32)
            results_df[f"{q_str}_llm_answer_logit"] = diff

        results_df[f"{q_str}_llm_answer_num"] = results_df[f"{q_str}_llm_answer"].apply(
            lambda x: cfg.system_answers.index(x) if x in cfg.system_answers else 1
        )
        results_per_system_prompt.append(results_df)

    # All system prompts answered
    dfs_out = (
        [
            df,
        ]
        if df2 is None
        else [df, df2]
    )

    results_final = pd.concat(
        dfs_out + results_per_system_prompt,
        axis=1,
    )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        if output_path is not None:
            filename = Path(output_path) / "0shot_eval.csv"
            results_final.to_csv(filename)
            print(f"Saved results in {filename}")
        if cfg.wandb.mode != "disabled":
            utils.log_wandb(evaluate_0shot=results_final)


@hydra.main(
    config_path="../act/configs", config_name="text_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
