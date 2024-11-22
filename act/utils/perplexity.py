# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import typing as t
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences: t.List[str], num_sentences: int = 20000):
        self.sentences = sentences
        if len(self.sentences) > num_sentences:
            self.sentences = self.sentences[:num_sentences]

    @staticmethod
    def dataset_names(cache_dir: Path = None) -> t.Dict[str, Path]:
        return {
            "wikipedia": os.environ.get("HF_HUB_CACHE", cache_dir)
            / "wikipedia_sentences.csv",
            # "identity_hate": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "insult": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "obscene": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "severe_toxic": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "threat": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "toxic": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
            # "negative-set": Path("/tmp/toxicity_concepts/jigsaw/en.csv"),
        }

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


@torch.no_grad()
def perplexity_batch(
    sentences: t.List[str],
    prompts: t.Optional[t.List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: t.Optional[int] = 128,
    max_generation_length: t.Optional[int] = 50,
    autoregressive: bool = False,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
        autoregressive: If True, use autoregressive decoding, otherwise use parallel decoding with causal masking.
    Returns:
        Perplexity per sentence in the batch
    """
    if autoregressive:
        return _autoregressive_perplexity_batch(
            sentences=sentences,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
        )
    else:
        return _parallel_perplexity_batch(
            sentences=sentences,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
        )


@torch.no_grad()
def _autoregressive_perplexity_batch(
    sentences: t.List[str],
    prompts: t.Optional[t.List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: t.Optional[int] = 128,
    max_generation_length: t.Optional[int] = 50,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
        autoregressive: If True, use autoregressive decoding, otherwise use causal masking.
    Returns:
        Perplexity per sentence in the batch
    """
    truncation = max_context_length is not None
    # Since we are going to concatenate the prompt at the left, add padding to the right
    tokenizer.padding_side = "right"
    # First tokenize sentence tokens since we will need them anyways
    tok_s = tokenizer(
        text=sentences,
        return_tensors="pt",
        truncation=truncation,
        padding=truncation,
        max_length=max_generation_length,
        add_special_tokens=(
            prompts is None
        ),  # if there is a prompt, it already contains BOS token
    ).to(device)
    tokenizer.padding_side = (
        "left"  # go back to original padding (to not messup things)
    )

    if prompts is not None:
        # Now we tokenize the prompts
        side = tokenizer.truncation_side
        # The sentence is the direct continuation of the prompt, so we truncate the prompt by the left
        tokenizer.truncation_side = "left"
        tok_p = tokenizer(
            text=prompts,
            return_tensors="pt",
            truncation=truncation,
            padding=True,
            add_special_tokens=True,
            max_length=max_context_length,
        ).to(device)
        tokenizer.truncation_side = side
        # Concatenate prompt tokens with sentence tokens.
        # This is the only way to know exactly at which token the prompt ends and the sentence starts.
        # (Note that tokenizer(prompt+continuation) != cat([tokenizer(prompt), tokenizer(continuation)]))
        tok_all = {k: torch.cat([tok_p[k], tok_s[k]], -1) for k in tok_p.keys()}
        # This tells us where prompts end and sentences start, so we can slice them later on.
        offset = tok_p["input_ids"].shape[-1]
    else:
        tok_all = tok_s
        offset = 1  # skips the BOS token

    input_ids = tok_all["input_ids"]
    attention_mask = tok_all["attention_mask"]
    # This is the number of tokens in each continuation. We will generate this amount of tokens, one by one.
    attention_mask_sum = tok_s["attention_mask"].sum(-1)
    # Buffer to keep track of ppls
    ppls = torch.zeros(attention_mask.shape[0], device=device, dtype=torch.float32)
    totals = torch.zeros_like(ppls)
    # Now we iterate a cursor over all continuations at the same time (they all start at the same position and end up at different positions, marked by attention_mask_sum)
    for ctx_len in range(1, attention_mask_sum.max() - 1):
        # We can stop computing perplexity for those continuations that have reached an end.
        mask = ctx_len < attention_mask_sum
        # Pick all tokens since prompt beginning to prompt + current cursor position
        _input_ids = input_ids[mask][:, : (offset + ctx_len)]
        _attention_mask = attention_mask[mask][:, : (offset + ctx_len)]
        logits = model(input_ids=_input_ids, attention_mask=_attention_mask).logits
        # Compute perplexity for last token (note that indexing at offset + ctx_len gives us the token id right after :(offset + ctx_len))
        loss = torch.nn.functional.cross_entropy(
            logits[:, -1],
            input_ids[mask][:, (offset + ctx_len)].reshape(-1),
            reduction="none",
        )
        ppls[mask] += loss
        totals[mask] += 1

    return torch.exp(ppls / totals)


@torch.no_grad()
def _parallel_perplexity_batch(
    sentences: t.List[str],
    prompts: t.Optional[t.List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: t.Optional[int] = 128,
    max_generation_length: t.Optional[int] = 50,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
    Returns:
        Perplexity per sentence in the batch
    """
    truncation = max_context_length is not None
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if prompts is not None:
        text = [p + s for p, s in zip(prompts, sentences)]
    else:
        text = sentences
    tok_all = tokenizer(
        text=text,
        return_tensors="pt",
        truncation=truncation,
        padding=True,
        add_special_tokens=True,
        max_length=max_generation_length if prompts is None else max_context_length,
    ).to(device)
    tokenizer.padding_size = padding_side
    logits = model(
        input_ids=tok_all["input_ids"], attention_mask=tok_all["attention_mask"]
    ).logits
    # Compute perplexity for last token (note that indexing at offset + ctx_len gives us the token id right after :(offset + ctx_len))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tok_all["input_ids"][:, 1:].reshape(-1),
        reduction="none",
    )
    loss = (tok_all["attention_mask"][:, 1:] * loss.view(logits.shape[0], -1)).sum(
        -1
    ) / tok_all["attention_mask"][:, 1:].sum(-1)

    return torch.exp(loss)


def perplexity_sequential(
    sentences: t.List[str],
    prompts: t.List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.

    Inspired by https://colab.research.google.com/drive/1X2ZfC4y8Jx8FbkR7m-bLi8Ifrq-8MPTO#scrollTo=BehgQO-Nbvj0&line=30&uniqifier=1

    Args:
        sentences: A sequence of sentences
        prompts: A sequence of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
    Returns:
        Perplexity per sentence in the batch
    """
    model.eval()
    ppls = []
    for s, p in zip(sentences, prompts):
        if p is not None:
            tok_p = tokenizer(p, return_tensors="pt")
            len_p = tok_p["input_ids"].shape[1]
        else:
            len_p = 0
            p = ""

        tok = tokenizer(p + s, return_tensors="pt")

        # Build attention mask to not attend to prompt
        with torch.no_grad():
            outputs = model(**tok)

        # Make tuple of scores
        logits_cont = outputs.logits[
            :, len_p - 1 : -1, :
        ]  # shifting by one, since last token is a "future" token
        scores_cont = logits_cont.to(torch.float64).unbind(dim=1)

        # Compute transition log-likelihoods
        lls = model.compute_transition_scores(
            sequences=tok["input_ids"][:, len_p:],
            scores=scores_cont,
            normalize_logits=True,
        )
        ppl_cont = torch.exp(-torch.mean(lls)).item()
        ppls.append(ppl_cont)
    return torch.tensor(ppls, device=device)


def measure_perplexity(
    continuations: t.Union[torch.utils.data.DataLoader, t.List[str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: t.Optional[t.Union[torch.utils.data.DataLoader, t.List[str]]] = None,
    device: str = None,
    batch_size: t.Optional[int] = 128,
    autoregressive: bool = False,
) -> np.ndarray:
    ppl = []

    if prompts is not None:
        if isinstance(prompts, list):
            prompts = torch.utils.data.DataLoader(
                dataset=prompts,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # no preprocessing happening here
            )

    if isinstance(continuations, list):
        continuations = torch.utils.data.DataLoader(
            dataset=continuations,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # no preprocessing happening here
        )

    if prompts is not None:
        for c, p in tqdm(zip(continuations, prompts)):
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=p,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)
    else:
        for c in tqdm(continuations):
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=None,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)

    ppl = torch.cat(ppl).detach().cpu().numpy()
    return ppl
