# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from act.utils import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

import torch.nn as nn
import torch.nn.functional as F


# https://huggingface.co/docs/diffusers/en/conceptual/evaluation
class DirectionalSimilarity(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        clip_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(
            device
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(
            device
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_id)

        self.device = device

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device),
        }

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(
        self, img_feat_one, img_feat_two, text_feat_one, text_feat_two
    ):
        sim_direction = F.cosine_similarity(
            img_feat_two - img_feat_one, text_feat_two - text_feat_one
        )
        return sim_direction

    @torch.inference_mode()
    def forward(
        self,
        image_one,
        image_two,
        caption_one,
        caption_two,
        caption_zero_shot_one,
        caption_zero_shot_two,
    ):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        text_feat_zero_shot_one = self.encode_text(caption_zero_shot_one)
        text_feat_zero_shot_two = self.encode_text(caption_zero_shot_two)
        text_similarity = (
            F.cosine_similarity(text_feat_one, text_feat_two).detach().cpu().numpy()
        )
        image_similarity = (
            F.cosine_similarity(img_feat_one, img_feat_two).detach().cpu().numpy()
        )
        conditional_similarity = (
            F.cosine_similarity(img_feat_two, text_feat_two).detach().cpu().numpy()
        )
        unconditional_similarity = (
            F.cosine_similarity(img_feat_two, text_feat_one).detach().cpu().numpy()
        )
        directional_similarity = (
            self.compute_directional_similarity(
                img_feat_one, img_feat_two, text_feat_one, text_feat_two
            )
            .detach()
            .cpu()
            .numpy()
        )
        unconditional_zero_shot_similarity = F.cosine_similarity(
            img_feat_two, text_feat_zero_shot_one
        )
        conditional_zero_shot_similarity = F.cosine_similarity(
            img_feat_two, text_feat_zero_shot_two
        )
        zero_shot_score = (
            F.softmax(
                torch.stack(
                    [
                        unconditional_zero_shot_similarity,
                        conditional_zero_shot_similarity,
                    ],
                    dim=1,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        return {
            "text_similarity": text_similarity,
            "image_similarity": image_similarity,
            "conditional_similarity": conditional_similarity,
            "unconditional_similarity": unconditional_similarity,
            "directional_similarity": directional_similarity,
            "conditional_zero_shot_score": zero_shot_score,
        }


def calculate_clip_score(cfg: DictConfig) -> None:
    """
    Main function to calculate CLIP scores for images based on prompts from JSON files or command line arguments.

    This function handles the parsing of command line arguments, reading of image data, and calculation of CLIP scores using zero-shot learning.

    Args:
        args (argparse.Namespace): The parsed command line arguments containing input folder path and prompt field.
    """
    meta_dict = defaultdict(list)
    logger.info(f"Processing directory: {cfg.input_folder}")
    for img_path in sorted(Path(cfg.input_folder).glob("**/*.png")):
        # images += [Image.open(img_path)]
        with (Path(img_path).with_suffix(".json")).open("r") as fp:
            meta = json.load(fp)
            for k, v in meta.items():
                if isinstance(v, list):
                    meta_dict[k].extend(v)
                else:
                    meta_dict[k].append(v)
    df = pd.DataFrame(meta_dict)
    assert len(df) > 0, "No images found in input folder."
    similarity = DirectionalSimilarity(cfg.device)
    results = []
    for id in df["id"].unique():
        df_id = df[df["id"] == id]
        unconditional_image_data = df_id[df_id["strength"] == 0]
        unconditional_image = Image.open(unconditional_image_data["image_path"].iloc[0])
        unconditional_prompt = [unconditional_image_data["original_prompt"].iloc[0]]
        conditional_images = []
        conditional_prompts = []
        conditional_zero_shot_prompt = []
        unconditional_zero_shot_prompt = []
        for idx, row in df_id.iterrows():
            condition = (
                row["src_subsets"]
                if "none" in row["dst_subsets"]
                else row["dst_subsets"]
            )
            conditional_images += [Image.open(row["image_path"])]
            conditional_prompts += [row["conditional_prompt"]]
            conditional_zero_shot_prompt += [
                f"A picture of {condition.replace('_', ' ').replace('-', ' ')}."
            ]
            unconditional_zero_shot_prompt += [f"A picture of something."]
        clip_score = similarity.forward(
            unconditional_image,
            conditional_images,
            unconditional_prompt,
            conditional_prompts,
            unconditional_zero_shot_prompt,
            conditional_zero_shot_prompt,
        )
        for k, v in clip_score.items():
            df_id[k] = v
        results += [df_id]
    results = pd.concat(results)
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir, output_path)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = None
    if output_path is not None:
        results.to_csv(Path(output_path) / "clip_score.csv")
    if cfg.wandb.mode != "disabled":
        utils.log_wandb(clip_score=results)
    return results


@hydra.main(config_path="../act/configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    calculate_clip_score(cfg)


if __name__ == "__main__":
    main()
