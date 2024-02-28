# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Tuple
import os
import random
from tqdm.auto import tqdm
import json
import torch
import tyro
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def load_sd(pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base", device: str = "cuda") -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_prompts(prompt_file: str):
    blip_prompts_dict = {}
    with open(prompt_file, "r") as ff:
        blip_prompts = json.load(ff)
    for category, sequence_dict in blip_prompts.items():
        if category not in blip_prompts_dict:
            blip_prompts_dict[category] = {}
        for sequence, prompts in sequence_dict.items():
            if sequence not in blip_prompts_dict[category]:
                blip_prompts_dict[category][sequence] = []
            blip_prompts_dict[category][sequence].extend(prompts)

    return blip_prompts_dict


def main(
        prompt_file: str,
        output_path: str,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda",
        max_sequences_per_category: int = 300,
        max_prompts_per_sequence: int = 2,
        num_images_per_prompt: int = 2,
        selected_categories: Tuple[str, ...] = ()
):
    # load
    pipe = load_sd(pretrained_model_name_or_path, device)
    prompts = load_prompts(prompt_file)

    # setup output
    os.makedirs(output_path, exist_ok=True)

    # iterate all categories
    for category, sequence_dict in prompts.items():
        # check if category is selected
        if len(selected_categories) > 0 and category not in selected_categories:
            continue

        # setup output dir
        category_out = os.path.join(output_path, category)
        os.makedirs(category_out, exist_ok=True)
        image_to_prompt = {}

        # subsample sequences
        sequences = sequence_dict.keys()
        n_sequences = len(sequences)
        sequences = random.sample(
            sequences,
            k=min(n_sequences, max_sequences_per_category),
        )

        # generate for all remaining sequences
        for s in tqdm(sequences, desc=f"Generate for {category}"):
            # subsample prompts
            prompts = sequence_dict[s]
            n_prompts = len(prompts)
            prompts = random.sample(
                prompts,
                k=min(n_prompts, max_prompts_per_sequence),
            )

            # generate + save next images
            image_counter = 0
            for p in prompts:
                images = pipe(p, num_images_per_prompt=num_images_per_prompt).images
                for img in images:
                    image_name = f"{s}_{image_counter}"
                    out_file = os.path.join(category_out, f"{image_name}.jpg")
                    image_to_prompt[image_name] = p
                    with open(out_file, "wb") as f:
                        img.save(f)
                    image_counter += 1

        # save image_to_prompt file
        image_to_prompt_file = os.path.join(category_out, "image_to_prompt.json")
        with open(image_to_prompt_file, "w") as f:
            json.dump(image_to_prompt, f, indent=4)


if __name__ == '__main__':
    tyro.cli(main)
