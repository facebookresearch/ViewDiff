# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json
import tyro
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from ...io_util import torch_to_pil
from .co3d_dataset import CO3DConfig, CO3DDataset


def load_blip2_model(pretrained_model_name_or_path: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
    processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    model = model.to(device)

    return model, processor


def save_captions_file(captions, output_file: str, intermediate: bool = False):
    date_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S.%f")
    output_file_parts = output_file.split(".")
    output_file_without_suffix = ".".join(output_file_parts[:-1])
    output_file_without_suffix += f"_{date_time}"
    if intermediate:
        output_file_without_suffix += f"_intermediate"
    output_file_with_time = f"{output_file_without_suffix}.{output_file_parts[-1]}"
    with open(output_file_with_time, "w") as f:
        json.dump(captions, f, indent=4)


@torch.no_grad()
def generate_blip2_captions(
    dataset_config: CO3DConfig,
    pretrained_model_name_or_path: str = "Salesforce/blip2-opt-2.7b",
    device: str = "cuda",
    batch_size: int = 4,
    output_file: str = "co3d_blip2_captions.json",
):
    # load blip2 model
    model, processor = load_blip2_model(pretrained_model_name_or_path, device)

    # make sure the important fields are set correctly
    dataset_config.dataset_args.load_point_clouds = False
    dataset_config.batch.load_recentered = False
    dataset_config.batch.need_mask_augmentations = False
    dataset_config.batch.n_parallel_images = 1
    dataset_config.dataset_args.n_frames_per_sequence = 5

    # can make it square here already s.t. collate works - processor makes it square anyways
    dataset_config.batch.crop = "resize"
    dataset_config.batch.image_height = 512
    dataset_config.batch.image_width = 512

    # Get the dataset: parse CO3Dv2
    dataset = CO3DDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # loop over data
    captions = {}
    for idx, batch in enumerate(tqdm(dataloader, desc="Generate Captions")):
        # get sequence, category from batch
        sequences = [os.path.basename(x) for x in batch["root"]]
        categories = [os.path.basename(os.path.dirname(x)) for x in batch["root"]]

        # get image from batch
        images = batch["images"]  # (batch_size, K=1, C, H, W)
        images = images.squeeze()  # (batch_size, C, H, W)
        images = [torch_to_pil(x) for x in images]  # processor expects PIL images

        # run captioning
        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [s.strip() for s in generated_text]

        # save captions
        for c, s, p in zip(categories, sequences, generated_text):
            if c not in captions:
                captions[c] = {}
            if s not in captions[c]:
                captions[c][s] = []
            captions[c][s].append(p)

        # save intermediate outputs in case this crashes at some point
        if idx % 5000 == 0:
            save_captions_file(captions, output_file, intermediate=True)

    # save final file
    save_captions_file(captions, output_file)


if __name__ == "__main__":
    tyro.cli(generate_blip2_captions)

    # jun
    
    # In file demo.py
    # from dataclasses import dataclass
    # import tyro

    # @dataclass
    # class demo:
    #     property1: str = "default string 1"
    #     preperty2: str = "default string 2"

    # def democlass(config:demo):
    #     print(config.property1)

    # tyro.cli(democlass)
    
    # In the terminal
    # python demo.py --config.property1 "new string"
    # or
    # python demo.py --config.property1="new string"
    
    
    # Output:
    # new string