# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Dict, Any
import cv2
import tyro
import os
from tqdm.auto import tqdm
import numpy as np
import io
import torch
from PIL import Image

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        bg_mask = image[..., 3:4] == 0
        return image, bg_mask


def load_carvekit_bkgd_removal_model(checkpoint_dir: str, device: str = "cuda"):
    if checkpoint_dir is not None:
        import carvekit.ml.files as cmf
        from pathlib import Path
        cmf.checkpoints_dir = Path(checkpoint_dir)
    return BackgroundRemoval(device=device)


def remove_background(image, mask_predictor):
    # predict mask
    rgba, mask = mask_predictor(image)  # [H, W, 4]

    # remove salt&pepper noise from mask (a common artifact of this method)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.astype(bool)[..., None]

    # white background
    rgb = rgba[..., :3] * (1 - mask) + 255 * mask
    rgb = rgb.astype(np.uint8)

    return rgb, mask


def segment_and_save_carvekit(image_path, mask_path, masked_image_path, mask_predictor):
    # read image
    with open(image_path, "rb") as f:
        array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    # convert image
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb, mask = remove_background(image, mask_predictor)

    # convert mask to uint8
    mask = (1 - mask[..., 0].astype(np.uint8)) * 255

    # save rgb image
    with open(masked_image_path, "wb") as f:
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not is_success:
            print("Could not write", mask_path)
        else:
            io_buf = io.BytesIO(buffer)
            f.write(io_buf.getbuffer())

    # save mask
    with open(mask_path, "wb") as f:
        is_success, buffer = cv2.imencode(".png", mask)
        if not is_success:
            print("Could not write", mask_path)
        else:
            io_buf = io.BytesIO(buffer)
            f.write(io_buf.getbuffer())


def main(run_folder: str, carvekit_checkpoint_dir: str, runs_offset: int = 0, max_n_runs: int = -1):
    # get all runs
    runs = os.listdir(run_folder)
    runs = [os.path.join(run_folder, f, "images") for f in runs]
    runs = [f for f in runs if os.path.isdir(f)]

    # filter runs
    runs = runs[runs_offset:runs_offset+max_n_runs]
    print("mask these runs", runs)

    # load carvekit model
    carvekit_model = load_carvekit_bkgd_removal_model(carvekit_checkpoint_dir)

    for input_image_folder in tqdm(runs, desc="Create masked images"):
        # setup output dir for masks
        output_mask_folder = os.path.join(input_image_folder, "masks")
        os.makedirs(output_mask_folder, exist_ok=True)

        # setup output dir for images
        output_image_folder = os.path.join(input_image_folder, "masked_images")
        os.makedirs(output_image_folder, exist_ok=True)

        # segment each image
        images = [f for f in os.listdir(input_image_folder) if ".png" in f and "pred_file" in f]
        for image in tqdm(images, desc="mask image", leave=True):
            segment_and_save_carvekit(
                image_path=os.path.join(input_image_folder, image),
                mask_path=os.path.join(output_mask_folder, image),
                masked_image_path=os.path.join(output_image_folder, image),
                mask_predictor=carvekit_model
            )


if __name__ == '__main__':
    tyro.cli(main)
