# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Literal
import os
import json
import copy
import shutil
from argparse import Namespace
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import tyro
from ...data.create_video_from_image_folder import main as create_video_from_image_folder
from .create_masked_images import load_carvekit_bkgd_removal_model, segment_and_save_carvekit
from .process_nerfstudio_to_sdfstudio import main as process_nerfstudio_to_sdfstudio

import imageio
import shutil


def get_transforms_header():
    return {
        "camera_model": "OPENCV",
        "aabb_scale": 1.0,
        "frames": []
    }


def save_smooth_video(
    image_folder: str,
    n_images_per_batch: int = 10,
    framerate: int = 15,
    skip_first_n_steps: int = 0,
    sort_type: Literal["alternating", "interleaving"] = "interleaving",
):
    images = os.listdir(image_folder)
    images = [x for x in images if "pred_file_" in x and "cond_" not in x]

    def sort_images(image_list: str):
        # get step,frame for each image
        def get_attr(x):
            parts = x.split("_")
            step = int(parts[3])
            frame = int(parts[-1].split(".")[0])
            return step, frame

        attrs = [get_attr(x) for x in image_list]

        # split into steps
        images_by_step = {}
        for (step, frame), img in zip(attrs, image_list):
            if step not in images_by_step:
                images_by_step[step] = []
            images_by_step[step].append((frame, img))

        # sort each step img_list
        for step, img_list in images_by_step.items():
            images_by_step[step] = sorted(img_list, key=lambda x: x[0])

        # combine
        final_list = []
        n_steps = len(images_by_step.keys())
        if sort_type == "interleaving":
            # sorting: from each step the _0000.png then _0001.png, ...
            for frame_idx in range(n_images_per_batch):
                for step_idx in range(skip_first_n_steps, n_steps):
                    final_list.append(images_by_step[step_idx][frame_idx][1])
        elif sort_type == "alternating":
            # sorting: first all odd steps in ascending order: _0000.png, _0001.png, ...
            #          second all even steps in descending order: _0009.png, _0008.png, ...
            odd_step_list = []
            even_step_list = []
            for step_idx in range(skip_first_n_steps, n_steps):
                is_even = (step_idx % 2) == 0
                l = even_step_list if is_even else odd_step_list
                for frame_idx in range(n_images_per_batch):
                    l.append(images_by_step[step_idx][frame_idx][1])
            final_list = [*odd_step_list, *even_step_list[::-1]]

        return final_list

    images = sort_images(images)
    images = [os.path.join(image_folder, x) for x in images]

    # copy images to tmp folder
    temp_dir = os.path.join(image_folder, f"tmp")
    file_paths = []
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for i, x in enumerate(images):
        file_name = f"{i:04d}_{os.path.basename(x)}_{i:04d}.png"
        file_path = os.path.join(str(temp_dir), file_name)
        file_paths.append(file_path)
        shutil.copy(x, file_path)

    # create video from tmp folder with images now in the correct order in that folder
    video_out_name = "smooth_render.mp4"
    temp_video_out_path = os.path.join(temp_dir, video_out_name)

    video_args = Namespace(
        **{
            "image_folder": temp_dir,
            "file_name_pattern_glob": "*.png",
            "output_path": temp_video_out_path,
            "framerate": framerate,
        }
    )
    create_video_from_image_folder(video_args)

    # create gif from tmp folder with images now in the correct order in that folder
    gif_out_name = "smooth_render.gif"
    temp_gif_out_path = os.path.join(temp_dir, gif_out_name)
    with imageio.get_writer(temp_gif_out_path, mode='I', duration=1.0 / framerate) as writer:
        for filename in file_paths:
            image = imageio.imread(filename)
            writer.append_data(image)

    # move video/gif out of tmp folder
    video_out_path = os.path.join(image_folder, video_out_name)
    gif_out_path = os.path.join(image_folder, gif_out_name)
    shutil.copy(temp_video_out_path, video_out_path)
    shutil.copy(temp_gif_out_path, gif_out_path)

    # remove tmp folder
    shutil.rmtree(temp_dir)


def main(
    input_path: str,
    output_path: str = None,
    step: int = -1,
    combine_all: bool = False,
    skip_first_n_steps: int = 0,
    create_smooth_video: bool = False,
    smooth_video_sort_type: Literal["alternating", "interleaving"] = "interleaving",
    smooth_video_framerate: int = 15,
    n_images_per_batch: int = 10,
    carvekit_checkpoint_dir: str = None,
):
    # get all cams and images files
    files = os.listdir(input_path)

    # determine which steps should be processed
    if step > -1:
        steps = [step]
    else:
        steps = [int(f.split(".")[0].split("_")[1]) for f in files if "cams" in f]
        steps = sorted(steps)
        steps = steps[skip_first_n_steps:]

    # where to save
    if output_path is None:
        output_path = os.path.join(input_path, "exported_nerf_convention")

    # get mask segmentation model
    carvekit_model = load_carvekit_bkgd_removal_model(carvekit_checkpoint_dir)

    # separately create transforms for all steps
    step_output_paths = []
    frame_dicts = []
    masked_images_frame_dicts = []
    for step in tqdm(steps, desc="Export files to NeRF convention"):
        # get corresponding files
        step_str = f"{step:04d}"
        cam_file = [os.path.join(input_path, f) for f in files if f"cams_{step_str}" in f]
        assert len(cam_file) == 1, f"found more than one possible cam_file: {cam_file}"
        cam_file = cam_file[0]
        image_files = [os.path.join(input_path, f) for f in files if f"pred_file_step_{step_str}" in f]

        # prepare transforms
        transforms_dict = get_transforms_header()
        transforms_dict_masked_images = get_transforms_header()

        step_output_path = os.path.join(output_path, step_str)
        step_output_paths.append(step_output_path)

        images_output_folder = "images"
        images_output_path = os.path.join(step_output_path, images_output_folder)
        os.makedirs(images_output_path, exist_ok=True)

        masked_images_output_folder = "masked_images"
        masked_images_output_path = os.path.join(step_output_path, masked_images_output_folder)
        os.makedirs(masked_images_output_path, exist_ok=True)

        masks_output_folder = "masks"
        masks_output_path = os.path.join(step_output_path, masks_output_folder)
        os.makedirs(masks_output_path, exist_ok=True)

        # load cams
        with open(cam_file, "rb") as f:
            cam = torch.load(f)
        poses = cam["poses"]
        intrs = cam["intrs"]

        # load each (cam, intr, image) tuple and add it to transforms
        for key in poses.keys():
            frame_dict = {}

            # add image file and h/w
            image_file_path = [f for f in image_files if key in f]
            if len(image_file_path) != 1:
                continue
            image_file_path = image_file_path[0]
            image_file_name = os.path.basename(image_file_path)
            image_file_output_path = os.path.join(images_output_path, image_file_name)
            shutil.copy(image_file_path, image_file_output_path)
            frame_dict["file_path"] = os.path.join(images_output_folder, image_file_name)
            with open(image_file_path, "rb") as f:
                image = np.array(Image.open(f))
                frame_dict["h"] = image.shape[0]
                frame_dict["w"] = image.shape[1]

            # add intr
            intr = intrs[key].numpy()
            frame_dict["fl_x"] = float(intr[0, 0])
            frame_dict["fl_y"] = float(intr[1, 1])
            frame_dict["cx"] = float(intr[0, 2])
            frame_dict["cy"] = float(intr[1, 2])

            # convert pose from OPENCV world2cam to OPEN_GL cam2world (it's what nerfstudio expects: https://docs.nerf.studio/quickstart/data_conventions.html)
            pose = poses[key].numpy()

            # first, convert from world2cam to cam2world
            R = pose[:3, :3]
            T = pose[:3, 3:4]
            Rinv = R.T
            Tinv = -Rinv @ T
            pose_cam2world = np.concatenate([Rinv, Tinv], axis=1)
            pose_cam2world = np.concatenate([pose_cam2world, pose[3:4]], axis=0)  # add hom

            # second, invert y/z coordinate
            pose_cam2world[:3, 1:3] *= -1

            frame_dict["transform_matrix"] = pose_cam2world.tolist()

            # save mask files
            masked_image_path = os.path.join(masked_images_output_path, image_file_name)
            mask_path = os.path.join(masks_output_path, image_file_name)
            if not os.path.exists(mask_path) or not os.path.exists(masked_image_path):
                segment_and_save_carvekit(image_file_output_path, mask_path, masked_image_path, mask_predictor=carvekit_model)
            frame_dict["mask_path"] = os.path.join(masks_output_folder, image_file_name)

            # add frame to unmasked transforms
            transforms_dict["frames"].append(frame_dict)
            frame_dicts.append((frame_dict, step_output_path))

            # replace image with masked image
            masked_image_frame_dict = copy.deepcopy(frame_dict)
            masked_image_frame_dict["file_path"] = os.path.join(masked_images_output_folder, image_file_name)

            # add frame to masked transforms
            transforms_dict_masked_images["frames"].append(masked_image_frame_dict)
            masked_images_frame_dicts.append((masked_image_frame_dict, step_output_path))

        # save transforms to file
        with open(os.path.join(step_output_path, "transforms.json"), "w") as f:
            json.dump(transforms_dict, f, indent=4)

        # save transforms to file
        with open(os.path.join(step_output_path, "transforms_masked_images.json"), "w") as f:
            json.dump(transforms_dict_masked_images, f, indent=4)

    # create a combined transforms from all steps
    if combine_all:
        step_output_path = os.path.join(output_path, "combined_all")
        step_output_paths.append(step_output_path)

        # prepare transforms
        transforms_dict = get_transforms_header()
        transforms_dict_masked_images = get_transforms_header()

        images_output_path = os.path.join(step_output_path, "images")
        os.makedirs(images_output_path, exist_ok=True)

        masked_images_output_path = os.path.join(step_output_path, "masked_images")
        os.makedirs(masked_images_output_path, exist_ok=True)

        masks_output_path = os.path.join(step_output_path, "masks")
        os.makedirs(masks_output_path, exist_ok=True)

        # copy together all files from all steps
        for (frame_dict, frame_dict_output_path), (masked_image_frame_dict, _) in zip(frame_dicts, masked_images_frame_dicts):
            image_file_name = os.path.basename(frame_dict["file_path"])
            if "cond" in image_file_name:
                continue

            # copy rgb image
            shutil.copy(
                os.path.join(frame_dict_output_path, frame_dict["file_path"]),
                os.path.join(images_output_path, image_file_name)
            )

            # copy mask image
            shutil.copy(
                os.path.join(frame_dict_output_path, frame_dict["mask_path"]),
                os.path.join(masks_output_path, image_file_name)
            )

            # copy masked rgb image
            shutil.copy(
                os.path.join(frame_dict_output_path, masked_image_frame_dict["file_path"]),
                os.path.join(masked_images_output_path, image_file_name)
            )

            # add frame
            transforms_dict["frames"].append(frame_dict)
            transforms_dict_masked_images["frames"].append(masked_image_frame_dict)

        # save transforms to file
        with open(os.path.join(step_output_path, "transforms.json"), "w") as f:
            json.dump(transforms_dict, f, indent=4)

        with open(os.path.join(step_output_path, "transforms_masked_images.json"), "w") as f:
            json.dump(transforms_dict_masked_images, f, indent=4)

        # create smooth videos
        if create_smooth_video:
            save_smooth_video(
                image_folder=masked_images_output_path,
                n_images_per_batch=n_images_per_batch,
                framerate=smooth_video_framerate,
                skip_first_n_steps=skip_first_n_steps,
                sort_type=smooth_video_sort_type
            )
            save_smooth_video(
                image_folder=images_output_path,
                n_images_per_batch=n_images_per_batch,
                framerate=smooth_video_framerate,
                skip_first_n_steps=skip_first_n_steps,
                sort_type=smooth_video_sort_type
            )

    # convert to sdfstudio format (without mono-cues)
    for step_output_path in tqdm(step_output_paths, desc="Convert to sdfstudio format"):
        args = {
            "input_dir": step_output_path,
            "output_dir": os.path.join(step_output_path, "sdfstudio-format"),
            "scene_type": "object",
            "scene_scale_mult": None,
            "mono_prior": False,
            "crop_mult": 1,
            "omnidata_path": None,
            "pretrained_models": None,
        }
        args = Namespace(**args)
        process_nerfstudio_to_sdfstudio(args)

    print("Done! The exported nerf data has been saved in:", output_path)


if __name__ == '__main__':
    tyro.cli(main)
