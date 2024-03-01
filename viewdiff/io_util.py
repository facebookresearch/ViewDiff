# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from argparse import Namespace
import json
import imageio
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Literal, Optional
from dataclasses import dataclass
from collections.abc import MutableMapping, Iterable

from PIL import Image
import torch
from torch.nn.functional import interpolate

from torchvision.utils import make_grid

from .data.co3d.co3d_dataset import CO3DConfig

from .data.create_video_from_image_folder import main as create_video_from_image_folder


@dataclass
class SaveConfig:
    """Which file types should be saved in #save_inference_outputs()."""

    image_grids: bool = False
    pred_files: bool = True
    pred_video: bool = True
    pred_gif: bool = False
    denoise_files: bool = False
    denoise_video: bool = False
    cams: bool = True
    prompts: bool = True
    rendered_depth: bool = False
    cond_files: bool = False
    image_metrics: bool = True


@dataclass
class IOConfig:
    """Arguments for IO."""

    save: SaveConfig = SaveConfig()

    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
    """Path to pretrained model or model identifier from huggingface.co/models"""

    revision: Optional[str] = None
    """Revision of pretrained model identifier from huggingface.co/models."""

    output_dir: str = "output"
    """The output directory where the model predictions and checkpoints will be written."""

    experiment_name: Optional[str] = None
    """If this is set, will use this instead of the datetime string as identifier for the experiment."""

    logging_dir: str = "logs"
    """[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
        *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."""

    log_images_every_nth: int = 500
    """log images every nth step"""

    report_to: Literal["tensorboard", "custom_tensorboard"] = "custom_tensorboard"
    """The integration to report the results and logs to. Supported platforms are `"tensorboard"`"""

    checkpointing_steps: int = 500
    """Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming
        training using `--resume_from_checkpoint`."""

    checkpoints_total_limit: int = 2
    """Max number of checkpoints to store."""

    resume_from_checkpoint: Optional[str] = None
    """Whether training should be resumed from a previous checkpoint. Use a path saved by
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint."""

    automatic_checkpoint_resume: bool = False


def setup_output_directories(
    io_config: IOConfig,
    model_config: dataclass,
    dataset_config: CO3DConfig,
    is_train: bool = False,
):
    """Creates output directories for an experiment as specified in the provided configs.

    Args:
        io_config (dataclass): _description_
        model_config (dataclass): _description_
        dataset_config (CO3Dv2MaskAugmentationConfig): _description_
        is_train (bool, optional): _description_. Defaults to False.
    """

    if isinstance(dataset_config, CO3DConfig):
        # append category to output_dir
        category_name = "all" if dataset_config.category is None else dataset_config.category
        io_config.output_dir = os.path.join(io_config.output_dir, category_name)

        # append n_sequences to output_dir
        if dataset_config.max_sequences > -1:
            io_config.output_dir = os.path.join(io_config.output_dir, f"{dataset_config.max_sequences}_sequences")

        # append picked_sequences to output_dir
        io_config.output_dir = os.path.join(io_config.output_dir, ",".join(dataset_config.dataset_args.pick_sequence))

        # append subset to output_dir
        io_config.output_dir = os.path.join(
            io_config.output_dir, "subset_all" if dataset_config.subset is None else f"subset_{dataset_config.subset}"
        )
    else:
        raise NotImplementedError("unsupported dataset config", type(dataset_config))

    # append input/output information to output_dir
    io_config.output_dir = os.path.join(
        io_config.output_dir, f"input_{model_config.n_input_images}"
    )

    # append train/test information to output_dir
    io_config.output_dir = os.path.join(io_config.output_dir, "train" if is_train else "test")

    # append experiment name or current datetime to output_dir
    has_exp_name = hasattr(io_config, "experiment_name") and io_config.experiment_name is not None
    if has_exp_name:
        io_config.output_dir = os.path.join(io_config.output_dir, io_config.experiment_name)
    if not has_exp_name or not is_train:
        date_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S.%f")
        io_config.output_dir = os.path.join(io_config.output_dir, date_time)


def make_output_directories(io_config: IOConfig):
    if not os.path.exists(io_config.output_dir):
        os.makedirs(io_config.output_dir)
    io_config.image_out_dir = os.path.join(io_config.output_dir, "images")
    io_config.rendered_depth_out_dir = os.path.join(io_config.output_dir, "rendered_depth")
    io_config.stats_out_dir = os.path.join(io_config.output_dir, "stats")
    if not os.path.exists(io_config.image_out_dir):
        os.makedirs(io_config.image_out_dir)
    if not os.path.exists(io_config.rendered_depth_out_dir):
        os.makedirs(io_config.rendered_depth_out_dir)
    if not os.path.exists(io_config.stats_out_dir):
        os.makedirs(io_config.stats_out_dir)


def make_image_grid(*images) -> torch.Tensor:
    """Returns a grid of image tuples for the given batch that can be used for logging.
    Each row represents one batch sample with K pairs.
    Each tuple is the horizontal concatenation of the list of images.

    Args:
        images: list of images where each value is a tensor of shape (N, K, C, H, W). Expects all tensors in the same value range, s.t. normalization maps them to 0..1

    Returns:
        torch.Tensor: the grid image represented as tensor in range [0..1]
    """
    N, K = images[0].shape[:2]
    combined_image_list = []
    for n in range(N):
        for k in range(K):
            combined_image = torch.cat([img[n, k, :3].detach().cpu().float() for img in images], dim=-1)
            combined_image_list.append(combined_image)
    combined_image = make_grid(combined_image_list, nrow=K, normalize=True)

    return combined_image


def make_pred_grid(pred_images: List[torch.Tensor], *images) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a grid of image tuples for the given batch and predictions that can be used for logging.
    Each row represents one batch sample with K triplets.
    Each tuple is the horizontal concatenation of the list of images and the prediction.

    Args:
        images: list of images where each value is a tensor of shape (N, K, C, H, W). Expects all tensors in the same value range, s.t. normalization maps them to 0..1
        pred_images (List[torch.Tensor]): tensor of shape (N, K, C, H', W') in range [0..1].

    Returns:
        torch.Tensor: the grid image represented as tensor in range [0..1]
    """

    N, K = images[0].shape[:2]

    # normalize and resize input images
    converted_images = []
    for img in images:
        # reshape images if necessary
        if img.shape[-2:] != pred_images.shape[-2:]:
            img = img.view(N * K, *img.shape[2:])
            img = interpolate(img, pred_images.shape[-2:], mode="bilinear")
            img = img.view(N, K, *img.shape[1:])

        # normalize to 0..1
        img = norm_0_1(img)

        converted_images.append(img)

    return make_image_grid(*converted_images, pred_images)


def norm_0_1(x: torch.Tensor) -> torch.Tensor:
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val)


def torch_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Converts a tensor to a np.ndarray.

    Args:
        img (torch.Tensor): tensor in arbitrary range of type float32

    Returns:
        np.ndarray: np image of type uint8 in range [0..255]
    """
    # normalize to [0, 1]
    img = norm_0_1(img)

    # to [0, 255]
    img = (img * 255.0).to(torch.uint8)

    # (C, H, W) torch.Tensor to (H, W, C) np.array
    img = img.permute(1, 2, 0).cpu().numpy()

    return img


def torch_to_pil(img: torch.Tensor) -> Image:
    """Converts a tensor to a PIL Image.

    Args:
        img (torch.Tensor): tensor in arbitrary range

    Returns:
        Image: PIL Image in range [0..255]
    """
    img = torch_to_numpy(img)

    # np.array to PIL.Image
    img = Image.fromarray(img)

    return img


def convert_to_tensorboard_dict(x: Dict) -> Dict:
    """Converts a dictionary to a format supported for logging in tensorboard.
    That is: flattens the dictionary, replaces None with "", and replaces lists with str(list).

    Args:
        x (Dict): the dict to convert

    Returns:
        Dict: the converted dict
    """

    def flatten(dictionary, parent_key="", separator="_"):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    # hierarchical dicts cannot be logged in tb, replace with flattened dict
    x = flatten(x)

    # None cannot be logged in tb, replace with ""
    x = {k: v if v is not None else "" for k, v in x.items()}

    # lists cannot be logged in tb, replace with str repr
    x = {k: ", ".join(map(str, v)) if isinstance(v, Iterable) else v for k, v in x.items()}

    return x


def save_inference_outputs(
    batch: Dict[str, torch.Tensor],
    output,
    io_config: IOConfig,
    writer,
    step: int = 0,
    prefix: str = None,
):
    """Saves one batch/prediction to output folders and tensorboard.

    Args:
        batch (Dict[str, torch.Tensor]): the input batch
        output: the predicted output
        io_config (IOConfig): specifying output folder location
        writer (SummaryWriter): the tensorboard logger
        step (int, optional): the index of the batch. Defaults to 0.
        prefix (str, optional): prefix to use for all file/tensorboard outputs. Defaults to None.
    """
    if prefix is not None:
        tb_prefix = f"{prefix}/"
        prefix = f"{prefix}_"
    else:
        tb_prefix = ""
        prefix = ""

    # parse input to log in the batch
    batch_input_list = []
    keys = ["images"]
    for frame_idx in keys:
        if frame_idx in batch:
            batch_input_list.append(batch[frame_idx])

    # save image grid
    if io_config.save.image_grids:
        pred_image_grid = make_pred_grid(output.images, *batch_input_list)
        writer.add_image(f"{tb_prefix}Images", pred_image_grid, global_step=step)
        pred_image_grid_pil = torch_to_pil(pred_image_grid)
        with open(os.path.join(io_config.image_out_dir, f"{prefix}image_grid_{step:04d}.png"), "wb") as f:
            pred_image_grid_pil.save(f)

    # add image diffusion process
    root_output_path = io_config.image_out_dir
    file_patterns = []
    if io_config.save.denoise_files and hasattr(output, "image_list"):
        # save image denoise predictions as separate files
        file_patterns.append("denoise_files_")
        for time_idx, img in enumerate(output.image_list):
            img = make_image_grid(img)
            writer.add_image(f"{tb_prefix}/Denoise/{time_idx}", img, global_step=step)
            img = torch_to_pil(img)
            with open(os.path.join(root_output_path, f"{prefix}denoise_files_{step:04d}_{time_idx:04d}.png"), "wb") as f:
                img.save(f)

        if io_config.save.denoise_video:
            # save image denoise predictions as video
            file_patterns.append("denoise_video_")
            output_path = os.path.join(root_output_path, f"{prefix}denoise_video_{step:04d}.mp4")
            video_args = Namespace(
                **{
                    "image_folder": root_output_path,
                    "file_name_pattern_glob": f"{prefix}denoise_files_{step:04d}_*.png",
                    "output_path": output_path,
                    "framerate": 10,
                }
            )
            create_video_from_image_folder(video_args)

    # create list of pil images for attention logging and filename logging (shared)
    # save poses/intrs to dict
    N, K = output.images.shape[:2]
    pil_images = []
    cams = {
        "poses": {},
        "intrs": {}
    }
    for n in range(N):
        for frame_idx in range(K):
            # save the predictions using their original filenames
            file_name = batch["file_names"][frame_idx][n]
            sequence = os.path.basename(batch["root"][n])
            key = f"step_{step:04d}_seq_{sequence}_file_{file_name}_frame_{frame_idx:04d}"
            if io_config.save.pred_files:
                file_patterns.append("pred_file_")
                if io_config.save.cond_files or "cond_" not in file_name:
                    # convert to pil
                    img = torch_to_pil(output.images[n, frame_idx].detach().cpu())
                    pil_images.append(img)

                    # save image as file
                    with open(
                        os.path.join(root_output_path, f"{prefix}pred_file_{key}.png"),
                        "wb",
                    ) as f:
                        img.save(f)

                    # save cams in dict
                    cams["poses"][key] = batch["pose"][n, frame_idx]
                    cams["intrs"][key] = batch["K"][n, frame_idx]

    if io_config.save.pred_video:
        # save filename predictions as video
        file_patterns.append("pred_video_")
        output_path = os.path.join(root_output_path, f"{prefix}pred_video_{step:04d}.mp4")
        video_args = Namespace(
            **{
                "image_folder": root_output_path,
                "file_name_pattern_glob": f"{prefix}pred_file_step_{step:04d}_*.png",
                "output_path": output_path,
                "framerate": 5,
            }
        )
        create_video_from_image_folder(video_args)

    if io_config.save.pred_gif:
        file_patterns.append("pred_gif_")
        output_path = os.path.join(root_output_path, f"{prefix}pred_gif_{step:04d}.gif")
        with imageio.get_writer(output_path, mode='I', duration=1.0 / 15.0) as writer:
            for im in pil_images:
                writer.append_data(np.array(im))

    # save poses/intrinsics
    if io_config.save.cams:
        file_patterns.append("cams_")
        with open(
                os.path.join(root_output_path, f"{prefix}cams_{step:04d}.pt"),
                "wb",
        ) as f:
            torch.save(cams, f)

    # save prompts
    if io_config.save.prompts:
        file_patterns.append("prompts_")
        with open(
                os.path.join(root_output_path, f"{prefix}prompts_{step:04d}.txt"),
                "w",
        ) as f:
            for p in batch["prompt"]:
                f.write(f"{p}\n")

    # save image metrics
    if io_config.save.image_metrics and hasattr(output, "image_metrics"):
        with open(os.path.join(io_config.stats_out_dir, f"image_metrics_{step}.json"), "w") as f:
            json.dump(output.image_metrics, f, indent=4)

    # save rendered_depth and rendered_mask
    if (
        io_config.save.rendered_depth
        and hasattr(output, "rendered_depth")
        and output.rendered_depth is not None
        and hasattr(output, "rendered_mask")
        and output.rendered_mask is not None
    ):
        root_out = io_config.rendered_depth_out_dir

        # t goes over the timesteps, have one list of "rendered_depth/mask per layer" per timestep
        per_layer_image_list = {}
        for t, (depth_per_layer, mask_per_layer) in enumerate(zip(output.rendered_depth, output.rendered_mask)):
            if depth_per_layer is None or mask_per_layer is None:
                continue
            # i goes over the layers, have one "rendered_depth/mask" per layer
            for i, (d, m) in enumerate(zip(depth_per_layer, mask_per_layer)):
                # get grid
                rendered_depth_mask_grid = make_image_grid(norm_0_1(d), norm_0_1(m))

                # write to tb
                writer.add_image(f"{tb_prefix}Rendered-Depth-Mask/{i}/{t}", rendered_depth_mask_grid, global_step=step)

                # convert to pil
                rendered_depth_mask_grid_pil = torch_to_pil(rendered_depth_mask_grid)

                # save for video
                if i not in per_layer_image_list:
                    per_layer_image_list[i] = []
                per_layer_image_list[i].append(rendered_depth_mask_grid_pil)

                # save individual files to disk
                with open(
                    os.path.join(
                        root_out,
                        f"{prefix}rendered_depth_mask_grid_{step:04d}_layer_{i:04d}_step_{t:04d}.png",
                    ),
                    "wb",
                ) as f:
                    rendered_depth_mask_grid_pil.save(f)

        # save video per layer
        for frame_idx, image_list in per_layer_image_list.items():
            video_out = os.path.join(root_out, f"diff_process_{step:04d}_{frame_idx}.mp4")

            # save all files locally
            for t, img in enumerate(image_list):
                file_out = os.path.join(
                    root_out, f"{prefix}rendered_depth_mask_grid_{step:04d}_layer_{frame_idx:04d}_step_{t:04d}.png"
                )
                if not os.path.exists(file_out):
                    with open(file_out, "wb") as f:
                        img.save(f)

            # create video locally
            video_args = Namespace(
                **{
                    "image_folder": root_out,
                    "file_name_pattern_glob": f"{prefix}rendered_depth_mask_grid_{step:04d}_layer_{frame_idx:04d}_step_*.png",
                    "output_path": video_out,
                    "framerate": 5,
                }
            )
            create_video_from_image_folder(video_args)


def create_videos(io: IOConfig, prefix: str = None):
    if prefix is not None:
        prefix = f"{prefix}_"
    else:
        prefix = ""

    root_output_path = io.image_out_dir
    output_path = os.path.join(root_output_path, "image_grid.mp4")
    image_folder = io.image_out_dir
    if image_folder[-1] != "/":
        image_folder += "/"
    video_args = Namespace(
        **{
            "image_folder": image_folder,
            "file_name_pattern_glob": f"{prefix}image_grid_*.png",
            "output_path": output_path,
            "framerate": 15,
        }
    )
    create_video_from_image_folder(video_args)
    video_args.file_name_pattern_glob = f"{prefix}pred_*.png"
    video_args.output_path = os.path.join(root_output_path, "pred.mp4")
    create_video_from_image_folder(video_args)
