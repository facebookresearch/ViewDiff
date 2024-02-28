# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Tuple
from omegaconf import DictConfig

import os
import torch

from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import expand_args_fields


def json_index_dataset_load_category(dataset_root: str, category: str, dataset_args: DictConfig):
    # adapted from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/implicitron/dataset/json_index_dataset_map_provider_v2.py
    frame_file = os.path.join(
        dataset_root,
        category,
        "frame_annotations.jgz",
    )
    sequence_file = os.path.join(
        dataset_root,
        category,
        "sequence_annotations.jgz",
    )

    if not os.path.isfile(frame_file):
        # The frame_file does not exist.
        # Most probably the user has not specified the root folder.
        raise ValueError(
            f"Looking for frame annotations in {frame_file}." + " Please specify a correct dataset_root folder."
        )

    # setup the common dataset arguments
    common_dataset_kwargs = {
        **dataset_args,
        "dataset_root": dataset_root,
        "frame_annotations_file": frame_file,
        "sequence_annotations_file": sequence_file,
        "subsets": None,
        "subset_lists_file": "",
    }

    # get the used dataset type
    expand_args_fields(JsonIndexDataset)
    dataset = JsonIndexDataset(**common_dataset_kwargs)

    return dataset


def get_dataset(co3d_root, category, subset, split, **dataset_kw_args):
    print(f"start parse dataset for category={category}, subset={subset}")
    dataset_args = DictConfig(dataset_kw_args)

    if subset is None:
        # directly load JsonIndexDataset, do not use subset
        dataset = json_index_dataset_load_category(dataset_root=co3d_root, category=category, dataset_args=dataset_args)
    else:
        # use subset with JsonIndexDatasetMapProviderV2
        expand_args_fields(JsonIndexDatasetMapProviderV2)
        dataset_map = JsonIndexDatasetMapProviderV2(
            category=category,  # load this category
            subset_name=subset,  # load all sequences/frames that are specified in this subset
            test_on_train=False,  # want to load the actual test data
            only_test_set=False,  # want to have train/val/test splits accessible
            load_eval_batches=False,  # for generation do not need eval batches, rather go through "test" split of each sequence
            dataset_JsonIndexDataset_args=dataset_args,
        ).get_dataset_map()
        dataset = dataset_map[split]
    print("finish parse dataset")

    return dataset


def has_pointcloud(co3d_root: str, category: str, sequence_name: str) -> bool:
    """checks if the specified sequence has at a pointcloud object in the dataset.

    Args:
        co3d_root (str): root dir of dataset
        category (str): category of sequence
        sequence_name (str): sequence to check

    Returns:
        bool: True if the pointcloud exists, else False.
    """
    return os.path.exists(os.path.join(co3d_root, category, sequence_name, "pointcloud.ply"))


def get_crop_around_mask(mask, th, tw):
    # sanity checks
    h, w = mask.shape
    assert (
        h >= th and w >= tw
    ), f"crop height/width must not be larger than original height/width: orig=({h}, {w}), crop=({th}, {tw})"
    if th > h or tw > w:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    # get mask center coordinate
    coords = torch.nonzero(mask).float()
    mean_coord = torch.mean(coords, dim=0).int()

    # get top/left corner of crop rectangle
    top = max(0, mean_coord[0] - th // 2)
    left = max(0, mean_coord[1] - tw // 2)

    # check bounds
    top -= max(0, top + th - h)
    left -= max(0, left + tw - w)

    return top, left, th, tw


def adjust_crop_size(orig_hw, crop_hw):
    # extract values
    h, w = orig_hw
    th, tw = crop_hw

    # adjust crop_size such that the larger crop_size has the same size as orig_size
    scale = min(h / th, w / tw)
    new_h = int(th * scale)
    new_w = int(tw * scale)

    return new_h, new_w


def scale_intrinsics(K: torch.Tensor, orig_hw: Tuple[int, int], resized_hw: Tuple[int, int]) -> torch.Tensor:
    scaling_factor_h = resized_hw[0] / orig_hw[0]
    scaling_factor_w = resized_hw[1] / orig_hw[1]

    K = K.clone()
    K[..., 0, 0] = K[..., 0, 0] * scaling_factor_w
    K[..., 1, 1] = K[..., 1, 1] * scaling_factor_h

    # K[..., 0, 2] = K[..., 0, 2] * scaling_factor_w
    # K[..., 1, 2] = K[..., 1, 2] * scaling_factor_h

    # We assume opencv-convention ((0, 0) refers to the center of the top-left pixel and (-0.5, -0.5) is the top-left corner of the image-plane):
    # We need to scale the principal offset with the 0.5 add/sub like here.
    # see this for explanation: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * scaling_factor_w - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * scaling_factor_h - 0.5

    return K


def scale_depth(
    depth: torch.Tensor,
    orig_K: torch.Tensor,
    orig_hw: Tuple[int, int],
    resized_K: torch.Tensor,
    resized_hw: Tuple[int, int],
) -> torch.Tensor:
    # downsampling depth w/ nearest interpolation results in wrong coordinates
    # this projects pixel centers from the resized to the original image-plane and grid_samples the depth value at those positions (using nearest interpolation).

    # construct coordinate grid
    # convention: (0, 0) is the center of the pixel
    x = torch.arange(resized_hw[1], device=depth.device, dtype=torch.float32)
    y = torch.arange(resized_hw[0], device=depth.device, dtype=torch.float32)

    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    grid_xy = torch.cat(
        [
            grid_x.view(resized_hw[0], resized_hw[1], 1),
            grid_y.view(resized_hw[0], resized_hw[1], 1),
        ],
        dim=2,
    )
    grid_xy = grid_xy[None].repeat(depth.shape[0], 1, 1, 1)

    # apply inverse scaled intrinsics
    grid_xy[..., 0] = (1.0 / resized_K[..., 0, 0][..., None, None]) * (
        grid_xy[..., 0] - resized_K[..., 0, 2][..., None, None]
    )
    grid_xy[..., 1] = (1.0 / resized_K[..., 1, 1][..., None, None]) * (
        grid_xy[..., 1] - resized_K[..., 1, 2][..., None, None]
    )

    # apply original intrinsics
    grid_xy[..., 0] = orig_K[..., 0, 0][..., None, None] * grid_xy[..., 0] + orig_K[..., 0, 2][..., None, None]
    grid_xy[..., 1] = orig_K[..., 1, 1][..., None, None] * grid_xy[..., 1] + orig_K[..., 1, 2][..., None, None]

    # go from [-0.5, orig_hw - 0.5] to [-1, 1]
    grid_xy[..., 0] = (grid_xy[..., 0] + 0.5) / orig_hw[1] * 2 - 1.0
    grid_xy[..., 1] = (grid_xy[..., 1] + 0.5) / orig_hw[0] * 2 - 1.0

    # grid_sample with coordinates in orig screen-space
    scaled_depth = torch.nn.functional.grid_sample(
        depth.unsqueeze(1),
        grid_xy,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)

    return scaled_depth


def scale_camera_center(world2cam: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = world2cam.clone()
    x[..., :3, 3:4] *= scale[..., None, None]

    return x


def scale_bbox(bbox: torch.Tensor) -> torch.Tensor:
    # bbox assumed to be (N, 2, 3)
    largest_side_length = torch.max(bbox[:, 1] - bbox[:, 0], dim=1).values  # e.g. 1.0
    scale = 2 / largest_side_length  # want largest side to be 2.0
    bbox = bbox * scale[..., None, None]  # scale bbox uniformly
    return bbox, scale