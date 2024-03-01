# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop
from torch.nn.functional import interpolate

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from pytorch3d.io import IO

from .util import (
    get_dataset,
    has_pointcloud,
    get_crop_around_mask,
    adjust_crop_size,
)


@dataclass
class DatasetArgsConfig:
    """Arguments for JsonIndexDataset. See here for a full list: pytorch3d/implicitron/dataset/json_index_dataset.py"""

    remove_empty_masks: bool = False
    """Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr)."""

    load_point_clouds: bool = False
    """If pointclouds should be loaded from the dataset"""

    load_depths: bool = False
    """If depth_maps should be loaded from the dataset"""

    load_depth_masks: bool = False
    """If depth_masks should be loaded from the dataset"""

    load_masks: bool = False
    """If foreground masks should be loaded from the dataset"""

    box_crop: bool = False
    """Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected."""

    image_width: Optional[int] = None
    """The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    image_height: Optional[int] = None
    """The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    pick_sequence: Tuple[str, ...] = ()
    """A list of sequence names to restrict the dataset to."""

    exclude_sequence: Tuple[str, ...] = ()
    """a list of sequences to exclude"""

    n_frames_per_sequence: int = -1
    """If > 0, randomly samples #n_frames_per_sequence
        frames in each sequences uniformly without replacement if it has
        more frames than that; applied before other frame-level filters."""


@dataclass
class BatchConfig:
    """Arguments for how batches are constructed."""

    n_parallel_images: int = 5
    """How many images of the same sequence are selected in one batch (used for multi-view supervision)."""

    image_width: int = 512
    """The desired image width after applying all augmentations (e.g. crop) and resizing operations."""

    image_height: int = 512
    """The desired image height after applying all augmentations (e.g. crop) and resizing operations."""

    other_selection: Literal["random", "sequence", "mix", "fixed-frames"] = "random"
    """How to select the other frames for each batch.
        The mode 'random' selects the other frames at random from all remaining images in the dataset.
        The mode 'sequence' selects the other frames in the order as they appear after the first frame (==idx) in the dataset. Selects i-th other image as (idx + i * sequence_offset).
        The mode 'mix' decides at random which of the other two modes to choose. It also randomly samples sequence_offset when choosing the mode 'sequence'.
        The mode 'fixed-frames' gets as frame indices as input and directly uses them."""

    other_selection_frame_indices: Tuple[int, ...] = ()
    """The frame indices to use when --other_selection=fixed-frames. Must be as many indices as --n_parallel_images."""

    sequence_offset: int = 1
    """If other_selection='sequence', uses this offset to determine how many images to skip for each next frame.
    Allows to do short-range and long-range consistency tests by setting to a small or large number."""

    crop: Literal["random", "foreground", "resize", "center"] = "random"
    """Performs a crop on the original image such that the desired (image_height, image_width) is achieved. 
       The mode 'random' crops randomly in the image.
       The mode 'foreground' crops centered around the foreground (==object) mask.
       The mode 'resize' performs brute-force resizing which ignores the aspect ratio.
       The mode 'center' crops centered around the middle pixel (similar to DFM baseline)."""

    mask_foreground: bool = False
    """If true, will mask out the background and only keep the foreground."""

    prompt: str = "Editorial Style Photo, ${category}, 4k --ar 16:9"
    """The text prompt for generation. The string ${category} will be replaced with the actual category."""

    use_blip_prompt: bool = False
    """If True, will use blip2 generated prompts for the sequence instead of the prompt specified in --prompt."""

    load_recentered: bool = False
    """If True, will load the recentered poses/bbox from the dataset. Will skip all sequences for which this was not pre-computed."""

    replace_pose_with_spherical_start_phi: float = -400.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -400, meaning do not replace."""

    replace_pose_with_spherical_end_phi: float = 360.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -1, meaning do not replace."""

    replace_pose_with_spherical_phi_endpoint: bool = False
    """If True, will set endpoint=True for np.linspace, else False."""

    replace_pose_with_spherical_radius: float = 4.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this radius. Default: 3.0."""

    replace_pose_with_spherical_theta: float = 45.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this elevation. Default: 45.0."""


@dataclass
class CO3DConfig:
    """Arguments for setup of the CO3Dv2_Dataset."""

    dataset_args: DatasetArgsConfig
    batch: BatchConfig

    co3d_root: str
    """Path to the co3dv2 root directory"""

    category: Optional[str] = None
    """If specified, only selects this category from the dataset. Can be a comma-separated list of categories as well."""

    subset: Optional[str] = None
    """If specified, only selects images corresponding to this subset. See https://github.com/facebookresearch/co3d for available options."""

    split: Optional[str] = None
    """Must be specified if --subset is specified. Tells which split to use from the subset."""

    max_sequences: int = -1
    """If >-1, randomly select max_sequence sequences per category. Only sequences _with pointclouds_ are selected. Mutually exclusive with --sequence."""

    seed: int = 42
    """Random seed for all rng objects"""


def spherical_to_cartesian(phi, theta, radius):
    # adapted from: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return torch.tensor([x, y, z], dtype=torch.float32)


def lookAt(eye, at, up):
    # adapted from: https://ksimek.github.io/2012/08/22/extrinsic/
    L = at - eye
    L = torch.nn.functional.normalize(L, dim=-1)
    s = torch.linalg.cross(L, up, dim=-1)
    s = torch.nn.functional.normalize(s, dim=-1)
    u = torch.linalg.cross(s, L, dim=-1)

    R = torch.stack([s, u, -L], dim=-2)
    t = torch.bmm(R, eye[..., None])
    w2c = torch.cat([R, t], dim=-1)
    hom = torch.zeros_like(w2c[..., 0:1, :])
    hom[..., -1] = 1
    w2c = torch.cat([w2c, hom], dim=-2)
    return w2c


def sample_uniform_poses_on_sphere(n_samples, radius=3.0, start_phi=0.0, end_phi=360.0, theta=30.0, endpoint: bool = False):
    eye = torch.stack([spherical_to_cartesian(phi=phi, theta=theta, radius=radius) for phi in np.linspace(start_phi, end_phi, n_samples, endpoint=endpoint)], dim=0)
    at = torch.tensor([[0, 0, 0]] * n_samples, dtype=torch.float32)
    up = torch.tensor([[0, 0, 1]] * n_samples, dtype=torch.float32)
    w2c = lookAt(eye, at, up)
    return w2c


def sample_random_poses_on_sphere(n_samples, radius=3.0, start_phi=0.0, end_phi=360.0, theta=30.0):
    phi_values = np.random.uniform(low=start_phi, high=end_phi, size=(n_samples,))
    eye = torch.stack([spherical_to_cartesian(phi=phi, theta=theta, radius=radius) for phi in phi_values], dim=0)
    at = torch.tensor([[0, 0, 0]] * n_samples, dtype=torch.float32)
    up = torch.tensor([[0, 0, 1]] * n_samples, dtype=torch.float32)
    w2c = lookAt(eye, at, up)
    return w2c


class CO3DDataset(Dataset):
    def __init__(self, config: CO3DConfig):
        self.config = config

        # sanity checks
        assert (
            os.getenv("CO3DV2_DATASET_ROOT") == self.config.co3d_root
        ), f"co3d env and arg do not match: {os.getenv('CO3DV2_DATASET_ROOT')} vs {self.config.co3d_root}"

        # get the category list
        with open(os.path.join(self.config.co3d_root, "category_to_subset_name_list.json"), "r") as f:
            category_to_subset_name_list = json.load(f)

        # try to load prompt file
        self.blip_prompts = {}
        if self.config.batch.use_blip_prompt:
            blip_prompt_files = [f for f in os.listdir(self.config.co3d_root) if "co3d_blip2_captions" in f]
            if len(blip_prompt_files) > 0:
                for f in tqdm(blip_prompt_files, desc="Load Blip Prompts"):
                    with open(os.path.join(self.config.co3d_root, f), "r") as ff:
                        blip_prompts = json.load(ff)
                    for category, sequence_dict in blip_prompts.items():
                        if category not in self.blip_prompts:
                            self.blip_prompts[category] = {}
                        for sequence, prompts in sequence_dict.items():
                            if sequence not in self.blip_prompts[category]:
                                self.blip_prompts[category][sequence] = []
                            self.blip_prompts[category][sequence].extend(prompts)

        # parse additional info from config
        self.subset_name = self.config.subset if self.config.subset is not None else "all"

        # update values in dataset_args according to config
        if config.batch.mask_foreground or config.batch.crop == "foreground":
            self.config.dataset_args.load_masks = True  # needed to perform foreground segmentation

        filter_frames_afterwards = self.subset_name != "all" and self.config.dataset_args.n_frames_per_sequence > -1
        if filter_frames_afterwards:
            # when filtering with subsets, disable n_frames_per_sequence during construction and filter afterwards
            # this is because of a bug in the pytorch3d dataset --> it will filter frames BEFORE filtering subsets
            # this way, we might be left with filtered frames that do not belong to the subset and then the subset is empty
            orig_n_frames_per_sequence = self.config.dataset_args.n_frames_per_sequence
            self.config.dataset_args.n_frames_per_sequence = -1

        # iterate over the co3d categories
        self.categories = sorted(list(category_to_subset_name_list.keys()))
        selected_categories = self.config.category.split(",") if self.config.category is not None else None
        self.category_dict = {}
        self.len = 0
        self.offset_to_category = []
        for category in tqdm(self.categories):
            # check if category is selected
            if selected_categories is not None and category not in selected_categories:
                # print(f"Skip {category}, not selected")
                continue

            # check if category exists
            category_path = os.path.join(self.config.co3d_root, category)
            if not os.path.exists(category_path):
                # print(f"Skip {category}, does not exist", category_path)
                continue

            # obtain the dataset (== metadata of all sequences for that category)
            dataset = get_dataset(
                co3d_root=self.config.co3d_root,
                category=category,
                subset=self.config.subset,
                split=self.config.split,
                **vars(self.config.dataset_args),
            )

            # get all sequences
            sequence_names = list(dataset.seq_annots.keys())

            # filter with custom filters, maybe can accelerate it when we have information in a single file already
            custom_sequence_metadata_file = os.path.join(category_path, "custom_sequence_metadata.json")
            update_custom_sequence_metadata = False
            if os.path.isfile(custom_sequence_metadata_file):
                with open(custom_sequence_metadata_file, "r") as f:
                    custom_sequence_metadata = json.load(f)
                n_before = len(sequence_names)

                # filter those sequences that exist
                sequence_names = [s for s in sequence_names if s in custom_sequence_metadata]

                # filter sequences that have pointclouds
                sequence_names = [s for s in sequence_names if custom_sequence_metadata[s]["has_pointcloud"]]
                print(f"Category {category} has {len(sequence_names)} sequences with valid pointclouds.")
                if len(sequence_names) == 0:
                    print(f"Skip {category}, found no sequences with valid pointclouds")
                    continue

                # filter those sequences that are valid, when we want recentered sequences
                if self.config.batch.load_recentered:
                    sequence_names = [s for s in sequence_names if custom_sequence_metadata[s]["is_valid_recentered"]]
                    if len(sequence_names) == 0:
                        print(f"Skip {category}, found no sequences with valid recentered poses/bbox.")
                        continue
                    print(f"Category {category} has {len(sequence_names)} sequences with valid recentered data.")

                print(
                    f"loaded {len(sequence_names)}/{n_before} valid sequences from",
                    custom_sequence_metadata_file,
                )
            else:
                print("did not find custom_sequence_metadata in", custom_sequence_metadata_file)
                custom_sequence_metadata = {}
                update_custom_sequence_metadata = True

                # filter those sequences that exist
                sequence_names = [s for s in sequence_names if os.path.exists(os.path.join(category_path, s))]
                if len(sequence_names) == 0:
                    print(f"Skip {category}, found no matching sequences in: {category_path}")
                    continue

                # collect all filters
                sequence_has_pointcloud = [has_pointcloud(self.config.co3d_root, category, s) for s in sequence_names]
                sequence_is_valid_recentered = [
                    os.path.exists(os.path.join(category_path, s, ".is_valid")) for s in sequence_names
                ]
                sequence_is_invalid_recentered = [
                    os.path.exists(os.path.join(category_path, s, ".is_invalid")) for s in sequence_names
                ]

                # write results to custom_sequence_metadata
                for s, has_pcl, is_valid_recentered, is_invalid_recentered in zip(
                    sequence_names,
                    sequence_has_pointcloud,
                    sequence_is_valid_recentered,
                    sequence_is_invalid_recentered,
                ):
                    custom_sequence_metadata[s] = {
                        "has_pointcloud": has_pcl,
                        "is_valid_recentered": is_valid_recentered,
                        "is_invalid_recentered": is_invalid_recentered,
                    }

                # filter sequences that have pointclouds
                sequence_names = [s for s, has_pcl in zip(sequence_names, sequence_has_pointcloud) if has_pcl]
                print(f"Category {category} has {len(sequence_names)} sequences with valid pointclouds.")
                if len(sequence_names) == 0:
                    print(f"Skip {category}, found no sequences with valid pointclouds")
                    continue

                # filter those sequences that are valid, when we want recentered sequences
                if self.config.batch.load_recentered:
                    sequence_names = [
                        s
                        for s, is_valid_recentered in zip(sequence_names, sequence_is_valid_recentered)
                        if is_valid_recentered
                    ]
                    if len(sequence_names) == 0:
                        print(f"Skip {category}, found no sequences with valid recentered poses/bbox.")
                        continue
                    print(f"Category {category} has {len(sequence_names)} sequences with valid recentered data.")

            # maybe update custom_sequence_metadata_file
            if update_custom_sequence_metadata:
                print("save custom_sequence_metadata to", custom_sequence_metadata_file)
                with open(custom_sequence_metadata_file, "w") as f:
                    json.dump(custom_sequence_metadata, f, indent=4)

            # filter max_sequences
            if self.config.max_sequences > -1:
                # randomly select sequences
                sequence_names = random.sample(
                    sequence_names,
                    k=min(self.config.max_sequences, len(sequence_names)),
                )
                print(f"Selected {len(sequence_names)} random sequences for category {category}.")

            # get frame ids that belong to the left-over sequences
            frame_ids = []
            offset_to_sequence = []
            sequence_lengths = []
            for s in sequence_names:
                frame_ids_s = [(s, x[1]) for x in list(dataset.sequence_frames_in_order(s))]
                if filter_frames_afterwards:
                    # now subsample frame_ids
                    print(
                        f"Filter {len(frame_ids_s)} frames of sequence {s} down to the requested {orig_n_frames_per_sequence} frames after constructing the dataset because we specified subset/split values."
                    )
                    random.shuffle(frame_ids_s)
                    frame_ids_s = frame_ids_s[:orig_n_frames_per_sequence]
                    self.config.dataset_args.n_frames_per_sequence = orig_n_frames_per_sequence
                offset_to_sequence.append((len(frame_ids), s))
                sequence_lengths.append(len(frame_ids_s))
                frame_ids.extend(frame_ids_s)
            dataset = dataset.subset_from_frame_index(frame_ids)

            # add category
            self.category_dict[category] = {
                "dataset": dataset,
                "sequences": sequence_names,
                "n_sequences": len(sequence_names),
                "offset_to_sequence": offset_to_sequence,
                "sequence_lengths": sequence_lengths
            }
            self.offset_to_category.append((self.len, category))
            self.len += len(dataset)
            print(f"Parsed {category}: using {len(sequence_names)} sequences with {len(dataset)} frames in total.")

        self.valid_categories = list(self.category_dict.keys())
        self.n_valid_categories = len(self.valid_categories)

    def get_all_sequences(self):
        sequences = ()
        for c in self.valid_categories:
            sequences += tuple(self.category_dict[c]["sequences"])
        return sequences

    def __len__(self):
        return self.len

    @staticmethod
    def find_entry_in_offset_list(idx, offset_list):
        val = None
        offset = 0
        for o, v in offset_list:
            if idx < o:
                break
            val = v
            offset = o

        if val is None:
            raise ValueError("idx is out of range", idx)

        return val, offset

    @staticmethod
    def convert_camera(
        camera: PerspectiveCameras,
        source_image_hw: Tuple[int, int],
        target_image_hw: Tuple[int, int],
        crop_params: Tuple[int, int, int, int] = (0, 0, -1, -1),  # top, left, h, w
    ):
        # have pytorch3d convention, want opencv convention --> convert it w/ official function
        image_hw = torch.tensor([list(source_image_hw)] * len(camera), device=camera.device)
        R, tvec, K = opencv_from_cameras_projection(camera, image_size=image_hw)
        pose_world2cam = torch.cat([R, tvec.unsqueeze(2)], dim=-1)

        # add hom row
        hom = torch.zeros(4, dtype=pose_world2cam.dtype)[None, None]
        hom[..., -1] = 1
        pose_world2cam = torch.cat([pose_world2cam, hom], dim=1)

        # if the source image was cropped from the top/left we also need to adjust the principal point and source_image_hw accordingly
        crop_top = crop_params[0]
        crop_left = crop_params[1]
        crop_h = crop_params[2]
        crop_w = crop_params[3]
        K[..., 0, 2] -= crop_left
        K[..., 1, 2] -= crop_top
        adjusted_source_image_hw = [
            crop_h if crop_h > 0 else source_image_hw[0],
            crop_w if crop_w > 0 else source_image_hw[1],
        ]

        # the intrinsics are stored in the convention that (0, 0) refers to the top-left corner of the image-plane (== pytorch3d convention)
        # We want the convention that (0, 0) refers to the center of the top-left pixel and (-0.5, -0.5) is the top-left corner of the image-plane:
        # We shift the principal point here, e.g. from 256 to 255.5.
        K[..., 0:2, 2] -= 0.5

        # scale intrinsics to new image size
        scaling_factor_h = target_image_hw[0] / adjusted_source_image_hw[0]
        scaling_factor_w = target_image_hw[1] / adjusted_source_image_hw[1]

        K[..., 0, 0] = K[..., 0, 0] * scaling_factor_w
        K[..., 1, 1] = K[..., 1, 1] * scaling_factor_h

        # We assume opencv-convention ((0, 0) refers to the center of the top-left pixel and (-0.5, -0.5) is the top-left corner of the image-plane):
        # We need to scale the principal offset with the 0.5 add/sub like here.
        # see this for explanation: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * scaling_factor_w - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * scaling_factor_h - 0.5

        return pose_world2cam, K

    def recenter_sequences(self, recompute: bool = False):
        for category_offset, category in self.offset_to_category:
            category_dict = self.category_dict[category]
            category_path = os.path.join(self.config.co3d_root, category)
            custom_sequence_metadata_file = os.path.join(category_path, "custom_sequence_metadata.json")
            with open(custom_sequence_metadata_file, "r") as f:
                custom_sequence_metadata = json.load(f)
                update_custom_sequence_metadata = False
            for sequence_offset, sequence in tqdm(
                category_dict["offset_to_sequence"], desc=f"Recenter Sequences: {category}"
            ):
                sequence_dir = os.path.join(category_path, sequence)
                valid_file = os.path.join(sequence_dir, ".is_valid")
                invalid_file = os.path.join(sequence_dir, ".is_invalid")

                if recompute:
                    update_custom_sequence_metadata = True
                    if os.path.exists(valid_file):
                        os.remove(valid_file)
                    if os.path.exists(invalid_file):
                        os.remove(invalid_file)
                else:
                    # check if the files exist
                    is_valid = os.path.exists(valid_file) and os.path.isfile(valid_file)
                    is_invalid = os.path.exists(invalid_file) and os.path.isfile(invalid_file)

                    # fast return: already checked once + check if we need to update in metadata
                    if is_valid or is_invalid:
                        if custom_sequence_metadata[sequence]["is_valid_recentered"] != is_valid:
                            update_custom_sequence_metadata = True
                            custom_sequence_metadata[sequence]["is_valid_recentered"] = is_valid
                        if custom_sequence_metadata[sequence]["is_invalid_recentered"] != is_invalid:
                            update_custom_sequence_metadata = True
                            custom_sequence_metadata[sequence]["is_invalid_recentered"] = is_invalid
                        continue

                def write_invalid():
                    custom_sequence_metadata[sequence]["is_valid_recentered"] = False
                    custom_sequence_metadata[sequence]["is_invalid_recentered"] = True
                    open(invalid_file, "a").close()

                def write_valid():
                    custom_sequence_metadata[sequence]["is_valid_recentered"] = True
                    custom_sequence_metadata[sequence]["is_invalid_recentered"] = False
                    open(valid_file, "a").close()

                def compute_camera_plane(poses_cam2world):
                    camera_centers = poses_cam2world[:, :3, 3]

                    # Compute camera centers' plane.
                    centers_mean = camera_centers.mean(axis=0)
                    normalized_centers = camera_centers - centers_mean[np.newaxis, :]

                    covariance_matrix = np.dot(
                        normalized_centers.T, normalized_centers
                    )  # Could also use np.cov(x) here.
                    U, D, W = np.linalg.svd(covariance_matrix)

                    # To transform from world to camera plane, the following transformation
                    # is needed: p' = W * (p - c_mean)
                    R_world2plane = W
                    t_world2plane = -np.matmul(W, centers_mean.reshape(3, 1)).reshape(3)

                    # T_world2plane = np.zeros((4, 4), dtype=poses_cam2world.dtype)
                    T_world2plane = np.eye(4, dtype=poses_cam2world.dtype)
                    T_world2plane[:3, :3] = R_world2plane
                    T_world2plane[:3, 3] = t_world2plane

                    return T_world2plane

                # get all indices for this sequence
                dataset_indices = [x[2] for x in list(category_dict["dataset"].sequence_frames_in_order(sequence))]

                # load data from all frames
                poses_world2cam = []
                filenames = []
                K_tensor = []
                has_pcl = False
                checked_pcl = False
                for i, frame_idx in enumerate(tqdm(dataset_indices, desc=f"{sequence}: Load Poses", leave=True)):
                    frame_data = self.load_frame_data(category, frame_idx)

                    # check has_pcl in first frame only
                    if not checked_pcl:
                        checked_pcl = True
                        has_pcl = frame_data.sequence_point_cloud is not None
                        if not has_pcl:
                            break
                        else:
                            pcl = frame_data.sequence_point_cloud

                    # load pose
                    image_hw = frame_data.image_rgb.shape[1:]
                    pose_world2cam, K = self.convert_camera(
                        frame_data.camera, source_image_hw=image_hw, target_image_hw=image_hw
                    )
                    poses_world2cam.append(pose_world2cam)
                    K_tensor.append(K)

                    # load filename
                    filename = Path(frame_data.image_path).stem
                    filenames.append(filename)

                # next sequence
                if not has_pcl:
                    write_invalid()
                    continue

                if len(pose_world2cam) == 0:
                    write_invalid()
                    continue

                world2cam_tensor = torch.cat(poses_world2cam).cpu().numpy()

                # get world2plane matrix
                cam2world_tensor = np.linalg.inv(world2cam_tensor)
                try:
                    T_world2plane = compute_camera_plane(cam2world_tensor)
                except np.linalg.LinAlgError:
                    # if SVD does not converge, we assume the scene is invalid
                    write_invalid()
                    continue

                # modify points & cams w/ world2plane
                cam2world_tensor = np.matmul(T_world2plane[np.newaxis, ...], cam2world_tensor)
                pcl_points = pcl.points_padded().permute(1, 2, 0)  # (N, 3, 1)
                pcl_points = torch.cat([pcl_points, torch.ones_like(pcl_points[:, 0:1])], dim=1)  # (N, 4, 1)
                T_world2plane = torch.from_numpy(T_world2plane)[None, ...]  # (N, 4, 4)
                T_world2plane = T_world2plane.repeat(pcl_points.shape[0], 1, 1)
                pcl_points = T_world2plane.bmm(pcl_points)
                pcl_points = pcl_points[:, :3]

                # center at origin
                bbox_min = pcl_points.min(dim=0).values
                bbox_max = pcl_points.max(dim=0).values
                offset = -(bbox_min + bbox_max) / 2
                offset = offset[None, ...]
                pcl_points = pcl_points + offset
                cam2world_tensor[..., :3, 3:4] += offset.cpu().numpy()

                # scale s.t. bbox has size 2 --> centered at [-1, 1]^3
                scale = 2 / (bbox_max - bbox_min).max()
                pcl_points *= scale
                cam2world_tensor[..., :3, 3:4] *= scale.cpu().numpy()

                # update final values
                pcl_points = pcl_points.permute(2, 0, 1)
                pcl = pcl.update_padded(pcl_points)
                world2cam_tensor = np.linalg.inv(cam2world_tensor)

                # vis_cams(world2cam_tensor, K_tensor, image_hw[0], image_hw[1], pcl=pcl)

                # save pointcloud
                IO().save_pointcloud(pcl, os.path.join(sequence_dir, "recentered.ply"))

                # save bbox
                bbox = pcl.get_bounding_boxes().transpose(1, 2).squeeze(0)
                bbox = bbox.cpu().numpy()
                bbox_path = os.path.join(sequence_dir, f"bbox_recentered.npy")
                np.save(bbox_path, bbox)

                # save poses
                poses_path = os.path.join(sequence_dir, "recentered_poses_world2cam")
                os.makedirs(poses_path, exist_ok=True)
                for p, f in tqdm(
                    zip(world2cam_tensor, filenames), total=len(filenames), desc=f"{sequence}: Save Poses", leave=True
                ):
                    pose_path = os.path.join(poses_path, f"pose_world2cam_{f}.npy")
                    np.save(pose_path, p)

                # mark finished
                write_valid()

            # after all sequences of the category were processed: save custom_sequence_metadata
            if update_custom_sequence_metadata:
                print("save custom_sequence_metadata to", custom_sequence_metadata_file)
                with open(custom_sequence_metadata_file, "w") as f:
                    json.dump(custom_sequence_metadata, f, indent=4)

    def get_frames(self, idx):
        # select dataset
        category, category_offset = self.find_entry_in_offset_list(idx, self.offset_to_category)
        category_dict = self.category_dict[category]
        idx = idx - category_offset

        # select sequence
        sequence, sequence_offset = self.find_entry_in_offset_list(idx, category_dict["offset_to_sequence"])
        idx = idx - sequence_offset

        # get all indices for this sequence
        dataset_indices = [x[2] for x in list(category_dict["dataset"].sequence_frames_in_order(sequence))]

        # first frame is specified by idx
        first_index = dataset_indices[idx]

        # determine other_selection
        other_selection_mode = self.config.batch.other_selection
        other_selection_sequence_offset = self.config.batch.sequence_offset
        if other_selection_mode == "mix":
            other_selection_mode = random.choice(["random", "sequence"])
            other_selection_sequence_offset = random.choice([i for i in range(5)])

        # get other frames
        if other_selection_mode == "random":
            # other frames are random from remaining
            dataset_indices.pop(idx)
            other_indices = random.sample(
                dataset_indices,
                k=self.config.batch.n_parallel_images - 1,
            )
        elif other_selection_mode == "sequence":
            # other frames are in order of dataset_indices with a constant offset in between
            other_indices = [
                idx + other_selection_sequence_offset * (i + 1) for i in range(self.config.batch.n_parallel_images - 1)
            ]
            other_indices = [dataset_indices[i % len(dataset_indices)] for i in other_indices]
        elif other_selection_mode == "fixed-frames":
            first_index = dataset_indices[self.config.batch.other_selection_frame_indices[0]]
            other_indices = [dataset_indices[x] for x in self.config.batch.other_selection_frame_indices[1:]]
        else:
            raise ValueError("unsupported other_selection", other_selection_mode)

        # combine into final selection
        dataset_indices = [first_index, *other_indices]

        return category, sequence, dataset_indices

    def get_prompt_for_category(self, category: str):
        return self.config.batch.prompt.replace("${category}", category)

    def get_blip_prompt(self, category: str, sequence: str):
        prompt = ""
        if category in self.blip_prompts:
            if sequence in self.blip_prompts[category]:
                prompt = random.choice(self.blip_prompts[category][sequence])

        if prompt == "":
            print("WARN: could not find prompt in blip prompts for", category, sequence)

        return prompt

    @cached(cache=LRUCache(maxsize=2048), key=lambda *args: hashkey(args[1], args[2]))
    def load_frame_data(self, category, index):
        # load frame data
        data = self.category_dict[category]["dataset"][index]
        if not isinstance(data.image_rgb, torch.Tensor):
            data.image_rgb = torch.from_numpy(data.image_rgb)

        return data

    @cached(cache=LRUCache(maxsize=2048), key=lambda *args: hashkey(args[1], args[2]))
    def load_mask(self, file_name, mask_folder):
        # load the mask_augmentation data (only mask to speed up loading)
        with open(os.path.join(mask_folder, f"masked_{file_name}.png"), "rb") as f:
            mask = Image.open(f)
            mask = np.array(mask)
            mask = torch.from_numpy(mask)

        return mask

    def __getitem__(self, idx):
        category, sequence, dataset_indices = self.get_frames(idx)
        root = os.path.join(self.config.co3d_root, category, sequence)

        # construct output
        output = {
            "images": [],
            "intensity_stats": [],
            "pose": [],
            "K": [],
            "root": str(root),
            "file_names": [],
        }

        # get prompt
        output["prompt"] = (
            self.get_prompt_for_category(category)
            if not self.config.batch.use_blip_prompt
            else self.get_blip_prompt(category, sequence)
        )

        # prepare additional outputs
        add_mask_fg = self.config.batch.mask_foreground or self.config.batch.crop == "foreground" or self.config.dataset_args.load_masks
        if add_mask_fg:
            output["foreground_mask"] = []
            output["foreground_prob"] = []

        # add each frame
        for frame_idx, dataset_index in enumerate(dataset_indices):
            # load data (with caching)
            frame_data = self.load_frame_data(category, dataset_index)
            image_rgb = frame_data.image_rgb
            file_name = Path(frame_data.image_path).stem

            # load mask foreground
            if add_mask_fg:
                fg_prob = frame_data.fg_probability[0]
                fg_mask = torch.zeros_like(fg_prob)
                thr = 0.2
                while fg_mask.sum() <= 1.0:
                    fg_mask = fg_prob > thr
                    thr -= 0.05
                fg_mask = ~fg_mask
                if self.config.batch.mask_foreground:
                    image_rgb[:, fg_mask] = 1.0  # white background

            # get bbox (only needed for first frame, since same sequence within each batch)
            if frame_idx == 0:
                if not self.config.batch.load_recentered:
                    if frame_data.sequence_point_cloud is not None:
                        output["bbox"] = frame_data.sequence_point_cloud.get_bounding_boxes().transpose(1, 2).squeeze(0)
                else:
                    bbox_file = os.path.join(root, "bbox_recentered.npy")
                    with open(bbox_file, "rb") as f:
                        output["bbox"] = torch.from_numpy(np.load(f))

            # resize / crop images
            src_hw = list(image_rgb.shape[1:])
            target_hw = [self.config.batch.image_height, self.config.batch.image_width]
            if target_hw[0] == -1:
                target_hw[0] = int(target_hw[1] * src_hw[0] / src_hw[1])
            if target_hw[1] == -1:
                target_hw[1] = int(target_hw[0] * src_hw[1] / src_hw[0])
            adjusted_size = adjust_crop_size(src_hw, target_hw)
            if self.config.batch.crop == "resize":
                if target_hw != src_hw:
                    # only resize if we really have to
                    image_rgb = interpolate(image_rgb[None], size=target_hw, mode="bilinear").squeeze()
                    if add_mask_fg:
                        fg_mask = (
                            interpolate(fg_mask[None, None].float(), size=target_hw, mode="nearest").squeeze().bool()
                        )
                        fg_prob = (
                            interpolate(fg_prob[None, None], size=target_hw, mode="bilinear").squeeze()
                        )

                pose_world2cam, K = self.convert_camera(
                    frame_data.camera,
                    source_image_hw=src_hw,
                    target_image_hw=target_hw,
                )

            elif self.config.batch.crop == "random":
                crop_params = RandomCrop.get_params(image_rgb, adjusted_size)
                image_rgb = crop(image_rgb, *crop_params)
                image_rgb = interpolate(image_rgb[None], size=target_hw, mode="bilinear").squeeze()
                if add_mask_fg:
                    fg_mask = crop(fg_mask, *crop_params)
                    fg_mask = interpolate(fg_mask[None, None].float(), size=target_hw, mode="nearest").squeeze().bool()
                    fg_prob = crop(fg_prob, *crop_params)
                    fg_prob = interpolate(fg_prob[None, None], size=target_hw, mode="bilinear").squeeze()

                pose_world2cam, K = self.convert_camera(
                    frame_data.camera,
                    source_image_hw=src_hw,
                    target_image_hw=target_hw,
                    crop_params=crop_params,
                )

            elif self.config.batch.crop == "foreground":
                mask_to_crop = mask if self.config.batch.crop == "mask" else ~fg_mask
                crop_params = get_crop_around_mask(mask_to_crop, *adjusted_size)
                image_rgb = crop(image_rgb, *crop_params)
                image_rgb = interpolate(image_rgb[None], size=target_hw, mode="bilinear").squeeze()
                if add_mask_fg:
                    fg_mask = crop(fg_mask, *crop_params)
                    fg_mask = interpolate(fg_mask[None, None].float(), size=target_hw, mode="nearest").squeeze().bool()
                    fg_prob = crop(fg_prob, *crop_params)
                    fg_prob = interpolate(fg_prob[None, None], size=target_hw, mode="bilinear").squeeze()

                pose_world2cam, K = self.convert_camera(
                    frame_data.camera,
                    source_image_hw=src_hw,
                    target_image_hw=target_hw,
                    crop_params=crop_params,
                )
            elif self.config.batch.crop == "center":
                foreground_center = torch.tensor([src_hw[0] // 2, src_hw[1] // 2])
                square_size = min(src_hw[0] // 2, src_hw[1] // 2)
                top = foreground_center[0] - square_size
                left = foreground_center[1] - square_size
                crop_params = (top, left, square_size * 2, square_size * 2)

                image_rgb = crop(image_rgb, *crop_params)
                image_rgb = interpolate(image_rgb[None], size=target_hw, mode="bilinear").squeeze()
                if add_mask_fg:
                    fg_mask = crop(fg_mask, *crop_params)
                    fg_mask = interpolate(fg_mask[None, None].float(), size=target_hw, mode="nearest").squeeze().bool()
                    fg_prob = crop(fg_prob, *crop_params)
                    fg_prob = interpolate(fg_prob[None, None], size=target_hw, mode="bilinear").squeeze()

                pose_world2cam, K = self.convert_camera(
                    frame_data.camera,
                    source_image_hw=src_hw,
                    target_image_hw=target_hw,
                    crop_params=crop_params,
                )
            else:
                raise NotImplementedError("config.batch.crop", self.config.batch.crop)

            # convert to needed output ranges
            image_rgb = image_rgb * 2.0 - 1.0  # StableDiffusion expects images in [-1, 1]

            # update pose if we want to load the recentered ones
            # these poses are already converted to opencv convention, so we do not need to call self.convert_camera on them again
            # also, the poses are independent of the image-cropping (only affects intrinsics), so we can load these poses as-is and use them.
            if self.config.batch.load_recentered:
                recentered_pose_file = os.path.join(
                    root, "recentered_poses_world2cam", f"pose_world2cam_{file_name}.npy"
                )
                with open(recentered_pose_file, "rb") as f:
                    recentered_pose = torch.from_numpy(np.load(f))
                    pose_world2cam = recentered_pose

            output["images"].append(image_rgb)
            output["file_names"].append(file_name)
            output["pose"].append(pose_world2cam.squeeze(0))
            output["K"].append(K.squeeze(0))

            # get intensity stats
            var, mean = torch.var_mean(image_rgb)
            intensity_stats = torch.stack([mean, var], dim=0)
            output["intensity_stats"].append(intensity_stats)

            if add_mask_fg:
                output["foreground_mask"].append(fg_mask)
                output["foreground_prob"].append(fg_prob)

        # update cams to perfect spherical ones
        if self.config.batch.replace_pose_with_spherical_start_phi > -400:
            # intrinsics uses focal length of last frame and centered principal point (i.e., we hard-code to crop in the center)
            # Alternative: set the principal point of the dimension that gets cropped to the other dimensions value (== simulate perfect center)
            crop_top = (src_hw[0] - adjusted_size[0]) // 2
            crop_left = (src_hw[1] - adjusted_size[1]) // 2
            crop_params = (crop_top, crop_left, adjusted_size[0], adjusted_size[1])
            _, K = self.convert_camera(
                frame_data.camera,
                source_image_hw=src_hw,
                target_image_hw=target_hw,
                crop_params=crop_params,
            )
            output["K"] = [K.squeeze(0)] * len(output["K"])

            # poses are sampled s.t. the camera center lies on a sphere of specified (radius, theta) and are sampled uniformly across (phi) with maximum degrees
            # all poses look at the origin (0, 0, 0)
            output["pose"] = sample_uniform_poses_on_sphere(
                n_samples=len(output["pose"]),
                radius=self.config.batch.replace_pose_with_spherical_radius,
                start_phi=self.config.batch.replace_pose_with_spherical_start_phi,
                end_phi=self.config.batch.replace_pose_with_spherical_end_phi,
                theta=self.config.batch.replace_pose_with_spherical_theta,
                endpoint=self.config.batch.replace_pose_with_spherical_phi_endpoint
            )

            # intensity uses the same from the last frame for all images
            output["intensity_stats"] = [intensity_stats] * len(output["intensity_stats"])

        # convert lists to tensors
        for k in output.keys():
            if isinstance(output[k], list) and isinstance(output[k][0], torch.Tensor):
                output[k] = torch.stack(output[k])

        return output


class CO3DDreamboothDataset(Dataset):
    DEFAULT_MIN_FOCAL: float = 565.0
    DEFAULT_MAX_FOCAL: float = 590.0
    DEFAULT_CX: float = 255.5
    DEFAULT_CY: float = 255.5
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512

    @staticmethod
    def get_intrinsics_for_image_size(width: int = 512, height: int = 512):
        # get scaling factors
        scaling_factor_h = height / CO3DDreamboothDataset.DEFAULT_HEIGHT
        scaling_factor_w = width / CO3DDreamboothDataset.DEFAULT_WIDTH

        # fx
        min_focal_x = CO3DDreamboothDataset.DEFAULT_MIN_FOCAL * scaling_factor_w
        max_focal_x = CO3DDreamboothDataset.DEFAULT_MAX_FOCAL * scaling_factor_w

        # fy
        min_focal_y = CO3DDreamboothDataset.DEFAULT_MIN_FOCAL * scaling_factor_h
        max_focal_y = CO3DDreamboothDataset.DEFAULT_MAX_FOCAL * scaling_factor_h

        # We assume opencv-convention ((0, 0) refers to the center of the top-left pixel and (-0.5, -0.5) is the top-left corner of the image-plane):
        # We need to scale the principal offset with the 0.5 add/sub like here.
        # see this for explanation: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        cx = (CO3DDreamboothDataset.DEFAULT_CX + 0.5) * scaling_factor_w - 0.5
        cy = (CO3DDreamboothDataset.DEFAULT_CY + 0.5) * scaling_factor_h - 0.5

        return min_focal_x, max_focal_x, min_focal_y, max_focal_y, cx, cy

    def __init__(
        self,
        co3d_root: str,
        selected_categories: Tuple[str, ...] = (),
        pose_spherical_min_radius: float = 2.0,
        pose_spherical_max_radius: float = 4.0,
        pose_spherical_min_theta: float = 30.0,
        pose_spherical_max_theta: float = 90.0,
        width: int = 512,
        height: int = 512,
    ):
        self.pose_spherical_min_radius = pose_spherical_min_radius
        self.pose_spherical_max_radius = pose_spherical_max_radius
        self.pose_spherical_min_theta = pose_spherical_min_theta
        self.pose_spherical_max_theta = pose_spherical_max_theta
        self.width = width
        self.height = height
        self.min_focal_x, self.max_focal_x, self.min_focal_y, self.max_focal_y, self.cx, self.cy = CO3DDreamboothDataset.get_intrinsics_for_image_size(width, height)
        self.bbox = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)

        root = os.path.join(co3d_root, "dreambooth_prior_preservation_dataset")
        categories = os.listdir(root)
        self.images = []
        self.prompts = []
        for c in categories:
            if len(selected_categories) > 0 and c not in selected_categories:
                continue

            category_path = os.path.join(root, c)
            if not os.path.isdir(category_path):
                continue

            files = os.listdir(os.path.join(root, c))
            images = [os.path.join(category_path, f) for f in files if ".jpg" in f]
            image_to_prompt = [os.path.join(category_path, f) for f in files if f == "image_to_prompt.json"][0]
            with open(image_to_prompt, "r") as f:
                image_to_prompt = json.load(f)
            prompts = [image_to_prompt[os.path.splitext(os.path.basename(img))[0]] for img in images]
            assert len(images) == len(prompts)

            self.images.extend(images)
            self.prompts.extend(prompts)

        print("Constructed CO3DDreamboothDataset. Total #images:", len(self.images), "Selected Categories:", selected_categories, "Image Size:", height, "x", width)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get random pose
        pose = sample_random_poses_on_sphere(
            n_samples=1,
            radius=random.uniform(self.pose_spherical_min_radius, self.pose_spherical_max_radius),
            end_phi=360.0,
            theta=random.uniform(self.pose_spherical_min_theta, self.pose_spherical_max_theta)
        )

        # get random intrinsics
        w = random.uniform(0.0, 1.0)
        K = np.eye(3, 3, dtype=np.float32)
        K[0, 0] = w * self.min_focal_x + (1 - w) * self.max_focal_x
        K[0, 2] = self.cx
        K[1, 1] = w * self.min_focal_y + (1 - w) * self.max_focal_y
        K[1, 2] = self.cy
        K = K[None, ...]
        K = torch.from_numpy(K)

        # get next image/prompt
        prompt = self.prompts[idx]
        image_file = self.images[idx]
        with open(image_file, "rb") as f:
            image = Image.open(f)
            if image.width != self.width or image.height != self.height:
                image = image.resize((self.width, self.height), Image.BILINEAR)
            image = np.array(image)
            image = image / 255.0
            image = image * 2.0 - 1.0
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            # get intensity stats
            var, mean = torch.var_mean(image)
            intensity_stats = torch.stack([mean, var], dim=0)
            intensity_stats = intensity_stats.unsqueeze(0)

        # build output
        item = {
            "images": image,
            "intensity_stats": intensity_stats,
            "prompt": prompt,
            "pose": pose,
            "K": K,
            "bbox": self.bbox.clone(),
        }

        return item
