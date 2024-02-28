# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Dict, Tuple, Literal, Union
import itertools
from dataclasses import dataclass

import torch

from transformers import CLIPTokenizer
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from .custom_unet_2d_condition import UNet2DConditionCrossFrameInExistingAttnModel
from .custom_attention import BasicTransformerWithCrossFrameAttentionBlock
from .custom_attention_processor import (
    CrossFrameAttentionProcessor2_0,
    PoseCondLoRAAttnProcessor2_0,
    CustomAttnProcessor2_0,
)

from .projection.layer import UnprojReprojLayer


@dataclass
class ModelConfig:
    """Arguments for model."""

    n_input_images: int = 5
    """How many images are expected as input in parallel."""

    pose_cond_mode: Literal["none", "ca", "sa-ca", "sa-ca-cfa", "ca-cfa"] = "none"
    """How to add the pose conditioning to the attention layers of the U-Net.
        "none": do not add any pose-conditioning.
        "ca": add it only to cross-attention (to text) layers.
        "sa-ca": add it to self-attention and cross-attention layers.
        "sa-ca-cfa: add it to self-attention, cross-attention, and cross-frame-attention layers.
        "ca-cfa": add it to cross-attention and cross-frame-attention layers.
    """

    pose_cond_coord_space: Literal["absolute", "relative-first"] = "absolute"
    """How to encode the pose conditioning.
        "absolute": encode poses like in Zero123, but relative to the world-space-origin.
        "relative-first": encode poses like in Zero123, where the first pose in each batch is the conditioning pose.
    """

    pose_cond_lora_rank: int = 4
    """rank of the lora matrices used for the pose conditioning."""

    pose_cond_dim: int = 10
    """How many things we provide as pose conditioning. Currently it is 4 values for extrinsics (as in Zero123) + 4 values for intrinsics + 2 values for mean/var image intensity."""

    conditioning_dropout_prob: float = 0.1
    """Conditioning dropout probability. Drops out the conditionings (text prompt) used in training."""

    use_ema: bool = False
    """Whether to use EMA model."""

    enable_xformers_memory_efficient_attention: bool = False
    """Whether or not to use xformers."""

    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."""


@dataclass
class CrossFrameAttentionConfig:
    mode: Literal["none", "pretrained", "add_in_existing_block"] = "none"
    """How to add cross-frame attention to the U-Net
        "none": do not add cross-frame attention.
        "pretrained": add cross-frame attention in the existing self-attention layers of the U-Net.
        "add_in_existing_block": add cross-frame attention by inserting a new attention layer in each BasicTransformerBlock.
    """

    n_cfa_down_blocks: int = 3
    """How many of the down_blocks in the U-Net should be replaced with blocks that contain cross-frame-attention."""

    n_cfa_up_blocks: int = 3
    """How many of the up_blocks in the U-Net should be replaced with blocks that contain cross-frame-attention."""

    no_cfa_in_mid_block: bool = False
    """If we should not use cross-frame-attention in the mid_block. Default behaviour: use it as soon as mode!=none. This forcibly overrides the setting."""

    to_k_other_frames: int = 4
    """How many of the other images in a batch to use as key/value."""

    with_self_attention: bool = False
    """If the key/value of the query image should be appended. Only relevant for mode='pretrained'."""

    random_others: bool = False
    """If True, will select the k_other_frames randomly, otherwise sequentially."""

    last_layer_mode: Literal["none", "zero-conv", "alpha", "no_residual_connection"] = "zero-conv"
    """How to add the contributions of cross-frame-attention.
        'none': directly add them to the residual connection in the ViT blocks.
        'zero-conv': add them with the zero-conv idea from ControlNet.
        'alpha': add them with the alpha idea from VideoLDM.
        'no_residual_connection': do not use residual connection, instead use the output directly."""

    unproj_reproj_mode: Literal["none", "only_unproj_reproj", "with_cfa"] = "none"
    """How to use unproj_reproj as layer in the model.
        "none": do not use the layer.
        "only_unproj_reproj": use the layer instead of cross-frame-attention.
        "with_cfa": use the layer in addition to cross-frame-attention.
    """

    num_3d_layers: int = 1
    """how many 3D layers to use in the 3D CNN."""

    dim_3d_latent: int = 32
    """dimension of the 3D latent features to process with the 3D CNN."""

    dim_3d_grid: int = 64
    """dimension of the 3D voxel grid to process with the 3D CNN."""

    vol_rend_proj_in_mode: Literal["single", "multiple", "unet"] = "unet"
    """How to convert hidden states into features for our proj-module.
        'single': a single 1x1 convolution is used
        'multiple': 2x{Conv_3x3, Relu} is used
        'unet': a 3-layer unet is used"""

    vol_rend_proj_out_mode: Literal["single", "multiple"] = "multiple"
    """How to convert projected features into hidden states after our proj-module. We always use a linear combination of foreground and background.
    For the background, we always use a single Conv_1x1. For the foreground we use either:
    'single': a single 1x1 convolution
    'multiple': 2x{Conv_1x1, Relu} followed by Conv_1x1 (==learned non-linear scale function)"""

    vol_rend_aggregator_mode: Literal["mean", "ibrnet"] = "ibrnet"
    """How to convert per-frame voxel grids into a joint voxel-grid across all frames.
    'mean': mean across each voxel in each frame.
    'ibrnet': use the aggregator as proposed in IBRNet (https://arxiv.org/abs/2102.13090)"""

    vol_rend_model_background: bool = False
    """If the volume-rendering module should model the background."""

    vol_rend_background_grid_percentage: float = 0.5
    """How much of the voxel grid should be used for background, if we need to model it."""

    vol_rend_disparity_at_inf: float = 0.5
    """The value for disparity_at_inf argument for the volume-rendering module."""

    n_novel_images: int = 1
    """during unprojection, how many images to consider novel. They are not used for unprojection, but only for reprojection (novel-view-synthesis)."""

    use_temb_cond: bool = False
    """If True, will use timestep embedding as additional input for cross-frame-attention and projection layers. Useful to process a batch with images of different timesteps."""


def get_attn_processor_infos(unet: UNet2DConditionModel):
    infos = []
    for name, proc in unet.attn_processors.items():
        is_cross_attention = "attn2" in name
        is_cfa_attention = "attn_cf" in name or isinstance(proc, CrossFrameAttentionProcessor2_0)
        is_self_attention = "attn1" in name
        cross_attention_dim = unet.config.cross_attention_dim if is_cross_attention else None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        infos.append(
            {
                "name": name,
                "is_cross_attention": is_cross_attention,
                "is_cfa_attention": is_cfa_attention,
                "is_self_attention": is_self_attention,
                "cross_attention_dim": cross_attention_dim,
                "name": name,
                "hidden_size": hidden_size,
                "proc": proc,
            }
        )

    return infos


def replace_self_attention_with_cross_frame_attention(
    unet: UNet2DConditionModel,
    n_input_images: int = 5,
    to_k_other_frames: int = 0,
    with_self_attention: bool = True,
    random_others: bool = False,
    use_lora_in_cfa: bool = False,
    use_temb_in_lora: bool = False,
    temb_out_size: int = 8,
    pose_cond_dim=8,
    rank=4,
    network_alpha=None,
):
    """Changes the self-attention layers in the U-net with cross-frame attention as defined in ~CrossFrameAttentionProcessor2_0.

    Args:
        unet (UNet2DConditionModel): the model where the self_attention layers should be replaced
        n_input_images (int, optional): How many images are in one batch. Defaults to 5.
        to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
        with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
        random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
    """
    infos = get_attn_processor_infos(unet)

    unet_attn_procs = {}
    unet_attn_parameters = []
    for info in infos:
        if info["is_self_attention"]:
            module = CrossFrameAttentionProcessor2_0(
                n_input_images=n_input_images,
                to_k_other_frames=to_k_other_frames,
                with_self_attention=with_self_attention,
                random_others=random_others,
                use_lora_in_cfa=use_lora_in_cfa,
                use_temb_in_lora=use_temb_in_lora,
                temb_size=unet.time_embedding.linear_1.out_features,
                temb_out_size=temb_out_size,
                hidden_size=info["hidden_size"],
                pose_cond_dim=pose_cond_dim,
                rank=rank,
                network_alpha=network_alpha,
            )
            unet_attn_procs[info["name"]] = module
            if hasattr(module, "parameters"):
                unet_attn_parameters.extend(module.parameters())
        elif info["is_cfa_attention"]:
            unet_attn_procs[info["name"]] = info["proc"]  # is already fixed to support kwargs
        else:
            assert isinstance(info["proc"], AttnProcessor2_0)
            unet_attn_procs[info["name"]] = CustomAttnProcessor2_0()  # to fix unsupported kwargs in original

    unet.set_attn_processor(unet_attn_procs)

    return unet_attn_procs, unet_attn_parameters


def add_pose_cond_to_attention_layers(
    unet: UNet2DConditionModel, rank: int = 4, pose_cond_dim: int = 8, only_cross_attention: bool = False
):
    infos = get_attn_processor_infos(unet)

    unet_attn_procs = {}
    unet_attn_parameters = []
    for info in infos:
        if info["is_cfa_attention"]:
            unet_attn_procs[info["name"]] = info["proc"]  # is already fixed to support kwargs
        elif only_cross_attention and not info["is_cross_attention"]:
            unet_attn_procs[info["name"]] = CustomAttnProcessor2_0()  # to fix unsupported kwargs in original
        else:
            module = PoseCondLoRAAttnProcessor2_0(
                hidden_size=info["hidden_size"],
                cross_attention_dim=info["cross_attention_dim"],
                rank=rank,
                pose_cond_dim=pose_cond_dim,
            )
            unet_attn_procs[info["name"]] = module
            unet_attn_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_attn_procs)

    return unet_attn_procs, unet_attn_parameters


def update_cross_frame_attention_config(
    unet: UNet2DConditionCrossFrameInExistingAttnModel,
    n_input_images: int = 5,
    to_k_other_frames: int = 0,
    with_self_attention: bool = True,
    random_others: bool = False,
    change_self_attention_layers: bool = False,
):
    """Changes the transformer-block layers in the U-net with cross-frame attention to process n_input_images.

    Args:
        unet (UNet2DConditionModel): the model where the n_input_images should be replaced
        n_input_images (int, optional): How many images are in one batch. Defaults to 5.
        to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
        with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
        random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
    """
    unet.n_input_images = n_input_images
    for k, block in unet.transformer_blocks.items():
        attn = None
        if isinstance(block, BasicTransformerWithCrossFrameAttentionBlock):
            block.n_input_images = n_input_images
            if change_self_attention_layers:
                attn = block.attn1
            elif block.use_cfa:
                attn = block.attn_cf
        else:
            if change_self_attention_layers:
                attn = block.attn1

        if attn is not None:
            processor = attn.processor
            assert isinstance(processor, CrossFrameAttentionProcessor2_0)
            processor.set_config(
                n_input_images=n_input_images,
                to_k_other_frames=to_k_other_frames,
                with_self_attention=with_self_attention,
                random_others=random_others,
            )


def update_last_layer_mode(
    unet: UNet2DConditionCrossFrameInExistingAttnModel,
    last_layer_mode: Literal["none", "zero-conv", "alpha", "no_residual_connection"] = "zero-conv",
):
    """Changes the transformer-block layers in the U-net with cross-frame attention to process n_input_images.

    Args:
        unet (UNet2DConditionModel): the model where the n_input_images should be replaced
        n_input_images (int, optional): How many images are in one batch. Defaults to 5.
        to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
        with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
        random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
    """
    unet.register_to_config(last_layer_mode=last_layer_mode)
    for k, block in unet.transformer_blocks.items():
        if isinstance(block, BasicTransformerWithCrossFrameAttentionBlock):
            block.last_layer_mode = last_layer_mode
            if last_layer_mode == "alpha":
                assert hasattr(
                    block, "alpha"
                ), f"Cannot change last_layer_mode to {last_layer_mode} because the block was not initialized with this mode."
            if last_layer_mode == "zero-conv":
                assert hasattr(
                    block, "cfa_controlnet_block"
                ), f"Cannot change last_layer_mode to {last_layer_mode} because the block was not initialized with this mode."


def update_vol_rend_inject_noise_sigma(
    unet: UNet2DConditionCrossFrameInExistingAttnModel,
    inject_noise_sigma: float = 1.0,
):
    """Changes the transformer-block layers in the U-net with cross-frame attention to process n_input_images.

    Args:
        unet (UNet2DConditionModel): the model where the n_input_images should be replaced
        n_input_images (int, optional): How many images are in one batch. Defaults to 5.
        to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
        with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
        random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
    """
    for k, block in unet.transformer_blocks.items():
        if isinstance(block, BasicTransformerWithCrossFrameAttentionBlock):
            if hasattr(block, "unproj_reproj_layer"):
                if hasattr(block.unproj_reproj_layer, "volume_renderer"):
                    block.unproj_reproj_layer.volume_renderer.inject_noise_sigma = inject_noise_sigma


def update_n_novel_images(
    unet: UNet2DConditionCrossFrameInExistingAttnModel,
    n_novel_images: int = 0,
):
    """Changes the transformer-block layers in the U-net with cross-frame attention to process n_input_images.

    Args:
        unet (UNet2DConditionModel): the model where the n_input_images should be replaced
        n_input_images (int, optional): How many images are in one batch. Defaults to 5.
        to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
        with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
        random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
    """
    for k, block in unet.transformer_blocks.items():
        if isinstance(block, BasicTransformerWithCrossFrameAttentionBlock):
            if hasattr(block, "unproj_reproj_layer"):
                block.unproj_reproj_layer.n_novel_images = n_novel_images


def collapse_prompt_to_batch_dim(prompt: List[str], k: int):
    return list(itertools.chain.from_iterable(itertools.repeat(p, k) for p in prompt))


def collapse_tensor_to_batch_dim(x: torch.Tensor):
    batch_size, k = x.shape[:2]
    return batch_size, x.view(batch_size * k, 1, *x.shape[2:]).contiguous()


def expand_tensor_to_k(x: torch.Tensor, batch_size, k):
    return x.view(batch_size, k, *x.shape[2:]).contiguous()


def expand_output_to_k(
    output, batch_size: int, k: int
):
    assert output.images.shape[1] == 1, f"can only expand when k=1, but images have shape {output.images.shape}"
    assert (
        output.images.shape[0] == batch_size * k
    ), f"want to expand to bs={batch_size}, k={k}, but images have shape {output.images.shape}"
    output.images = expand_tensor_to_k(output.images, batch_size, k)
    if hasattr(output, "image_list"):
        for i, x in enumerate(output.image_list):
            output.image_list[i] = expand_tensor_to_k(x.unsqueeze(1), batch_size, k)


def tokenize_captions(tokenizer: CLIPTokenizer, captions: List[str]):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def encode_batch(batch, vae: AutoencoderKL, dtype, K_out: int):
    N, K, C, H, W = batch["images"].shape
    images = batch["images"][:, :K_out]  # need latents/noise/pred_noise for the first K_out images

    # Get the latent embeddings for  GT images
    images = images.view(N * K_out, C, H, W).to(dtype)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    latents = latents.view(N, K_out, *latents.shape[1:]).contiguous()

    # Get the latent embeddings for masked_image
    masked_images = batch["masked_images"]
    masked_images = masked_images.view(N * K, C, H, W).to(dtype)
    masked_image_latents = vae.encode(masked_images).latent_dist.sample()
    masked_image_latents = masked_image_latents * vae.config.scaling_factor
    masked_image_latents = masked_image_latents.view(N, K, *masked_image_latents.shape[1:]).contiguous()

    # resize the mask to latents shape as we concatenate the mask to the latents
    mask = batch["masks"]
    mask = mask.view(N * K, 1, H, W)
    mask = torch.nn.functional.interpolate(mask, size=latents.shape[3:])
    mask = mask.view(N, K, *mask.shape[1:]).contiguous()

    return latents, masked_image_latents, mask


def load_latents(batch, vae: AutoencoderKL, dtype, K_out: int):
    N, K, C, H, W = batch["image_latents"].shape
    latent_parameters = batch["image_latents"][:, :K_out]  # need latents/noise/pred_noise for the first K_out images

    # Get the latent embeddings for  GT images
    latent_parameters = latent_parameters.view(N * K_out, C, H, W).to(dtype)
    latents = DiagonalGaussianDistribution(parameters=latent_parameters).sample()
    latents = latents * vae.config.scaling_factor
    latents = latents.view(N, K_out, *latents.shape[1:]).contiguous()

    # Get the latent embeddings for masked_image
    masked_image_latents_parameters = batch["masked_image_latents"]
    masked_image_latents_parameters = masked_image_latents_parameters.view(N * K, C, H, W).to(dtype)
    masked_image_latents = DiagonalGaussianDistribution(parameters=masked_image_latents_parameters).sample()
    masked_image_latents = masked_image_latents * vae.config.scaling_factor
    masked_image_latents = masked_image_latents.view(N, K, *masked_image_latents.shape[1:]).contiguous()

    # Get the mask
    mask = batch["masks"].view(N, K, 1, H, W).to(dtype).contiguous()

    return latents, masked_image_latents, mask


# adapted from https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py
def cartesian_to_spherical(xyz):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = torch.sqrt(xy + xyz[:, 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = torch.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, azimuth, z


# adapted from https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py
def get_T(target_RT: torch.Tensor, cond_RT: torch.Tensor, intrinsics: torch.Tensor = None, orig_hw: Tuple[int, int] = None, intensity_stats: torch.Tensor = None) -> torch.Tensor:
    """Get the pose conditioning tensor from target and condition pose.

    Args:
        target_RT (torch.Tensor): shape (N, 4, 4)
        cond_RT (torch.Tensor): shape (N, 4, 4)

    Returns:
        torch.Tensor: shape (N, 4)
    """

    R, T = target_RT[:, :3, :3], target_RT[:, :3, 3:4]
    T_target = -R.transpose(1, 2).bmm(T)

    R, T = cond_RT[:, :3, :3], cond_RT[:, :3, 3:4]
    T_cond = -R.transpose(1, 2).bmm(T)

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond)
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target)

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * torch.pi)
    d_z = z_target - z_cond

    pose_cond = [d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z]

    if intrinsics is not None:
        # also concatenate the intrinsics into the pose condition
        # in contrast to Zero123, we use real dataset with different camera intrinsics (not the same blender camera for all renderings)
        fx = intrinsics[:, 0, 0:1].clone()
        fy = intrinsics[:, 1, 1:2].clone()
        cx = intrinsics[:, 0, 2:3].clone()
        cy = intrinsics[:, 1, 2:3].clone()

        if orig_hw is not None:
            # make intrinsics independent from image resolution: at inference, could use different image size than during training
            fx /= orig_hw[1]
            cx /= orig_hw[1]
            fy /= orig_hw[0]
            cy /= orig_hw[0]

        pose_cond.extend([fx, fy, cx, cy])

    if intensity_stats is not None:
        mean_intensity = intensity_stats[:, 0:1]
        var_intensity = intensity_stats[:, 1:2]
        pose_cond.extend([mean_intensity, var_intensity])

    pose_cond = torch.cat(pose_cond, dim=-1)
    return pose_cond


def build_cross_attention_kwargs(
    model_config: ModelConfig,
    cfa_config: CrossFrameAttentionConfig,
    pose: torch.Tensor,
    K: torch.Tensor,
    intensity_stats: torch.Tensor,
    bbox: torch.Tensor = None,
    world_space_transform: torch.Tensor = None,
    orig_hw: Tuple[int, int] = None,
):
    cross_attention_kwargs = {}

    # add pose conditioning
    if model_config.pose_cond_mode != "none":
        if model_config.pose_cond_coord_space == "relative-first":
            origin_pose = pose[0:1].repeat(pose.shape[0], 1, 1)
        elif model_config.pose_cond_coord_space == "absolute":
            origin_pose = torch.zeros_like(pose)
            origin_pose[:, 0, 0] = 1
            origin_pose[:, 1, 1] = 1
            origin_pose[:, 2, 2] = 1
            origin_pose[:, 3, 3] = 1
        else:
            raise NotImplementedError("pose_cond_coord_space", model_config.pose_cond_coord_space)
        cross_attention_kwargs = {
            "pose_cond": get_T(
                pose,
                origin_pose,
                K,
                orig_hw,
                intensity_stats)
        }

    # add unproj kwargs
    if cfa_config.unproj_reproj_mode != "none":
        # add it to ca_kwargs s.t. we can still use the default pipelines (e.g. no new function parameter introduced)
        cross_attention_kwargs["unproj_reproj_kwargs"] = {
            "pose": pose,
            "K": K,
            "bbox": bbox,
            "world_space_transform": world_space_transform,
            "orig_hw": orig_hw,
        }

    return cross_attention_kwargs
