# Copyright (c) Meta Platforms, Inc. and affiliates.
# This file is partially based on the diffusers library, which licensed the code under the following license:

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from .custom_unet_2d_blocks import (
    CrossFrameInExistingAttnDownBlock2D,
    CrossFrameInExistingAttnUpBlock2D,
    UNetMidBlock2DCrossFrameInExistingAttn,
    get_last_layer_in_existing,
    get_cross_frame_parameters_in_existing,
    get_other_parameters_in_existing,
)
from .custom_attention_processor import expand_batch, collapse_batch

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEFAULT_DOWN_BLOCK_TYPES: Tuple[str] = (
    CrossAttnDownBlock2D.__name__,
    CrossAttnDownBlock2D.__name__,
    CrossAttnDownBlock2D.__name__,
    DownBlock2D.__name__,
)

DEFAULT_MID_BLOCK_TYPE: str = UNetMidBlock2DCrossAttn.__name__
CROSS_FRAME_MID_BLOCK_TYPE: str = UNetMidBlock2DCrossFrameInExistingAttn.__name__

DEFAULT_UP_BLOCK_TYPES: Tuple[str] = (
    UpBlock2D.__name__,
    CrossAttnUpBlock2D.__name__,
    CrossAttnUpBlock2D.__name__,
    CrossAttnUpBlock2D.__name__,
)


def get_down_block_types(n_cross_frame_blocks: int = 3):
    n_total_blocks = len(DEFAULT_DOWN_BLOCK_TYPES)
    n_unchanged_blocks = n_total_blocks - n_cross_frame_blocks
    assert 0 <= n_unchanged_blocks <= n_total_blocks

    # change the blocks from *bottom to top* (e.g. from bottleneck outwards)
    blocks = [k for k in DEFAULT_DOWN_BLOCK_TYPES]
    start_index = n_total_blocks - 1 - (n_unchanged_blocks > 0)
    for i in range(start_index, start_index - n_cross_frame_blocks, -1):
        blocks[i] = CrossFrameInExistingAttnDownBlock2D.__name__

    return blocks


def get_up_block_types(n_cross_frame_blocks: int = 3):
    n_total_blocks = len(DEFAULT_UP_BLOCK_TYPES)
    n_unchanged_blocks = n_total_blocks - n_cross_frame_blocks
    assert 0 <= n_unchanged_blocks <= n_total_blocks

    # change the blocks from *top to bottom* (e.g. from bottleneck outwards)
    blocks = [k for k in DEFAULT_UP_BLOCK_TYPES]
    start_index = n_unchanged_blocks > 0
    for i in range(start_index, start_index + n_cross_frame_blocks):
        blocks[i] = CrossFrameInExistingAttnUpBlock2D.__name__

    return blocks


def get_mid_block_type(use_cross_frame_block: bool = False):
    if use_cross_frame_block:
        return CROSS_FRAME_MID_BLOCK_TYPE
    else:
        return DEFAULT_MID_BLOCK_TYPE


@dataclass
class UNet3DConsistencyOutput(BaseOutput):
    """
    The output of [`UNet2DConditionCrossFrameInExistingAttnModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
        depth_loss ('List[torch.FloatTensor]): optional list of depth-losses calculated in intermediate layers of the model.
    """

    unet_sample: torch.FloatTensor = None
    rendered_depth: List[torch.FloatTensor] = None
    rendered_mask: List[torch.FloatTensor] = None
    unproj_reproj_kwargs: Dict[str, Any] = None


class UNet2DConditionCrossFrameInExistingAttnModel(ModelMixin, ConfigMixin):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `(CrossAttnDownBlock2D.__name__, CrossAttnDownBlock2D.__name__, CrossAttnDownBlock2D.__name__, DownBlock2D.__name__)`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `UNetMidBlock2DCrossAttn.__name__`):
            The mid block type. Choose from `UNetMidBlock2DCrossAttn` or `UNetMidBlock2DSimpleCrossAttn`, will skip the
            mid block layer if `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `(UpBlock2D.__name__, "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to None):
            If given, the `encoder_hidden_states` and potentially other embeddings will be down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to None):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, default to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, default to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, default to `None`):
            Optional activation function to use on the time embeddings only one time before they as passed to the rest
            of the unet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str, *optional*, default to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, default to `None`):
            The dimension of `cond_proj` layer in timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            using the "projection" `class_embed_type`. Required when using the "projection" `class_embed_type`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is None, the
            `only_cross_attention` value will be used as the value for `mid_block_only_cross_attention`. Else, it will
            default to `False`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = DEFAULT_DOWN_BLOCK_TYPES,
        mid_block_type: Optional[str] = DEFAULT_MID_BLOCK_TYPE,
        up_block_types: Tuple[str] = DEFAULT_UP_BLOCK_TYPES,
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        dropout: float = 0.0,
        attention_type: str = "default",
        # new arguments
        n_input_images: int = 5,
        to_k_other_frames: int = 4,
        random_others: bool = False,
        last_layer_mode: Literal["none", "zero-conv", "alpha"] = "none",
        use_lora_in_cfa: bool = False,
        use_temb_in_lora: bool = False,
        temb_out_size: int = 10,
        pose_cond_dim=10,
        rank=4,
        network_alpha=None,
        unproj_reproj_mode: Literal["none", "only_unproj_reproj", "with_cfa"] = "none",
        num_3d_layers: int = 1,
        dim_3d_latent: int = 32,
        dim_3d_grid: int = 64,
        n_novel_images: int = 1,
        vol_rend_proj_in_mode: Literal["single", "multiple", "unet"] = "unet",
        vol_rend_proj_out_mode: Literal["single", "multiple"] = "multiple",
        vol_rend_aggregator_mode: Literal["mean", "ibrnet"] = "ibrnet",
        vol_rend_model_background: bool = False,
        vol_rend_background_grid_percentage: float = 0.5,
        vol_rend_disparity_at_inf: float = 1e-3,
    ):
        super().__init__()

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # parse unproj_reproj_mode into flags
        self.n_input_images = n_input_images
        self.unproj_reproj_mode = unproj_reproj_mode
        self.use_unproj_reproj_in_blocks = (
            unproj_reproj_mode == "only_unproj_reproj" or unproj_reproj_mode == "with_cfa"
        )
        self.use_cfa = (
            unproj_reproj_mode == "with_cfa" or unproj_reproj_mode == "none"
        )

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == DownBlock2D.__name__:
                down_block = DownBlock2D(
                    num_layers=layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                )
            elif down_block_type == CrossFrameInExistingAttnDownBlock2D.__name__:
                if cross_attention_dim[i] is None:
                    raise ValueError("cross_attention_dim must be specified for CrossFrameInExistingAttnDownBlock2D")
                # transformer_layers_per_block, num_attention_heads instead of attn_num_head_channels
                down_block = CrossFrameInExistingAttnDownBlock2D(
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                    ### new args
                    n_input_images=n_input_images,
                    to_k_other_frames=to_k_other_frames,
                    random_others=random_others,
                    last_layer_mode=last_layer_mode,
                    use_lora_in_cfa=use_lora_in_cfa,
                    use_temb_in_lora=use_temb_in_lora,
                    temb_size=time_embed_dim,
                    temb_out_size=temb_out_size,
                    pose_cond_dim=pose_cond_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                    use_cfa=self.use_cfa,
                    use_unproj_reproj=self.use_unproj_reproj_in_blocks,
                    num_3d_layers=num_3d_layers,
                    dim_3d_latent=dim_3d_latent,
                    dim_3d_grid=dim_3d_grid,
                    n_novel_images=n_novel_images,
                    vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                    vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                    vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                    vol_rend_model_background=vol_rend_model_background,
                    vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                    vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                )
            elif down_block_type == CrossAttnDownBlock2D.__name__:
                down_block = CrossAttnDownBlock2D(
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                    attention_type=attention_type,
                )
            else:
                raise ValueError(f"unknown down_block_type : {down_block_type}")

            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == UNetMidBlock2DCrossFrameInExistingAttn.__name__:
            self.mid_block = UNetMidBlock2DCrossFrameInExistingAttn(
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                dropout=dropout,
                ### new args
                n_input_images=n_input_images,
                to_k_other_frames=to_k_other_frames,
                random_others=random_others,
                last_layer_mode=last_layer_mode,
                use_lora_in_cfa=use_lora_in_cfa,
                use_temb_in_lora=use_temb_in_lora,
                temb_size=time_embed_dim,
                temb_out_size=temb_out_size,
                pose_cond_dim=pose_cond_dim,
                rank=rank,
                network_alpha=network_alpha,
                use_cfa=self.use_cfa,
                use_unproj_reproj=self.use_unproj_reproj_in_blocks,
                num_3d_layers=num_3d_layers,
                dim_3d_latent=dim_3d_latent,
                dim_3d_grid=dim_3d_grid,
                n_novel_images=n_novel_images,
                vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                vol_rend_model_background=vol_rend_model_background,
                vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
            )
        elif mid_block_type == UNetMidBlock2DCrossAttn.__name__:
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                dropout=dropout,
                attention_type=attention_type,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == UpBlock2D.__name__:
                up_block = UpBlock2D(
                    num_layers=reversed_layers_per_block[i] + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                )
            elif up_block_type == "CrossAttnUpBlock2D":
                if reversed_cross_attention_dim[i] is None:
                    raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
                up_block = CrossAttnUpBlock2D(
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                    attention_type=attention_type,
                )
            elif up_block_type == CrossFrameInExistingAttnUpBlock2D.__name__:
                if reversed_cross_attention_dim[i] is None:
                    raise ValueError("cross_attention_dim must be specified for CrossFrameInExistingAttnUpBlock2D")
                up_block = CrossFrameInExistingAttnUpBlock2D(
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dropout=dropout,
                    ### new args
                    n_input_images=n_input_images,
                    to_k_other_frames=to_k_other_frames,
                    random_others=random_others,
                    last_layer_mode=last_layer_mode,
                    use_lora_in_cfa=use_lora_in_cfa,
                    use_temb_in_lora=use_temb_in_lora,
                    temb_size=time_embed_dim,
                    temb_out_size=temb_out_size,
                    pose_cond_dim=pose_cond_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                    use_cfa=self.use_cfa,
                    use_unproj_reproj=self.use_unproj_reproj_in_blocks,
                    num_3d_layers=num_3d_layers,
                    dim_3d_latent=dim_3d_latent,
                    dim_3d_grid=dim_3d_grid,
                    n_novel_images=n_novel_images,
                    vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                    vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                    vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                    vol_rend_model_background=vol_rend_model_background,
                    vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                    vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                )
            else:
                raise ValueError(f"unknown up_block_type : {up_block_type}")

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    @classmethod
    def from_source(
        cls,
        src: UNet2DConditionModel,
        load_weights: bool = True,
        down_block_types: Tuple[str] = (
            CrossFrameInExistingAttnDownBlock2D.__name__,
            CrossFrameInExistingAttnDownBlock2D.__name__,
            CrossFrameInExistingAttnDownBlock2D.__name__,
            DownBlock2D.__name__,
        ),
        mid_block_type: Optional[str] = UNetMidBlock2DCrossFrameInExistingAttn.__name__,
        up_block_types: Tuple[str] = (
            UpBlock2D.__name__,
            CrossFrameInExistingAttnUpBlock2D.__name__,
            CrossFrameInExistingAttnUpBlock2D.__name__,
            CrossFrameInExistingAttnUpBlock2D.__name__,
        ),
        n_input_images: int = 5,
        to_k_other_frames: int = 4,
        random_others: bool = False,
        last_layer_mode: Literal["none", "zero-conv", "alpha"] = "none",
        use_lora_in_cfa: bool = False,
        use_temb_in_lora: bool = False,
        temb_out_size: int = 10,
        pose_cond_dim=10,
        rank=4,
        network_alpha=None,
        unproj_reproj_mode: Literal["none", "only_unproj_reproj", "with_cfa"] = "none",
        num_3d_layers: int = 1,
        dim_3d_latent: int = 32,
        dim_3d_grid: int = 64,
        n_novel_images: int = 1,
        vol_rend_proj_in_mode: Literal["single", "multiple", "unet"] = "unet",
        vol_rend_proj_out_mode: Literal["single", "multiple"] = "multiple",
        vol_rend_aggregator_mode: Literal["mean", "ibrnet"] = "ibrnet",
        vol_rend_model_background: bool = False,
        vol_rend_background_grid_percentage: float = 0.5,
        vol_rend_disparity_at_inf: float = 1e-3,
    ):
        r"""
        Instantiate UNet2DConditionCrossFrameAttnModel class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the UNet2DConditionCrossFrameAttnModel where applicable. Note that all configuration options are also
                copied where applicable.
        """

        # replace parts of src config that should be different
        config = {**src.config}
        config["down_block_types"] = down_block_types
        config["mid_block_type"] = mid_block_type
        config["up_block_types"] = up_block_types

        # add parts to src config that are novel
        config["n_input_images"] = n_input_images
        config["to_k_other_frames"] = to_k_other_frames
        config["random_others"] = random_others
        config["last_layer_mode"] = last_layer_mode
        config["use_lora_in_cfa"] = use_lora_in_cfa
        config["use_temb_in_lora"] = use_temb_in_lora
        config["temb_out_size"] = temb_out_size
        config["pose_cond_dim"] = pose_cond_dim
        config["rank"] = rank
        config["network_alpha"] = network_alpha
        config["unproj_reproj_mode"] = unproj_reproj_mode
        config["num_3d_layers"] = num_3d_layers
        config["dim_3d_latent"] = dim_3d_latent
        config["dim_3d_grid"] = dim_3d_grid
        config["n_novel_images"] = n_novel_images
        config["vol_rend_proj_in_mode"] = vol_rend_proj_in_mode
        config["vol_rend_proj_out_mode"] = vol_rend_proj_out_mode
        config["vol_rend_aggregator_mode"] = vol_rend_aggregator_mode
        config["vol_rend_model_background"] = vol_rend_model_background
        config["vol_rend_background_grid_percentage"] = vol_rend_background_grid_percentage
        config["vol_rend_disparity_at_inf"] = vol_rend_disparity_at_inf
        config["_class_name"] = cls.__name__

        # construct from new config
        cf_unet = cls(**config)

        if load_weights:
            cf_unet.conv_in.load_state_dict(src.conv_in.state_dict())
            cf_unet.time_proj.load_state_dict(src.time_proj.state_dict())
            cf_unet.time_embedding.load_state_dict(src.time_embedding.state_dict())
            cf_unet.conv_out.load_state_dict(src.conv_out.state_dict())

            if cf_unet.class_embedding is not None:
                cf_unet.class_embedding.load_state_dict(src.class_embedding.state_dict())

            if cf_unet.conv_norm_out is not None:
                cf_unet.conv_norm_out.load_state_dict(src.conv_norm_out.state_dict())

            if cf_unet.encoder_hid_proj is not None:
                cf_unet.encoder_hid_proj.load_state_dict(src.encoder_hid_proj.state_dict())

            if hasattr(cf_unet, "add_embedding") and cf_unet.add_embedding is not None:
                cf_unet.add_embedding.load_state_dict(src.add_embedding.state_dict())

            if hasattr(cf_unet, "add_time_proj") and cf_unet.add_time_proj is not None:
                cf_unet.add_time_proj.load_state_dict(src.add_time_proj.state_dict())

            # down_blocks
            for i, b in enumerate(src.down_blocks):
                if down_block_types[i] == CrossFrameInExistingAttnDownBlock2D.__name__:
                    # construct block from source by copying those weights that should be re-used
                    # initialize other weights randomly
                    if isinstance(b, CrossAttnDownBlock2D):
                        new_block = CrossFrameInExistingAttnDownBlock2D.from_source(
                            b,
                            load_weights=load_weights,
                            n_input_images=n_input_images,
                            to_k_other_frames=to_k_other_frames,
                            random_others=random_others,
                            last_layer_mode=last_layer_mode,
                            use_lora_in_cfa=use_lora_in_cfa,
                            use_temb_in_lora=use_temb_in_lora,
                            temb_size=cf_unet.time_embedding.linear_1.out_features,
                            temb_out_size=temb_out_size,
                            pose_cond_dim=pose_cond_dim,
                            rank=rank,
                            network_alpha=network_alpha,
                            use_cfa=cf_unet.use_cfa,
                            use_unproj_reproj=cf_unet.use_unproj_reproj_in_blocks,
                            num_3d_layers=num_3d_layers,
                            dim_3d_latent=dim_3d_latent,
                            dim_3d_grid=dim_3d_grid,
                            n_novel_images=n_novel_images,
                            vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                            vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                            vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                            vol_rend_model_background=vol_rend_model_background,
                            vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                            vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                        )
                        cf_unet.down_blocks[i].load_state_dict(new_block.state_dict())
                    else:
                        # TODO: support also construct from DownBlock2D, e.g. does not have any attention --> initialize them from scratch as well
                        raise NotImplementedError(f"cannot construct from_source for block of type: {type(b)}")
                else:
                    cf_unet.down_blocks[i].load_state_dict(b.state_dict())

            # mid_block
            if mid_block_type == UNetMidBlock2DCrossFrameInExistingAttn.__name__:
                if isinstance(src.mid_block, UNetMidBlock2DCrossAttn):
                    new_block = UNetMidBlock2DCrossFrameInExistingAttn.from_source(
                        src.mid_block,
                        load_weights=load_weights,
                        n_input_images=n_input_images,
                        to_k_other_frames=to_k_other_frames,
                        random_others=random_others,
                        last_layer_mode=last_layer_mode,
                        use_lora_in_cfa=use_lora_in_cfa,
                        use_temb_in_lora=use_temb_in_lora,
                        temb_size=cf_unet.time_embedding.linear_1.out_features,
                        temb_out_size=temb_out_size,
                        pose_cond_dim=pose_cond_dim,
                        rank=rank,
                        network_alpha=network_alpha,
                        use_cfa=cf_unet.use_cfa,
                        use_unproj_reproj=cf_unet.use_unproj_reproj_in_blocks,
                        num_3d_layers=num_3d_layers,
                        dim_3d_latent=dim_3d_latent,
                        dim_3d_grid=dim_3d_grid,
                        n_novel_images=n_novel_images,
                        vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                        vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                        vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                        vol_rend_model_background=vol_rend_model_background,
                        vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                        vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                    )
                    cf_unet.mid_block.load_state_dict(new_block.state_dict())
                else:
                    raise NotImplementedError(f"cannot construct from_source for block of type: {type(src.mid_block)}")
            else:
                cf_unet.mid_block.load_state_dict(src.mid_block.state_dict())

            # up_blocks
            for i, b in enumerate(src.up_blocks):
                if up_block_types[i] == CrossFrameInExistingAttnUpBlock2D.__name__:
                    if isinstance(b, CrossAttnUpBlock2D):
                        new_block = CrossFrameInExistingAttnUpBlock2D.from_source(
                            b,
                            load_weights=load_weights,
                            n_input_images=n_input_images,
                            to_k_other_frames=to_k_other_frames,
                            random_others=random_others,
                            last_layer_mode=last_layer_mode,
                            use_lora_in_cfa=use_lora_in_cfa,
                            use_temb_in_lora=use_temb_in_lora,
                            temb_size=cf_unet.time_embedding.linear_1.out_features,
                            temb_out_size=temb_out_size,
                            pose_cond_dim=pose_cond_dim,
                            rank=rank,
                            network_alpha=network_alpha,
                            use_cfa=cf_unet.use_cfa,
                            use_unproj_reproj=cf_unet.use_unproj_reproj_in_blocks,
                            num_3d_layers=num_3d_layers,
                            dim_3d_latent=dim_3d_latent,
                            dim_3d_grid=dim_3d_grid,
                            n_novel_images=n_novel_images,
                            vol_rend_proj_in_mode=vol_rend_proj_in_mode,
                            vol_rend_proj_out_mode=vol_rend_proj_out_mode,
                            vol_rend_aggregator_mode=vol_rend_aggregator_mode,
                            vol_rend_model_background=vol_rend_model_background,
                            vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                            vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                        )
                        cf_unet.up_blocks[i].load_state_dict(new_block.state_dict())
                    else:
                        # TODO: support also construct from UpBlock2D, e.g. does not have any attention --> initialize them from scratch as well
                        raise NotImplementedError(f"cannot construct from_source for block of type: {type(b)}")
                else:
                    cf_unet.up_blocks[i].load_state_dict(b.state_dict())

        return cf_unet

    def get_cross_frame_params(self, vol_rend_mode: Literal["with", "without", "only"] = "with"):
        params = []

        # down_blocks
        for b in self.down_blocks:
            if isinstance(b, CrossFrameInExistingAttnDownBlock2D):
                params.extend(get_cross_frame_parameters_in_existing(b, vol_rend_mode=vol_rend_mode))

        # mid_block
        if isinstance(self.mid_block, UNetMidBlock2DCrossFrameInExistingAttn):
            params.extend(get_cross_frame_parameters_in_existing(self.mid_block, vol_rend_mode=vol_rend_mode))

        # up_blocks
        for b in self.up_blocks:
            if isinstance(b, CrossFrameInExistingAttnUpBlock2D):
                params.extend(get_cross_frame_parameters_in_existing(b, vol_rend_mode=vol_rend_mode))

        return params

    def get_last_layer_params(self):
        params = []

        # down_blocks
        for b in self.down_blocks:
            if isinstance(b, CrossFrameInExistingAttnDownBlock2D):
                params.extend(get_last_layer_in_existing(b))

        # mid_block
        if isinstance(self.mid_block, UNetMidBlock2DCrossFrameInExistingAttn):
            params.extend(get_last_layer_in_existing(self.mid_block))

        # up_blocks
        for b in self.up_blocks:
            if isinstance(b, CrossFrameInExistingAttnUpBlock2D):
                params.extend(get_last_layer_in_existing(b))

        return params

    def get_params_without_volume_rendering(self):
        params = []
        for n, p in self.named_parameters():
            if "volume_renderer" not in n:
                params.append(p)
        return params

    def get_old_params(self):
        params = [
            *list(self.conv_in.parameters()),
            *list(self.time_proj.parameters()),
            *list(self.time_embedding.parameters()),
            *list(self.conv_out.parameters()),
        ]

        if self.class_embedding is not None:
            params.extend(list(self.class_embedding.parameters()))

        if self.conv_norm_out is not None:
            params.extend(list(self.conv_norm_out.parameters()))

        if self.encoder_hid_proj is not None:
            params.extend(list(self.encoder_hid_proj.parameters()))

        if hasattr(self, "add_embedding") and self.add_embedding is not None:
            params.extend(list(self.add_embedding.parameters()))

        # down_blocks
        for b in self.down_blocks:
            if isinstance(b, CrossFrameInExistingAttnDownBlock2D):
                params.extend(get_other_parameters_in_existing(b))
            else:
                params.extend(list(b.parameters()))

        # mid_block
        if isinstance(self.mid_block, UNetMidBlock2DCrossFrameInExistingAttn):
            params.extend(get_other_parameters_in_existing(self.mid_block))
        else:
            params.extend(list(self.mid_block.parameters()))

        # up_blocks
        for b in self.up_blocks:
            if isinstance(b, CrossFrameInExistingAttnUpBlock2D):
                params.extend(get_other_parameters_in_existing(b))
            else:
                params.extend(list(b.parameters()))

        return params

    @property
    def transformer_blocks(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of BasicTransformerBlocks: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_transformer_blocks(
            name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]
        ):
            if isinstance(module, BasicTransformerBlock):
                processors[f"{name}"] = module

            for sub_name, child in module.named_children():
                fn_recursive_add_transformer_blocks(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_transformer_blocks(name, module, processors)

        return processors

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (CrossFrameInExistingAttnDownBlock2D, DownBlock2D, CrossFrameInExistingAttnUpBlock2D, UpBlock2D)
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        n_images_per_batch: int = 0,
        n_known_images: int = 0,
    ) -> Union[UNet3DConsistencyOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            encoder_attention_mask (`torch.Tensor`):
                (batch, sequence_length) cross-attention mask, applied to encoder_hidden_states. True = keep, False =
                discard. Mask will be converted into a bias, which adds large negative values to attention scores
                corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            added_cond_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified includes additonal conditions that can be used for additonal time
                embeddings or encoder hidden states projections. See the configurations `encoder_hid_dim_type` and
                `addition_embed_type` for more information.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        input_sample = sample.clone()
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep.clone()
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        if n_known_images > 0:
            timesteps = timesteps.clone()
            timesteps = expand_batch(timesteps, n_images_per_batch)
            timesteps[:, 0:n_known_images] = 0
            timesteps = collapse_batch(timesteps)
            #print(f"set timestep 0 for n_known_images={n_known_images} -->", timesteps, timesteps.shape)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)

        # 2. pre-process
        sample = self.conv_in(sample)
        if cross_attention_kwargs is not None:
            proc_ca_kwargs = {k: v for k, v in cross_attention_kwargs.items()}
            proc_ca_kwargs["temb"] = emb  # add it here for cfa timestep conditioning
            unproj_reproj_kwargs = proc_ca_kwargs.pop("unproj_reproj_kwargs", None)
            if unproj_reproj_kwargs is not None:
                # add it here for proj layer timestep conditioning
                unproj_reproj_kwargs["temb"] = emb
                unproj_reproj_kwargs["timestep"] = timesteps
        else:
            proc_ca_kwargs = None
            unproj_reproj_kwargs = None

        # 3. down

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                if isinstance(downsample_block, CrossFrameInExistingAttnDownBlock2D):
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=proc_ca_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        unproj_reproj_kwargs=unproj_reproj_kwargs,
                        **additional_residuals,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=proc_ca_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if isinstance(self.mid_block, UNetMidBlock2DCrossFrameInExistingAttn):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=proc_ca_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    unproj_reproj_kwargs=unproj_reproj_kwargs,
                )
            else:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=proc_ca_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if isinstance(upsample_block, CrossFrameInExistingAttnUpBlock2D):
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=proc_ca_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        unproj_reproj_kwargs=unproj_reproj_kwargs,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=proc_ca_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if unproj_reproj_kwargs is not None:
            rendered_depth_output = unproj_reproj_kwargs.pop("rendered_depth", None)
            rendered_mask_output = unproj_reproj_kwargs.pop("rendered_mask", None)
        else:
            rendered_depth_output = None
            rendered_mask_output = None

        if not return_dict:
            return (
                sample,
                rendered_depth_output,
                rendered_mask_output,
                unproj_reproj_kwargs,
            )

        return UNet3DConsistencyOutput(
            unet_sample=sample,
            rendered_depth=rendered_depth_output,
            rendered_mask=rendered_mask_output,
            unproj_reproj_kwargs=unproj_reproj_kwargs,
        )
