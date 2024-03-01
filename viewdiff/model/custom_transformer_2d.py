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
from typing import Any, Dict, Optional, Literal

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import deprecate
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformer_2d import Transformer2DModelOutput

from .custom_attention import BasicTransformerWithCrossFrameAttentionBlock

from .projection.layer import UnprojReprojLayer


class Transformer2DSelfAttnCrossAttnCrossFrameAttnModel(ModelMixin, ConfigMixin):
    """
    Transformer model with self attention, cross attention, and cross-frame attention for image-like data. Takes continuous (actual
    embeddings) inputs.
    The final layer is initialized to be zero as suggested in ControlNet paper (https://arxiv.org/pdf/2302.05543.pdf).

    First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        # new arguments
        n_input_images: int = 5,
        to_k_other_frames: int = 4,
        random_others: bool = False,
        last_layer_mode: Literal["none", "zero-conv", "alpha"] = "none",
        use_lora_in_cfa: bool = False,
        use_temb_in_lora: bool = False,
        temb_size: int = 1280,
        temb_out_size: int = 10,
        pose_cond_dim=10,
        rank=4,
        network_alpha=None,
        use_cfa: bool = True,
        use_unproj_reproj: bool = False,
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
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerWithCrossFrameAttentionBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    only_cross_attention=only_cross_attention,
                    n_input_images=n_input_images,
                    to_k_other_frames=to_k_other_frames,
                    random_others=random_others,
                    last_layer_mode=last_layer_mode,
                    use_lora_in_cfa=use_lora_in_cfa,
                    use_temb_in_lora=use_temb_in_lora,
                    temb_size=temb_size,
                    temb_out_size=temb_out_size,
                    pose_cond_dim=pose_cond_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                    use_cfa=use_cfa,
                    use_unproj_reproj=use_unproj_reproj,
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
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def get_cross_frame_parameters(self, vol_rend_mode: Literal["with", "without", "only"] = "with"):
        params = []
        for b in self.transformer_blocks:
            if isinstance(b, BasicTransformerWithCrossFrameAttentionBlock):
                params.extend(b.get_cross_frame_parameters(vol_rend_mode=vol_rend_mode))
        return params

    def get_last_layer_params(self):
        params = []
        for b in self.transformer_blocks:
            if isinstance(b, BasicTransformerWithCrossFrameAttentionBlock):
                params.extend(b.get_last_layer_params())
        return params

    def get_other_parameters(self):
        params = [
            *list(self.norm.parameters()),
            *list(self.proj_in.parameters()),
            *list(self.proj_out.parameters()),
        ]

        for b in self.transformer_blocks:
            if isinstance(b, BasicTransformerWithCrossFrameAttentionBlock):
                params.extend(b.get_other_parameters())
            else:
                params.extend(list(b.parameters()))
        return params

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        unproj_reproj_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            encoder_attention_mask ( `torch.Tensor`, *optional* ).
                Cross-attention mask, applied to encoder_hidden_states. Two formats supported:
                    Mask `(batch, sequence_length)` True = keep, False = discard. Bias `(batch, 1, sequence_length)` 0
                    = keep, -10000 = discard.
                If ndim == 2: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                unproj_reproj_kwargs=unproj_reproj_kwargs,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
