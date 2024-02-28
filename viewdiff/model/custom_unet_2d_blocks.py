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
from typing import Any, Dict, Optional, Tuple, Union, Literal

import torch
from torch import nn

from diffusers.utils import is_torch_version
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.resnet import (
    Downsample2D,
    Upsample2D,
    ResnetBlock2D,
)
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn, CrossAttnUpBlock2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock

from .custom_transformer_2d import Transformer2DSelfAttnCrossAttnCrossFrameAttnModel
from .custom_attention import BasicTransformerWithCrossFrameAttentionBlock

from .projection.layer import UnprojReprojLayer


def load_transformer_blocks(
    target: nn.ModuleList,
    src: nn.ModuleList,
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
    for tgt_a, src_a in zip(target, src):
        if isinstance(tgt_a, Transformer2DSelfAttnCrossAttnCrossFrameAttnModel) and isinstance(
            src_a, Transformer2DModel
        ):
            for tgt_b, src_b in zip(tgt_a.transformer_blocks, src_a.transformer_blocks):
                if isinstance(tgt_b, BasicTransformerWithCrossFrameAttentionBlock) and isinstance(
                    src_b, BasicTransformerBlock
                ):
                    new_transformer_block = BasicTransformerWithCrossFrameAttentionBlock.from_source(
                        src_b,
                        last_layer_mode=last_layer_mode,
                        n_input_images=n_input_images,
                        to_k_other_frames=to_k_other_frames,
                        random_others=random_others,
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
                    tgt_b.load_state_dict(new_transformer_block.state_dict())


class CrossFrameInExistingAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
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
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DSelfAttnCrossAttnCrossFrameAttnModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        ### new args
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
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    @classmethod
    def from_source(
        cls,
        src: CrossAttnDownBlock2D,
        load_weights: bool = True,
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
        if src.attentions[0].transformer_blocks[0].only_cross_attention:
            raise NotImplementedError("we assume only_cross_attention=False and double_self_attention=False")

        def get_temb_channels(src_block: CrossAttnDownBlock2D):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                temb_channels = r.norm1.linear.in_features
            elif r.time_embedding_norm == "spatial":
                temb_channels = r.norm1.conv_y.in_channels
            elif r.time_emb_proj is not None:
                temb_channels = r.time_emb_proj.in_features
            else:
                temb_channels = None

            return temb_channels

        def get_resnet_eps_num_groups(src_block: CrossAttnDownBlock2D):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                return r.norm1.eps, r.norm1.num_groups
            elif r.time_embedding_norm == "spatial":
                return 1e-6, 32  # not important anyways, so return default values
            else:
                return r.norm1.eps, r.norm1.num_groups

        def get_resnet_act_fn(src_block: CrossAttnDownBlock2D):
            r = src_block.resnets[0].nonlinearity
            if isinstance(r, nn.SiLU):
                return "swish"  # or silu
            elif isinstance(r, nn.Mish):
                return "mish"
            elif isinstance(r, nn.GELU):
                return "gelu"
            raise ValueError(r)

        resnet_eps, resnet_groups = get_resnet_eps_num_groups(src)
        block = cls(
            in_channels=src.resnets[0].in_channels,
            out_channels=src.resnets[0].out_channels,
            temb_channels=get_temb_channels(src),
            dropout=src.resnets[0].dropout.p,
            num_layers=len(src.resnets),
            transformer_layers_per_block=len(src.attentions[0].transformer_blocks),
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=src.resnets[0].time_embedding_norm,
            resnet_act_fn=get_resnet_act_fn(src),
            resnet_groups=resnet_groups,
            resnet_pre_norm=src.resnets[0].pre_norm,
            num_attention_heads=src.num_attention_heads,
            cross_attention_dim=src.attentions[0].transformer_blocks[0].attn2.to_k.in_features,
            output_scale_factor=src.resnets[0].output_scale_factor,
            downsample_padding=src.downsamplers[0].padding if src.downsamplers is not None else None,
            add_downsample=src.downsamplers is not None,
            dual_cross_attention=isinstance(src.attentions[0], DualTransformer2DModel),
            use_linear_projection=src.attentions[0].use_linear_projection,
            only_cross_attention=False,
            upcast_attention=src.attentions[0].transformer_blocks[0].attn1.upcast_attention,
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

        if load_weights:
            # load state_dict for everything except the cross-frame-attention layers
            block.attentions.load_state_dict(src.attentions.state_dict(), strict=False)

            # load cross-frame-attention layers
            load_transformer_blocks(
                block.attentions,
                src.attentions,
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

            block.resnets.load_state_dict(src.resnets.state_dict())

            if src.downsamplers is not None:
                block.downsamplers.load_state_dict(src.downsamplers.state_dict())

        return block

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        unproj_reproj_kwargs: Dict[str, Any] = None,
    ):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    unproj_reproj_kwargs,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    unproj_reproj_kwargs=unproj_reproj_kwargs,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossFrameInExistingAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
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

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DSelfAttnCrossAttnCrossFrameAttnModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
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
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    @classmethod
    def from_source(
        cls,
        src: UNetMidBlock2DCrossAttn,
        load_weights: bool = True,
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
        if src.attentions[0].transformer_blocks[0].only_cross_attention:
            raise NotImplementedError("we assume only_cross_attention=False and double_self_attention=False")

        def get_temb_channels(src_block: UNetMidBlock2DCrossAttn):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                temb_channels = r.norm1.linear.in_features
            elif r.time_embedding_norm == "spatial":
                temb_channels = r.norm1.conv_y.in_channels
            elif r.time_emb_proj is not None:
                temb_channels = r.time_emb_proj.in_features
            else:
                temb_channels = None

            return temb_channels

        def get_resnet_eps_num_groups(src_block: UNetMidBlock2DCrossAttn):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                return r.norm1.eps, r.norm1.num_groups
            elif r.time_embedding_norm == "spatial":
                return 1e-6, 32  # not important anyways, so return default values
            else:
                return r.norm1.eps, r.norm1.num_groups

        def get_resnet_act_fn(src_block: UNetMidBlock2DCrossAttn):
            r = src_block.resnets[0].nonlinearity
            if isinstance(r, nn.SiLU):
                return "swish"  # or silu
            elif isinstance(r, nn.Mish):
                return "mish"
            elif isinstance(r, nn.GELU):
                return "gelu"
            raise ValueError(r)

        resnet_eps, resnet_groups = get_resnet_eps_num_groups(src)
        block = cls(
            in_channels=src.resnets[0].in_channels,
            temb_channels=get_temb_channels(src),
            dropout=src.resnets[0].dropout.p,
            num_layers=len(src.attentions),
            transformer_layers_per_block=len(src.attentions[0].transformer_blocks),
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=src.resnets[0].time_embedding_norm,
            resnet_act_fn=get_resnet_act_fn(src),
            resnet_groups=resnet_groups,
            resnet_pre_norm=src.resnets[0].pre_norm,
            num_attention_heads=src.num_attention_heads,
            cross_attention_dim=src.attentions[0].transformer_blocks[0].attn2.to_k.in_features,
            output_scale_factor=src.resnets[0].output_scale_factor,
            dual_cross_attention=isinstance(src.attentions[0], DualTransformer2DModel),
            use_linear_projection=src.attentions[0].use_linear_projection,
            upcast_attention=src.attentions[0].transformer_blocks[0].attn1.upcast_attention,
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

        if load_weights:
            # load state_dict for everything except the cross-frame-attention layers
            block.attentions.load_state_dict(src.attentions.state_dict(), strict=False)

            # load cross-frame-attention layers
            load_transformer_blocks(
                block.attentions,
                src.attentions,
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
            block.resnets.load_state_dict(src.resnets.state_dict())

        return block

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        unproj_reproj_kwargs: Dict[str, Any] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                unproj_reproj_kwargs=unproj_reproj_kwargs,
                return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossFrameInExistingAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
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
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DSelfAttnCrossAttnCrossFrameAttnModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
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
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    @classmethod
    def from_source(
        cls,
        src: CrossAttnUpBlock2D,
        load_weights: bool = True,
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
        if src.attentions[0].transformer_blocks[0].only_cross_attention:
            raise NotImplementedError("we assume only_cross_attention=False and double_self_attention=False")

        def get_in_prev_channels(src_block: CrossAttnUpBlock2D):
            if len(src_block.resnets) > 0:
                return (
                    src_block.resnets[-1].in_channels - src_block.resnets[-1].out_channels,
                    src_block.resnets[0].in_channels - src_block.resnets[0].out_channels,
                )
            else:
                raise NotImplementedError()

        def get_temb_channels(src_block: CrossAttnUpBlock2D):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                temb_channels = r.norm1.linear.in_features
            elif r.time_embedding_norm == "spatial":
                temb_channels = r.norm1.conv_y.in_channels
            elif r.time_emb_proj is not None:
                temb_channels = r.time_emb_proj.in_features
            else:
                temb_channels = None

            return temb_channels

        def get_resnet_eps_num_groups(src_block: CrossAttnUpBlock2D):
            r = src_block.resnets[0]
            if r.time_embedding_norm == "ada_group":
                return r.norm1.eps, r.norm1.num_groups
            elif r.time_embedding_norm == "spatial":
                return 1e-6, 32  # not important anyways, so return default values
            else:
                return r.norm1.eps, r.norm1.num_groups

        def get_resnet_act_fn(src_block: CrossAttnUpBlock2D):
            r = src_block.resnets[0].nonlinearity
            if isinstance(r, nn.SiLU):
                return "swish"  # or silu
            elif isinstance(r, nn.Mish):
                return "mish"
            elif isinstance(r, nn.GELU):
                return "gelu"
            raise ValueError(r)

        resnet_eps, resnet_groups = get_resnet_eps_num_groups(src)
        in_channels, prev_output_channel = get_in_prev_channels(src)
        block = cls(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=src.resnets[0].out_channels,
            temb_channels=get_temb_channels(src),
            dropout=src.resnets[0].dropout.p,
            num_layers=len(src.resnets),
            transformer_layers_per_block=len(src.attentions[0].transformer_blocks),
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=src.resnets[0].time_embedding_norm,
            resnet_act_fn=get_resnet_act_fn(src),
            resnet_groups=resnet_groups,
            resnet_pre_norm=src.resnets[0].pre_norm,
            num_attention_heads=src.num_attention_heads,
            cross_attention_dim=src.attentions[0].transformer_blocks[0].attn2.to_k.in_features,
            output_scale_factor=src.resnets[0].output_scale_factor,
            add_upsample=src.upsamplers is not None,
            dual_cross_attention=isinstance(src.attentions[0], DualTransformer2DModel),
            use_linear_projection=src.attentions[0].use_linear_projection,
            only_cross_attention=False,
            upcast_attention=src.attentions[0].transformer_blocks[0].attn1.upcast_attention,
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

        if load_weights:
            # load state_dict for everything except the cross-frame-attention layers
            block.attentions.load_state_dict(src.attentions.state_dict(), strict=False)

            # load cross-frame-attention layers
            load_transformer_blocks(
                block.attentions,
                src.attentions,
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
            block.resnets.load_state_dict(src.resnets.state_dict())

            if src.upsamplers is not None:
                block.upsamplers.load_state_dict(src.upsamplers.state_dict())

        return block

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        unproj_reproj_kwargs: Dict[str, Any] = None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    unproj_reproj_kwargs,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    unproj_reproj_kwargs=unproj_reproj_kwargs,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


def get_cross_frame_parameters_in_existing(
    block: Union[
        CrossFrameInExistingAttnDownBlock2D, CrossFrameInExistingAttnUpBlock2D, UNetMidBlock2DCrossFrameInExistingAttn
    ],
    vol_rend_mode: Literal["with", "without", "only"] = "with",
):
    params = []
    for b in block.attentions:
        if isinstance(b, Transformer2DSelfAttnCrossAttnCrossFrameAttnModel):
            params.extend(b.get_cross_frame_parameters(vol_rend_mode=vol_rend_mode))
    return params


def get_last_layer_in_existing(
    block: Union[
        CrossFrameInExistingAttnDownBlock2D, CrossFrameInExistingAttnUpBlock2D, UNetMidBlock2DCrossFrameInExistingAttn
    ]
):
    params = []
    for b in block.attentions:
        if isinstance(b, Transformer2DSelfAttnCrossAttnCrossFrameAttnModel):
            params.extend(b.get_last_layer_params())
    return params


def get_other_parameters_in_existing(
    block: Union[
        CrossFrameInExistingAttnDownBlock2D, CrossFrameInExistingAttnUpBlock2D, UNetMidBlock2DCrossFrameInExistingAttn
    ]
):
    params = [*list(block.resnets.parameters())]

    if isinstance(block, CrossFrameInExistingAttnDownBlock2D) and block.downsamplers is not None:
        params.extend(list(block.downsamplers.parameters()))

    if isinstance(block, CrossFrameInExistingAttnUpBlock2D) and block.upsamplers is not None:
        params.extend(list(block.upsamplers.parameters()))

    for b in block.attentions:
        if isinstance(b, Transformer2DSelfAttnCrossAttnCrossFrameAttnModel):
            params.extend(b.get_other_parameters())
        else:
            params.extend(list(b.parameters()))

    return params


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
