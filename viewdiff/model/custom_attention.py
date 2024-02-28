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
import math
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import (
    BasicTransformerBlock,
    AdaLayerNorm,
    AdaLayerNormZero,
    GEGLU,
    GELU,
    ApproximateGELU,
)

from .custom_attention_processor import CrossFrameAttentionProcessor2_0, expand_batch, collapse_batch

from .projection.layer import UnprojReprojLayer

from diffusers.models.controlnet import zero_module


@maybe_allow_in_graph
class BasicTransformerWithCrossFrameAttentionBlock(BasicTransformerBlock):
    r"""
    A Transformer block that does self-attention, cross-frame-attention, unproj-reproj, and optionally cross-attention to separate conditioning (e.g. text).

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        # parent arguments
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # new arguments
        last_layer_mode: Literal["none", "zero-conv", "alpha", "no_residual_connection"] = "none",
        n_input_images: int = 5,
        to_k_other_frames: int = 0,
        random_others: bool = False,
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
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_elementwise_affine,
            norm_type,
            final_dropout,
        )

        self.use_cfa = use_cfa
        self.use_unproj_reproj = use_unproj_reproj
        self.n_input_images = n_input_images
        self.last_layer_mode = last_layer_mode

        if use_cfa or use_unproj_reproj:
            if last_layer_mode == "alpha":
                # start with disabling cross-frame attention, i.e. sigmoid(-10) ~= 0
                self.alpha = torch.nn.Parameter(torch.zeros(1) - 10)
            elif last_layer_mode == "zero-conv":
                # start with disabling cross-frame attention, i.e. zero-conv from ControlNet in the end
                self.cfa_controlnet_block = zero_module(nn.Linear(dim, dim))

            if self.use_ada_layer_norm:
                self.norm_cf = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm_cf = AdaLayerNormZero(dim, num_embeds_ada_norm)
            else:
                self.norm_cf = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

            if use_cfa:
                self.attn_cf = Attention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                    upcast_attention=upcast_attention,
                    processor=CrossFrameAttentionProcessor2_0(
                        n_input_images=n_input_images,
                        to_k_other_frames=to_k_other_frames,
                        with_self_attention=False,
                        random_others=random_others,
                        use_lora_in_cfa=use_lora_in_cfa,
                        use_temb_in_lora=use_temb_in_lora,
                        temb_size=temb_size,
                        temb_out_size=temb_out_size,
                        pose_cond_dim=pose_cond_dim,
                        rank=rank,
                        network_alpha=network_alpha,
                    ),
                )

            if use_unproj_reproj:
                if dim == 320:
                    scale = 1
                elif dim == 640:
                    scale = 2
                elif dim == 1280:
                    scale = 4
                else:
                    raise NotImplementedError("unexpected dim", dim)

                self.unproj_reproj_layer = UnprojReprojLayer(
                    latent_channels=dim,
                    num_3d_layers=num_3d_layers,
                    dim_3d_latent=dim_3d_latent,
                    dim_3d_grid=dim_3d_grid // scale,
                    n_novel_images=n_novel_images,
                    proj_in_mode=vol_rend_proj_in_mode,
                    proj_out_mode=vol_rend_proj_out_mode,
                    aggregator_mode=vol_rend_aggregator_mode,
                    vol_rend_model_background=vol_rend_model_background,
                    vol_rend_background_grid_percentage=vol_rend_background_grid_percentage,
                    vol_rend_disparity_at_inf=vol_rend_disparity_at_inf,
                    use_temb=use_temb_in_lora,
                    temb_dim=temb_size
                )

    @classmethod
    def from_source(
        cls,
        src: BasicTransformerBlock,
        last_layer_mode: Literal["none", "zero-conv", "alpha"] = "none",
        n_input_images: int = 5,
        to_k_other_frames: int = 0,
        random_others: bool = False,
        load_existing_weights: bool = True,
        copy_weights_from_self_attention_to_cross_frame_attention: bool = False,
        use_lora_in_cfa: bool = False,
        use_temb_in_lora: bool = False,
        temb_size: int = 1280,
        temb_out_size: int = 8,
        pose_cond_dim=8,
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
        if src.only_cross_attention:
            raise NotImplementedError("we assume only_cross_attention=False and double_self_attention=False")

        def get_act_fn_str(block: BasicTransformerBlock) -> str:
            act_fn = block.ff.net[0]
            if isinstance(act_fn, GELU):
                if act_fn.approximate == "tanh":
                    activation_fn = "gelu-approximate"
                else:
                    activation_fn = "gelu"
            elif isinstance(act_fn, GEGLU):
                activation_fn = "geglu"
            elif isinstance(act_fn, ApproximateGELU):
                activation_fn = "geglu-approximate"
            else:
                raise ValueError("unsupported act_fn type", act_fn)

            return activation_fn

        def get_norm_type_and_num_embeds_ada_norm(block: BasicTransformerBlock):
            if isinstance(block.norm1, AdaLayerNorm):
                num_embeds_ada_norm = block.norm1.emb.num_embeddings
                norm_type = "ada_norm"
            elif isinstance(block.norm1, AdaLayerNormZero):
                num_embeds_ada_norm = block.norm1.emb.class_embedder.num_classes
                norm_type = "ada_norm_zero"
            else:
                num_embeds_ada_norm = None
                norm_type = "layer_norm"

            return num_embeds_ada_norm, norm_type

        # construct block
        num_embeds_ada_norm, norm_type = get_norm_type_and_num_embeds_ada_norm(src)
        extended_block = cls(
            dim=src.attn1.to_q.in_features,
            num_attention_heads=src.attn1.heads,
            attention_head_dim=src.attn1.to_q.out_features // src.attn1.heads,
            dropout=src.attn1.to_out[-1].p,
            cross_attention_dim=src.attn2.to_k.in_features,
            activation_fn=get_act_fn_str(src),
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=src.attn1.to_q.bias is not None,
            only_cross_attention=False,
            double_self_attention=False,
            upcast_attention=src.attn1.upcast_attention,
            norm_elementwise_affine=src.norm3.elementwise_affine,
            norm_type=norm_type,
            final_dropout=isinstance(src.ff.net[-1], torch.nn.Dropout),
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

        if load_existing_weights:
            extended_block.norm1.load_state_dict(src.norm1.state_dict())
            extended_block.attn1.load_state_dict(src.attn1.state_dict())
            extended_block.norm2.load_state_dict(src.norm2.state_dict())
            extended_block.attn2.load_state_dict(src.attn2.state_dict())
            extended_block.norm3.load_state_dict(src.norm3.state_dict())
            extended_block.ff.load_state_dict(src.ff.state_dict())

        if copy_weights_from_self_attention_to_cross_frame_attention and use_cfa:
            src_modules = [src.norm1, src.attn1]
            target_modules = [extended_block.norm_cf, extended_block.attn_cf]
            for src, target in zip(src_modules, target_modules):
                for name, param in src.named_parameters():
                    parts = name.split(".")  # e.g. splits to_q.weight
                    modules = parts[:-1]  # e.g. to_q
                    name = parts[-1]  # e.g. weight
                    curr_target = target
                    for m in modules:
                        curr_target = getattr(curr_target, m)  # moves from attn_cf to attn_cf.to_q
                    setattr(curr_target, name, torch.nn.Parameter(param.data.clone()))

        return extended_block

    def get_other_parameters(self):
        return [
            *list(self.norm1.parameters()),
            *list(self.attn1.parameters()),
            *list(self.norm2.parameters()),
            *list(self.attn2.parameters()),
            *list(self.norm3.parameters()),
            *list(self.ff.parameters()),
        ]

    def get_cross_frame_parameters(self, vol_rend_mode: Literal["with", "without", "only"] = "with"):
        params = []
        if self.use_cfa or self.use_unproj_reproj:
            if vol_rend_mode != "only":
                params = self.get_last_layer_params()
                params.extend(list(self.norm_cf.parameters()))
                if self.use_cfa:
                    params.extend(list(self.attn_cf.parameters()))
            if self.use_unproj_reproj:
                if vol_rend_mode == "only" or vol_rend_mode == "with":
                    params.extend(self.unproj_reproj_layer.get_volume_renderer_params())
                if vol_rend_mode == "without" or vol_rend_mode == "with":
                    params.extend(self.unproj_reproj_layer.get_other_params())
        return params

    def get_last_layer_params(self):
        params = []
        if self.use_cfa or self.use_unproj_reproj:
            if self.last_layer_mode == "alpha":
                params.append(self.alpha)
            elif self.last_layer_mode == "zero-conv":
                params.extend(list(self.cfa_controlnet_block.parameters()))
        return params

    def unproj_reproj(self, kwargs, hidden_state: torch.Tensor):
        # reshape to image-dim (N, C, h, w)
        N, hw, C = hidden_state.shape
        h = w = math.floor(math.sqrt(hw))
        latents = hidden_state.reshape(N, h, w, C)
        latents = latents.permute(0, 3, 1, 2)

        # reshape to batches of n_images: (batches_of_frames, self.n_input_images, C, h, w)
        latents = expand_batch(latents, self.n_input_images)
        pose = expand_batch(kwargs["pose"], self.n_input_images)
        K = expand_batch(kwargs["K"], self.n_input_images)
        timestep = expand_batch(kwargs["timestep"], self.n_input_images) if kwargs["timestep"] is not None else None
        temb = expand_batch(kwargs["temb"], self.n_input_images) if kwargs["temb"] is not None else None

        # do unproj
        unproj_output, unproj_reproj_mask, unproj_reproj_depth = self.unproj_reproj_layer(
            latents=latents,
            pose=pose,
            K=K,
            orig_hw=kwargs["orig_hw"],
            timestep=timestep,
            temb=temb,
            bbox=kwargs["bbox"],
            deactivate_view_dependent_rendering=kwargs.pop("deactivate_view_dependent_rendering", False)
        )

        # save rendered mask & depth for logging --> already convert to cpu
        if "rendered_depth" not in kwargs:
            kwargs["rendered_depth"] = []
        kwargs["rendered_depth"].append(unproj_reproj_depth.detach().cpu())
        if "rendered_mask" not in kwargs:
            kwargs["rendered_mask"] = []
        kwargs["rendered_mask"].append(unproj_reproj_mask.detach().cpu())

        # reshape back to batch_size (N, C, h, w)
        unproj_output = collapse_batch(unproj_output)

        # reshape to hidden_state dim (N, hw, C)
        unproj_output = unproj_output.permute(0, 2, 3, 1)
        unproj_output = unproj_output.reshape(N, hw, C)

        return unproj_output

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        unproj_reproj_kwargs: Dict[str, Any] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Cross-Frame-Attention and Unproj-Reproj
        if self.use_cfa or self.use_unproj_reproj:
            # normalize hidden_states
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm_cf(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm_cf(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm_cf(hidden_states)

            if self.use_cfa:
                # perform cross-frame-attention
                cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
                attn_output = self.attn_cf(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                # skip cross-frame-attention
                attn_output = norm_hidden_states

            if self.use_unproj_reproj:
                # Unproj-Reproj (after cross-frame-attention or stand-alone)
                attn_output = self.unproj_reproj(unproj_reproj_kwargs, attn_output)

            # normalize
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # add to hidden_states
            if self.last_layer_mode == "alpha":
                w = torch.sigmoid(self.alpha)
                hidden_states = w * attn_output.float() + (1 - w) * hidden_states.float()
            elif self.last_layer_mode == "zero-conv":
                attn_output = self.cfa_controlnet_block(attn_output)
                hidden_states = attn_output + hidden_states
            elif self.last_layer_mode == "no_residual_connection":
                hidden_states = attn_output
                # add dummy 0 values if the model has these layers actually defined
                # this way they still receive gradients and DDP does not complain
                # TODO better: delete params, construct new optimizer without them
                if hasattr(self, "alpha"):
                    hidden_states = hidden_states + 0 * self.alpha
                if hasattr(self, "cfa_controlnet_block"):
                    hidden_states = hidden_states + 0 * self.cfa_controlnet_block(attn_output)
            elif self.last_layer_mode == "none":
                hidden_states = attn_output + hidden_states
            else:
                raise NotImplementedError("last_layer_mode", self.last_layer_mode)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
