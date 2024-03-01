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
import random

import torch
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers.models.attention_processor import LoRALinearLayer


def expand_batch(x: torch.Tensor, frames_per_batch: int) -> torch.Tensor:
    n = x.shape[0]
    other_dims = x.shape[1:]
    return x.reshape(n // frames_per_batch, frames_per_batch, *other_dims)


def collapse_batch(x: torch.Tensor) -> torch.Tensor:
    n, k = x.shape[:2]
    other_dims = x.shape[2:]
    return x.reshape(n * k, *other_dims)


class CrossFrameAttentionProcessor2_0(torch.nn.Module):
    """
    Processor for implementing scaled dot-product attention between multiple images within each batch (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        n_input_images: int = 5,
        to_k_other_frames: int = 0,
        with_self_attention: bool = True,
        random_others: bool = False,
        use_lora_in_cfa: bool = False,
        use_temb_in_lora: bool = False,
        temb_size: int = 1280,
        temb_out_size: int = 8,
        hidden_size: int = 320,
        pose_cond_dim=8,
        rank=4,
        network_alpha=None,
    ):
        """
        Args:
            n_input_images (int, optional): How many images are in one batch. Defaults to 5.
            to_k_other_frames (int, optional): How many of the other images in a batch to use as key/value. Defaults to 0.
            with_self_attention (bool, optional): If the key/value of the query image should be appended. Defaults to True.
            random_others (bool, optional): If True, will select the k_other_frames randomly, otherwise sequentially. Defaults to False.
        """
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CrossFrameAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        if not 0 <= to_k_other_frames + with_self_attention <= n_input_images:
            raise ValueError(
                f"Need 0 <= to_k_other_frames + with_self_attention <= n_input_images, but got: to_k_other_frames={to_k_other_frames}, with_self_attention={with_self_attention}, n_input_images={n_input_images}"
            )

        self.set_config(n_input_images, to_k_other_frames, with_self_attention, random_others)

        self.set_save_attention_matrix(False)

        # init lora-specific layers
        self.use_lora_in_cfa = use_lora_in_cfa
        self.use_temb_in_lora = use_temb_in_lora
        self.hidden_size = hidden_size
        self.pose_cond_dim = pose_cond_dim
        self.rank = rank
        self.network_alpha = network_alpha

        if use_lora_in_cfa:
            lora_in_size = hidden_size + pose_cond_dim
            if use_temb_in_lora:
                self.temb_proj = torch.nn.Sequential(
                    torch.nn.Linear(temb_size, temb_size // 2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(temb_size // 2, temb_out_size),
                    torch.nn.ELU(inplace=True),
                )
                lora_in_size += temb_out_size

            self.to_q_lora = LoRALinearLayer(lora_in_size, hidden_size, rank, network_alpha)
            self.to_k_lora = LoRALinearLayer(lora_in_size, hidden_size, rank, network_alpha)
            self.to_v_lora = LoRALinearLayer(lora_in_size, hidden_size, rank, network_alpha)
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def set_config(self,
        n_input_images: int,
        to_k_other_frames: int = 0,
        with_self_attention: bool = True,
        random_others: bool = False,
    ):
        self.n = n_input_images

        # [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
        self.ids = [[k for k in range(self.n) if k != i] for i in range(self.n)]
        for i in range(self.n):
            if random_others:
                random.shuffle(self.ids[i])
            self.ids[i] = self.ids[i][:to_k_other_frames]
        self.ids = torch.tensor(self.ids, dtype=torch.long)

        if with_self_attention:
            # [0, 1, 2, 3, 4]
            self_attention_ids = torch.tensor([i for i in range(self.n)], dtype=torch.long)

            # [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 0, 1, 3, 4], [3, 0, 1, 2, 4], [4, 0, 1, 2, 3]]
            self.ids = torch.cat([self_attention_ids[..., None], self.ids], dim=-1)

        self.k = self.ids.shape[1]  # how many of the frames in a batch to attend to. 1 < k <= n

    def set_save_attention_matrix(self, save: bool, on_cpu: bool = False, only_uncond: bool = True):
        self.do_save_attention_matrix = save
        self.save_attention_matrix_on_cpu = on_cpu
        self.save_only_uncond_batch = only_uncond

    @torch.no_grad()
    def save_attention_matrix(
        self, attn: Attention, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor
    ):
        if self.do_save_attention_matrix:
            if self.save_only_uncond_batch:
                batches_of_frames = query.shape[0] // self.n
                assert (
                    batches_of_frames == 2
                ), "only support save_attention_matrix for batch_size=1 with classifier-free-guidance"
                query = query.reshape(batches_of_frames, self.n, *query.shape[1:])
                key = key.reshape(batches_of_frames, self.n, *key.shape[1:])
                # Filter out unconditional TODO FIXME or should we use [0]?
                query = query[1]
                key = key[1]

            N, heads = query.shape[:2]
            query = query.reshape(N * heads, *query.shape[2:])
            key = key.reshape(N * heads, *key.shape[2:])

            if self.save_attention_matrix_on_cpu:
                query = query.cpu()
                key = key.cpu()
                attention_mask = attention_mask.cpu() if attention_mask is not None else None

            self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attention_probs = self.attention_probs.reshape(
                N, heads, *self.attention_probs.shape[1:]
            )  # (N, attn.heads, query_dim, key_dim)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        pose_cond=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # prepare encoder_hidden_states
        is_self_attention = encoder_hidden_states is None
        use_lora_in_cfa = self.use_lora_in_cfa and pose_cond is not None and is_self_attention
        if is_self_attention:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if use_lora_in_cfa:
            # prepare pose_cond for lora
            pose_cond = (
                pose_cond[:, None, :]
                .repeat(1, hidden_states.shape[1], 1)
                .to(device=hidden_states.device, dtype=hidden_states.dtype)
            )
            lora_cond = [hidden_states, pose_cond]

            # prepare temb for lora
            if self.use_temb_in_lora:
                temb = self.temb_proj(temb)
                temb = (
                    temb[:, None, :]
                    .repeat(1, hidden_states.shape[1], 1)
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                )
                lora_cond.append(temb)

            # construct final lora_cond tensor
            lora_cond = torch.cat(lora_cond, dim=-1)

            # encode with lora -- encoder_hidden_states is the same as hidden_states
            query = attn.to_q(hidden_states) + scale * self.to_q_lora(lora_cond)
            key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(lora_cond)
            value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(lora_cond)
        else:
            # encode without lora
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        # we want to change key/value only in case of self_attention. cross_attention refers to text-conditioning which should remain unchanged.
        if is_self_attention:
            # update key/value to contain the values of other frames within the same batch
            batches_of_frames = key.size()[0] // self.n

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = expand_batch(key, self.n)
            key = key[:, self.ids]  # (batches_of_frames, self.n, self.k, sequence_length, inner_dim)
            key = key.view(
                batches_of_frames, self.n, -1, key.shape[-1]
            ).contiguous()  # (batches_of_frames, self.n, self.k * sequence_length, inner_dim)

            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = expand_batch(value, self.n)
            value = value[:, self.ids]
            value = value.view(batches_of_frames, self.n, -1, value.shape[-1]).contiguous()

            # rearrange back to original shape
            key = collapse_batch(key)
            value = collapse_batch(value)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        self.save_attention_matrix(attn, query, key, attention_mask)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        if use_lora_in_cfa:
            hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PoseCondLoRAAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing the pose-conditioned LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.
    Can be used for self-attention and cross-attention as an alternative to Zero-123 style of pose conditioning (which requires finetuning the whole model).

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        pose_cond_dim (`int`, *optional*):
            The number of channels in the pose_conditioning.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, pose_cond_dim=8, rank=4, network_alpha=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PoseCondLoRAAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.pose_cond_dim = pose_cond_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size + pose_cond_dim, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(
            (cross_attention_dim or hidden_size) + pose_cond_dim, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            (cross_attention_dim or hidden_size) + pose_cond_dim, hidden_size, rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0, pose_cond=None
    ):
        if pose_cond is None:
            raise ValueError("pose_cond cannot be None")

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # prepare pose_cond for query
        q_pose_cond = (
            pose_cond[:, None, :]
            .repeat(1, hidden_states.shape[1], 1)
            .to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        q_lora_cond = torch.cat([hidden_states, q_pose_cond], dim=-1)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(q_lora_cond)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # prepare pose_cond for key/value
        kv_pose_cond = (
            pose_cond[:, None, :]
            .repeat(1, encoder_hidden_states.shape[1], 1)
            .to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        )
        kv_lora_cond = torch.cat([encoder_hidden_states, kv_pose_cond], dim=-1)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(kv_lora_cond)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(kv_lora_cond)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
