# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import math
from typing import Optional, Tuple, Union
from .fastplane_sig_function import (
    fastplane,
    FastplaneShapeRepresentation,
    N_LAYERS,
    FastplaneActivationFun,
)


MIN_KERNEL_RENDER_DIM = 16


class FastplaneModule(torch.nn.Module):
    def __init__(
        self,
        mlp_n_hidden: int,
        render_dim: int,
        num_samples: int,
        num_samples_inf: int = 0,
        opacity_init_bias: float = -5.0,
        gain: float = 1.0,
        BLOCK_SIZE: int = 16,
        transmittance_thr: float = 0.0,
        mask_out_of_bounds_samples: bool = False,
        inject_noise_sigma: float = 0.0,
        inject_noise_seed: Optional[int] = None,
        contract_coords: bool = False,
        contract_perc_foreground: float = 0.5,
        disparity_at_inf: float = 1e-5,
        shape_representation: FastplaneShapeRepresentation = FastplaneShapeRepresentation.TRIPLANE,
        activation_fun: FastplaneActivationFun = FastplaneActivationFun.SOFTPLUS,
        bg_color: Union[Tuple[float, ...], float] = 0.0,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.num_samples_inf = num_samples_inf
        self.opacity_init_bias = opacity_init_bias
        self.gain = gain
        self.BLOCK_SIZE = BLOCK_SIZE
        self.transmittance_thr = transmittance_thr
        self.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        self.inject_noise_sigma = inject_noise_sigma
        self.inject_noise_seed = inject_noise_seed
        self.contract_coords = contract_coords
        self.contract_perc_foreground = contract_perc_foreground
        self.disparity_at_inf = disparity_at_inf
        self.shape_representation = shape_representation
        self.activation_fun = activation_fun
        self.render_dim = render_dim

        kernel_render_dim = max(MIN_KERNEL_RENDER_DIM, render_dim)
        self.mlp_weights = torch.nn.Parameter(torch.zeros(N_LAYERS, mlp_n_hidden, mlp_n_hidden))
        for i in range(N_LAYERS):
            torch.nn.init.xavier_uniform_(self.mlp_weights.data[i])
        self.mlp_biases = torch.nn.Parameter(torch.zeros(N_LAYERS, mlp_n_hidden))
        self.weight_opacity = torch.nn.Parameter(
            torch.rand(mlp_n_hidden) * (2 / math.sqrt(mlp_n_hidden)) - 1 / math.sqrt(mlp_n_hidden)
        )  # xavier init
        self.bias_opacity = torch.nn.Parameter(torch.zeros(1) + opacity_init_bias)
        self.weight_color = torch.nn.Parameter(torch.zeros(mlp_n_hidden, max(render_dim, kernel_render_dim)))
        torch.nn.init.xavier_uniform_(self.weight_color.data)
        self.bias_color = torch.nn.Parameter(torch.zeros(kernel_render_dim))
        self.register_buffer("bg_color", self._process_bg_color(bg_color))

    def _process_bg_color(self, bg_color: Union[Tuple[float, ...], float]) -> torch.Tensor:
        if isinstance(bg_color, float):
            bg_color = torch.tensor([bg_color] * self.render_dim, dtype=torch.float)
        elif not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color, dtype=torch.float)
            assert len(bg_color) == self.render_dim
        return bg_color

    def forward(
        self,
        rays: torch.Tensor,
        centers: torch.Tensor,
        rays_encoding: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        v: Optional[torch.Tensor] = None,  # voxel grid input
        v_color: Optional[torch.Tensor] = None,
        xy: Optional[torch.Tensor] = None,  # triplane input
        yz: Optional[torch.Tensor] = None,
        zx: Optional[torch.Tensor] = None,
        xy_color: Optional[torch.Tensor] = None,
        yz_color: Optional[torch.Tensor] = None,
        zx_color: Optional[torch.Tensor] = None,
        inject_noise_sigma: Optional[float] = None,
        bg_color: Union[Tuple[float, ...], float, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inject_noise_sigma = self.inject_noise_sigma if inject_noise_sigma is None else inject_noise_sigma

        device = rays.device

        # check inputs
        if xy is not None:
            # triplane input
            assert v is None
            assert v_color is None
            xy_or_v = xy
            xy_color_or_v_color = xy_color
        else:
            # voxel grid input
            assert v is not None
            for x_ in [xy, yz, zx, xy_color, yz_color, zx_color]:
                assert x_ is None
            xy_or_v = v
            xy_color_or_v_color = v_color

        bg_color = (self.bg_color if bg_color is None else self._process_bg_color(bg_color)).to(device)

        ray_length_render, negative_log_transmittance, feature_render = fastplane(
            xy=xy_or_v,
            yz=yz,
            zx=zx,
            xy_color=xy_color_or_v_color,
            yz_color=yz_color,
            zx_color=zx_color,
            weights=self.mlp_weights,
            biases=self.mlp_biases,
            weight_opacity=self.weight_opacity,
            bias_opacity=self.bias_opacity,
            weight_color=self.weight_color,
            bias_color=self.bias_color,
            rays=rays,
            centers=centers,
            rays_encoding=rays_encoding,
            near=near,
            far=far,
            num_samples=self.num_samples,
            num_samples_inf=self.num_samples_inf,
            gain=self.gain,
            BLOCK_SIZE=self.BLOCK_SIZE,
            transmittance_thr=self.transmittance_thr,
            mask_out_of_bounds_samples=self.mask_out_of_bounds_samples,
            inject_noise_sigma=inject_noise_sigma,
            inject_noise_seed=self.inject_noise_seed,
            contract_coords=self.contract_coords,
            contract_perc_foreground=self.contract_perc_foreground,
            disparity_at_inf=self.disparity_at_inf,
            shape_representation=self.shape_representation,
            activation_fun=self.activation_fun,
        )

        mask = 1 - torch.exp(-negative_log_transmittance)

        # apply the bg color
        feature_render = feature_render[..., : self.render_dim] + (1 - mask[..., None]) * bg_color

        return feature_render, mask, ray_length_render
