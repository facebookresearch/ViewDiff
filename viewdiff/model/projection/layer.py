# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Tuple, Literal
import torch
from torch import nn

from ...data.co3d.util import scale_intrinsics


from diffusers.models.unet_2d import UNet2DModel

from .util import get_pixel_grids
from .voxel_proj import (
    build_cost_volume,
    mean_aggregate_cost_volumes,
    get_rays_in_unit_cube,
    IBRNet_Aggregator,
)
from ..custom_unet_3d import ResnetBlock3D, UNet3DModel
from ..custom_attention_processor import collapse_batch, expand_batch

from .fastplane.fastplane_module import FastplaneModule, FastplaneShapeRepresentation


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int = 3, padding: int = 1, norm: bool = False):
        super().__init__()

        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.norm = norm
        if norm:
            # self.norm = nn.InstanceNorm2d(channels_out, affine=True)
            self.norm = nn.LayerNorm([channels_out, 64, 64])

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class UnprojReprojLayer(nn.Module):
    def __init__(
        self,
        latent_channels: int = 320,
        num_3d_layers: int = 1,
        dim_3d_latent: int = 32,
        dim_3d_grid: int = 64,
        vol_rend_num_samples_per_ray: int = 128,
        vol_rend_near: float = 0.5,
        vol_rend_far: float = 4.5,
        vol_rend_model_background: bool = False,
        vol_rend_background_grid_percentage: float = 0.5,
        vol_rend_disparity_at_inf: float = 1e-3,
        n_novel_images: int = 0,
        proj_in_mode: Literal["single", "multiple", "unet"] = "single",
        proj_out_mode: Literal["single", "multiple"] = "multiple",
        aggregator_mode: Literal["mean", "ibrnet"] = "ibrnet",
        use_temb: bool = False,
        temb_dim: int = 1280,
    ):
        super().__init__()
        self.proj_in_mode = proj_in_mode
        self.proj_out_mode = proj_out_mode
        self.n_novel_images = n_novel_images
        self.dim_3d_grid = dim_3d_grid
        self.use_3d_net = num_3d_layers > 0
        self.use_3d_unet = num_3d_layers >= 3
        self.vol_rend_model_background = vol_rend_model_background
        self.vol_rend_background_grid_percentage = vol_rend_background_grid_percentage

        if self.use_3d_unet:
            assert (
                num_3d_layers - 1
            ) % 2 == 0, "if num_3d_layers >=3 we want to construct a UNet. So specify an odd number of num_3d_layers, e.g. 3,5,7,9"
            n_blocks = (num_3d_layers - 1) // 2
            block_out_channels = [dim_3d_latent * i for i in range(1, n_blocks + 1)]
            self.blocks_3d = UNet3DModel(
                in_channels=dim_3d_latent,
                out_channels=dim_3d_latent,
                down_block_types=["DownBlock3D"] * n_blocks,
                up_block_types=["UpBlock3D"] * n_blocks,
                block_out_channels=block_out_channels,
                layers_per_block=1,
                norm_num_groups=min(32, dim_3d_latent),
            )
        else:
            self.blocks_3d = nn.ModuleList(
                [ResnetBlock3D(in_channels=dim_3d_latent, groups=min(32, dim_3d_latent))] * num_3d_layers
            )
        # renderer
        # turn incoming SD-latent-features into 3D features (e.g. normalization, feature-reduction)
        if proj_in_mode == "single":
            self.proj_in_2d = nn.Conv2d(latent_channels, dim_3d_latent, kernel_size=1, padding=0)
        elif proj_in_mode == "multiple":
            self.proj_in_2d = nn.Sequential(
                ConvBlock(latent_channels, latent_channels, kernel_size=3, padding=1),
                ConvBlock(latent_channels, dim_3d_latent, kernel_size=3, padding=1),
                nn.Conv2d(dim_3d_latent, dim_3d_latent, kernel_size=1, padding=0),
            )
        elif proj_in_mode == "unet":
            n_blocks = 3
            block_out_channels = [dim_3d_latent * i for i in range(1, n_blocks + 1)]
            self.proj_in_2d = UNet2DModel(
                in_channels=latent_channels,
                out_channels=dim_3d_latent,
                down_block_types=["DownBlock2D"] * n_blocks,
                up_block_types=["UpBlock2D"] * n_blocks,
                block_out_channels=block_out_channels,
                layers_per_block=1,
                norm_num_groups=min(32, dim_3d_latent),
                add_attention=False,
            )
        else:
            raise NotImplementedError("proj_in_mode", proj_in_mode)

        # turn projected 3D features into SD-latent-features (e.g. de-normalization, feature-increase)
        # it is necessary because vol-renderer outputs in [0, 1] because it needs sigmoid in the end to converge in general
        # however, latent features can be in arbitrary floating point feature value
        # this scale fct should learn to convert back to the arbitrary range
        # it should be a conv_1x1 to not turn this into a neural renderer that could destroy the 3D consistency (instead: only scale)
        # we have a linear and a non-linear scale fct and allow to choose them according to the --proj_out_mode flag
        # the background will only use a nonlinear_scale_fct if the background should actually be modeled (e.g. if --vol_rend_model_background is set)

        def linear_scale_fct():
            return nn.Conv2d(dim_3d_latent, latent_channels, kernel_size=1, padding=0)

        def nonlinear_scale_fct():
            return nn.Sequential(
                ConvBlock(dim_3d_latent, dim_3d_latent, kernel_size=1, padding=0),
                ConvBlock(dim_3d_latent, latent_channels, kernel_size=1, padding=0),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=1, padding=0),
            )

        if proj_out_mode == "single":
            self.proj_out_2d_fg = linear_scale_fct()
            self.proj_out_2d_bg = linear_scale_fct()
        elif proj_out_mode == "multiple":
            self.proj_out_2d_fg = nonlinear_scale_fct()
            if vol_rend_model_background:
                self.proj_out_2d_bg = nonlinear_scale_fct()
            else:
                self.proj_out_2d_bg = linear_scale_fct()
        else:
            raise NotImplementedError("proj_out_mode", proj_out_mode)

        # reduce multiple per-frame dense voxel-grids into a single dense voxel-grid
        self.aggregator_mode = aggregator_mode
        if aggregator_mode == "ibrnet":
            self.ibrnet_aggregator = IBRNet_Aggregator(feature_dim=dim_3d_latent, kernel_size=1, padding=0, use_temb=use_temb, temb_dim=temb_dim)

        # ray-direction encoder for volume_renderer
        self.linear_ray = nn.Linear(3, dim_3d_latent)
        nn.init.xavier_uniform_(self.linear_ray.weight.data)
        self.linear_ray.bias.data *= 0.0

        # volume renderer of the dense voxel-grid
        self.vol_rend_near = vol_rend_near
        self.vol_rend_far = vol_rend_far
        if vol_rend_model_background:
            self.volume_renderer = FastplaneModule(
                mlp_n_hidden=dim_3d_latent,
                render_dim=dim_3d_latent,
                num_samples=vol_rend_num_samples_per_ray,
                bg_color=1.0,
                shape_representation=FastplaneShapeRepresentation.VOXEL_GRID,
                mask_out_of_bounds_samples=True,
                num_samples_inf=vol_rend_num_samples_per_ray,
                contract_coords=True,
                contract_perc_foreground=1.0 - vol_rend_background_grid_percentage,
                disparity_at_inf=vol_rend_disparity_at_inf,
                inject_noise_sigma=0.0,
            )
        else:
            self.volume_renderer = FastplaneModule(
                mlp_n_hidden=dim_3d_latent,
                render_dim=dim_3d_latent,
                num_samples=vol_rend_num_samples_per_ray,
                bg_color=1.0,
                shape_representation=FastplaneShapeRepresentation.VOXEL_GRID,
                mask_out_of_bounds_samples=True,
            )

    def get_volume_renderer_params(self):
        params = []

        params.extend(list(self.volume_renderer.parameters()))

        return params

    def get_other_params(self):
        params = []
        for n, p in self.named_parameters():
            if "volume_renderer" not in n:
                params.append(p)

        return params

    def forward(
        self,
        latents: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        orig_hw: Tuple[int, int],
        timestep: torch.Tensor = None,
        temb: torch.Tensor = None,
        bbox: torch.Tensor = None,
        deactivate_view_dependent_rendering: bool = False
    ):
        """
        Args:
            latents (torch.Tensor): (batch_size, num_images, C, h', w')
            pose (torch.Tensor): (batch_size, num_images, 4, 4)
            K (torch.Tensor): (batch_size, num_images, 3, 3)
            orig_hw (Tuple[int, int]): same across all batches
            temb (torch.Tensor, optional): (batch_size, num_images, temb_dim). Defaults to None.
            bbox (torch.Tensor, optional): (batch_size, 2, 3). Defaults to None.

        Returns:
            _type_: _description_
        """
        # downscale to latent dimension
        batch_size = latents.shape[0]
        num_images = latents.shape[1]
        latent_hw = latents.shape[3:]
        K_scaled = scale_intrinsics(K, orig_hw, latent_hw)
        K = K_scaled

        # lazy init canonical rays
        if not hasattr(self, "canonical_rays") or self.canonical_rays.shape[1] != (latent_hw[0] * latent_hw[1]):
            self.canonical_rays = get_pixel_grids(latent_hw[0], latent_hw[1]).to(latents.device).float()

        # get ray information in the correct world-space
        rays, centers, near_t, far_t, scale = get_rays_in_unit_cube(
            bbox,
            pose,
            K,
            self.canonical_rays,
            default_near=self.vol_rend_near,
            default_far=self.vol_rend_far,
            use_ray_aabb=not self.vol_rend_model_background,
        )

        # reduce feature dim
        n_known_images = latents.shape[1] - self.n_novel_images
        features = collapse_batch(latents[:, :n_known_images])

        if self.proj_in_mode == "unet":
            assert timestep is not None
            features = self.proj_in_2d(features, collapse_batch(timestep[:, :n_known_images])).sample
        else:
            features = self.proj_in_2d(features)

        features = expand_batch(features, n_known_images)
        features = features.to(
            rays.dtype
        )  # rays have fp32, want features to have it too, e.g. build_cost_volume always in highest precision

        # concat features with rays to get per-voxel results for both
        if self.aggregator_mode == "ibrnet":
            rays_for_cost_volume = (
                rays[:, :n_known_images]
                .permute(0, 1, 3, 2)
                .reshape(batch_size, n_known_images, 3, latent_hw[0], latent_hw[1])
            )
            features = torch.cat([features, rays_for_cost_volume], dim=2)

        # unproj to dense voxel grid (per-frame)
        features, weights, points, voxel_depth = build_cost_volume(
            features,
            pose[:, :n_known_images],
            K[:, :n_known_images],
            bbox,
            grid_dim=self.dim_3d_grid,
            contract_background=self.vol_rend_model_background,
            contract_background_percentage=self.vol_rend_background_grid_percentage,
        )

        # aggregate per-frame grids into single 3D grid
        features = features.to(latents.dtype)  # 3D network should be in half-precision if specified
        agg_temb = temb[:, :n_known_images] if temb is not None else None

        if self.aggregator_mode == "ibrnet":
            features, voxel_dir = torch.split(features, [features.shape[2] - 3, 3], dim=2)
            features = self.ibrnet_aggregator(features, weights, voxel_depth, voxel_dir, agg_temb)
        elif self.aggregator_mode == "mean":
            features, _ = mean_aggregate_cost_volumes(features, weights)

        # apply 3D layers on grid
        if self.use_3d_unet:
            assert timestep is not None
            # give last timestep as input --> assumption is that non-noisy images are never in last position (e.g. sliding window inputs always are the first images)
            features = self.blocks_3d(features, timestep[:, -1]).sample
        else:
            for block in self.blocks_3d:
                # give last timestep as input --> assumption is that non-noisy images are never in last position (e.g. sliding window inputs always are the first images)
                features = block(features, temb[:, -1] if temb is not None else None)

        features = features.permute(0, 2, 3, 4, 1)

        # collapse (num_images, h*w) to render all rays jointly
        rays = rays.reshape(batch_size, -1, rays.shape[3])
        centers = centers.reshape(batch_size, -1, centers.shape[3])
        near_t = near_t.reshape(batch_size, -1)
        far_t = far_t.reshape(batch_size, -1)

        # volume-render (grid, rays)
        # only supports fp32 (there are some triton kernel impls that cast to float32, so make it the dtype from the very beginning)
        with torch.autocast("cuda", enabled=False):
            if deactivate_view_dependent_rendering:
                dummy_pose = torch.tensor([
                    [0, 1, 0, 0],
                    [-0.7071, 0, 0.7071, 0],
                    [0.7071, 0, 0.7071, 3.0],
                    [0, 0, 0, 1.0],
                ], device=pose.device, dtype=pose.dtype)
                dummy_pose = dummy_pose[None, None].repeat(pose.shape[0], pose.shape[1], 1, 1)
                dummy_rays, _, _, _, _ = get_rays_in_unit_cube(
                    bbox,
                    dummy_pose,
                    K,
                    self.canonical_rays,
                    default_near=self.vol_rend_near,
                    default_far=self.vol_rend_far,
                    use_ray_aabb=not self.vol_rend_model_background,
                )
                dummy_rays = dummy_rays.reshape(batch_size, -1, dummy_rays.shape[3])
                rays_encoding = self.linear_ray(dummy_rays).float()
            else:
                rays_encoding = self.linear_ray(rays).float()

            projected_latents, projected_mask, projected_depth = self.volume_renderer(
                v=features.float(),
                rays_encoding=rays_encoding,
                rays=rays,
                centers=centers,
                near=near_t,
                far=far_t,
            )

            projected_latents = projected_latents.to(latents.dtype)
            projected_mask = projected_mask.to(latents.dtype)
            projected_depth = projected_depth.to(latents.dtype)

        # reshape back to (num_images, h*w)
        hw = latent_hw[0] * latent_hw[1]
        projected_latents = projected_latents.reshape(batch_size, num_images, hw, projected_latents.shape[2])
        projected_depth = projected_depth.reshape(batch_size, num_images, hw)
        projected_mask = projected_mask.reshape(batch_size, num_images, hw)

        # reshape to (batch_size, num_images, C, h, w)
        projected_latents = projected_latents.permute(0, 1, 3, 2)
        projected_latents = projected_latents.reshape(
            batch_size,
            num_images,
            projected_latents.shape[2],
            latent_hw[0],
            latent_hw[1],
        )
        projected_depth = projected_depth.reshape(batch_size, num_images, latent_hw[0], latent_hw[1])
        projected_depth = projected_depth.unsqueeze(2)
        projected_mask = projected_mask.reshape(batch_size, num_images, latent_hw[0], latent_hw[1])
        projected_mask = projected_mask.unsqueeze(2)

        # proj-out (back to larger channels, back from 0..1 to correct output range)
        # have separate de-normalization layers for fg and bg
        projected_latents = collapse_batch(projected_latents)
        p_fg = self.proj_out_2d_fg(projected_latents)
        p_bg = self.proj_out_2d_bg(projected_latents)
        m = collapse_batch(projected_mask).repeat(1, p_fg.shape[1], 1, 1)
        projected_latents = m * p_fg + (1 - m) * p_bg
        projected_latents = expand_batch(projected_latents, num_images)

        return projected_latents, projected_mask, projected_depth
