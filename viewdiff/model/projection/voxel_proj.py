# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Literal, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from ..custom_attention_processor import collapse_batch, expand_batch
from .util import screen_to_ndc, project_batch
from ...data.co3d.util import scale_bbox, scale_camera_center


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var


# adapted from: https://github.com/googleinterns/IBRNet/blob/master/ibrnet/mlp_network.py
class IBRNet_Aggregator(nn.Module):
    def __init__(self, feature_dim: int = 32, anti_alias_pooling: bool = False, kernel_size: int = 1, padding: int = 0, use_temb: bool = False, temb_dim: int = 1280):
        super().__init__()

        self.anti_alias_pooling = anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)

        # turn voxel_depth, rays into encoding to add to features
        self.ray_depth_encoder = nn.Sequential(
            nn.Conv3d(4, feature_dim // 2, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim // 2, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # turn time embedding into encoding to add to features
        self.use_temb = use_temb
        if use_temb:
            self.temb_encoder = nn.Sequential(
                nn.Linear(temb_dim, temb_dim // 2),
                activation_func,
                nn.Linear(temb_dim // 2, feature_dim),
                activation_func,
            )

        # shared part of feature/weight encoding
        self.base_fc = nn.Sequential(
            nn.Conv3d(feature_dim * 3, feature_dim * 2, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim * 2, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # compute first part of averaging weights, final features
        self.vis_fc = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim, feature_dim + 1, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # compute second part of averaging weights
        self.vis_fc2 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

        # combine (weight, mean, var) into final grid
        self.statistics_out = nn.Sequential(
            nn.Conv3d(2 * feature_dim + 1, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor, voxel_depth: torch.Tensor, voxel_dir: torch.Tensor, temb: torch.Tensor = None):
        """

        Args:
            features (torch.Tensor): tensor of shape (batch_size, num_images, feature_dim, grid_dim, grid_dim, grid_dim)
            ray_diff (torch.Tensor): tensor of shape (batch_size, num_images, 3, grid_dim, grid_dim, grid_dim)
            mask (torch.Tensor): tensor of shape (batch_size, num_images, 1, grid_dim, grid_dim, grid_dim)
            temb (torch.Tensor) tensor of shape (batch_size, num_images, temb_dim)
        """
        num_images = features.shape[1]

        # add ray encoding and depth
        ray_depth_enc = torch.cat([voxel_dir, voxel_depth], dim=2)
        ray_depth_enc = collapse_batch(ray_depth_enc)
        ray_depth_enc = self.ray_depth_encoder(ray_depth_enc)
        ray_depth_enc = expand_batch(ray_depth_enc, num_images)
        features = features + ray_depth_enc

        # add temb encoding
        if self.use_temb:
            temb_enc = self.temb_encoder(temb)
            temb_enc = temb_enc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            temb_enc = temb_enc.repeat(1, 1, 1, *features.shape[3:]).contiguous()
            features = features + temb_enc

        if self.anti_alias_pooling:
            raise NotImplementedError()
        else:
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each voxel (== same as aggregate_cost_volume(agg_fn="mean"))
        mean, var = fused_mean_variance(
            features, weight
        )  # (batch_size, 1, feature_dim, grid_dim, grid_dim, grid_dim)
        globalfeat = torch.cat([mean, var], dim=2)  # (batch_size, 1, 2*feature_dim, grid_dim, grid_dim, grid_dim)

        # combine each voxel with the globalfeat across all views
        # (batch_size, num_images, 3*feature_dim, grid_dim, grid_dim, grid_dim)
        x = torch.cat([globalfeat.expand(-1, num_images, -1, -1, -1, -1), features], dim=2)

        # encode base_fc: shared part of feature/weight encoding
        x = collapse_batch(x)  # (batch_size * num_images, 3*feature_dim, grid_dim, grid_dim, grid_dim)
        weight = collapse_batch(weight)
        mask = collapse_batch(mask)
        x = self.base_fc(x)

        # get averaging weights, final features
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[1] - 1, 1], dim=1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        # compute weighted average
        vis = expand_batch(vis, num_images)
        x = expand_batch(x, num_images)
        weight = vis / (torch.sum(vis, dim=1, keepdim=True) + 1e-8)
        mean, var = fused_mean_variance(x, weight)

        # combine (mean, var, weight) and let a final custom layer transform it into the feature grid
        # (batch_size, 2*feature_dim + 1, grid_dim, grid_dim, grid_dim)
        globalfeat = torch.cat([mean.squeeze(1), var.squeeze(1), weight.mean(dim=1)], dim=1)
        globalfeat = self.statistics_out(globalfeat)

        return globalfeat


def get_grid_to_world_space_matrix(bbox: torch.Tensor) -> torch.Tensor:
    """Calculates the matrix that transforms grid-space coordinates to world-space coordinates.
    World-space is defined through the bbox. We calculate a matrix that translates and uniformly scales the grid-space to fit into the bbox.
    Grid-space is defined in [-1, 1] where the extrema refer to the corners of the grids, e.g. voxels refer to the centers shifted by half-pixel coordinate.

    Args:
        bbox (torch.Tensor): tensor of shape (B, 2, 3) giving the min-xyz and max-xyz bbox corners.

    Returns:
        torch.Tensor: the grid_to_world_space_matrix of shape (4, 4)
    """
    bbox_min = bbox[:, 0]
    bbox_max = bbox[:, 1]

    # scale from 2.0 to largest_side_length
    largest_side_length = (bbox_max - bbox_min).max(dim=1).values
    uniform_scale = largest_side_length / 2.0

    # translate from center = (0, 0) to bbox_center
    bbox_center = (bbox_min + bbox_max) / 2

    # build final matrix combining scale and translation
    grid2world = torch.zeros(bbox.shape[0], 4, 4, device=bbox.device)
    grid2world[:, 0, 0] = uniform_scale
    grid2world[:, 1, 1] = uniform_scale
    grid2world[:, 2, 2] = uniform_scale
    grid2world[:, 3, 3] = 1
    grid2world[:, :3, 3] = bbox_center

    return grid2world


def _contract_pi_inv(x, perc_foreground: float = 0.5):
    max_index = torch.argmax(x.abs(), dim=1, keepdim=True)
    n = torch.gather(x, dim=1, index=max_index)
    p = 1.0 / perc_foreground
    n_inv = torch.where(n > 0, -(p - 1) / (n - p), -(p - 1) / (n + p))
    x_inv = torch.where((x.abs() - n).abs() <= 1e-7, n_inv.repeat(1, x.shape[1]), n_inv.abs() * x)
    x_inv = torch.where(n.abs() <= 1.0, x, x_inv)
    return x_inv


@torch.autocast("cuda", enabled=False)
def build_cost_volume(
    features: torch.Tensor,
    poses_world2cam: torch.Tensor,
    intrinsics: torch.Tensor,
    bbox: torch.Tensor,
    grid_dim: int = 128,
    feature_sampling_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
    depth: torch.Tensor = None,
    depth_threshold: float = 1e-8,
    contract_background: bool = False,
    contract_background_percentage: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a per-frame cost-volume of the features.

    Args:
        features (torch.Tensor): (batch_size, num_images, feature_dim, height, width)
        poses_world2cam (torch.Tensor): (batch_size, num_images, 4, 4)
        intrinsics (torch.Tensor): (batch_size, num_images, 4, 4) - already downsampled to respect (height, width).
        bbox (torch.Tensor): (batch_size, 2, 3)
        world_space_transform (torch.Tensor): (batch_size, 3, 3)
        grid_dim (int): voxel-grid dimension in xyz
        feature_sampling_mode (Literal["bilinear", "nearest", "bicubic"], optional): sampling mode for interpolation of features. Defaults to "bilinear".
        depth (torch.Tensor, optional): (batch_size, num_images, height, width). depth map to build up the cost-volume. Defaults to None.
        depth_threshold (float, optional): if depth is given, uses this threshold to filter out voxels that are not close enough to GT depth. Defaults to 1e-3.

    Returns:
        (torch.Tensor: cost-volume per frame, shape (batch_size, num_images, feature_dim, grid_dim, grid_dim, grid_dim),
        torch.Tensor: weights of cost-volume per frame, shape (batch_size, num_images, 1, grid_dim, grid_dim, grid_dim))
    """
    # get shape info
    batch_size = features.shape[0]
    num_images = features.shape[1]
    feature_dim = features.shape[2]
    height = features.shape[3]
    width = features.shape[4]

    # Generate voxel indices. --> xyz coordinates in [0...grid_dim - 1]
    x = torch.arange(grid_dim, dtype=poses_world2cam.dtype, device=poses_world2cam.device)
    y = torch.arange(grid_dim, dtype=poses_world2cam.dtype, device=poses_world2cam.device)
    z = torch.arange(grid_dim, dtype=poses_world2cam.dtype, device=poses_world2cam.device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="xy")
    grid_xyz = torch.cat(
        [
            grid_x.view(grid_dim, grid_dim, grid_dim, 1),
            grid_y.view(grid_dim, grid_dim, grid_dim, 1),
            grid_z.view(grid_dim, grid_dim, grid_dim, 1),
        ],
        dim=3,
    )

    grid_xyz = grid_xyz.view(grid_dim * grid_dim * grid_dim, 3).contiguous()
    num_voxels = grid_xyz.shape[0]

    # convert grid coordinates to [-1, 1].
    # convention: coordinates refer to voxel centers and the centers are at half-pixel coordinates
    grid_xyz = (grid_xyz + 0.5) / grid_dim * 2.0 - 1.0

    if contract_background:
        # what is the valid range of coordinates in foreground/background separ`1ation
        # e.g. if we have 50% foreground, 50% background, then it goes from [-2, 2] and the values in [-1, 1] are foreground
        # e.g. if we have 80% foreground, 20% background, then it goes from [-1.25, 1.25] and the values in [-1, 1] are foreground
        # we invert this step here
        grid_xyz = grid_xyz * (1.0 / contract_background_percentage)

        # see MERF (https://arxiv.org/pdf/2302.12249.pdf): we do not want to store features at the ill-defined regions of the contraction function
        invalid_contract_voxels_mask = (grid_xyz > 1.0).sum(dim=-1) > 1
        invalid_contract_voxels_mask = invalid_contract_voxels_mask.view(grid_dim, grid_dim, grid_dim).contiguous()

        # invert the contraction --> store image features for voxels at their true world-space coordinates
        grid_xyz = _contract_pi_inv(grid_xyz, contract_background_percentage)

    # get grid2world matrices
    grid2world = get_grid_to_world_space_matrix(bbox).to(poses_world2cam.device)

    # convert grid-space points to world-space points
    world_xyz = grid_xyz[None].repeat(batch_size, 1, 1).transpose(1, 2)  # (batch_size, 3, num_voxels)
    world_xyz = grid2world[:, :3, :3].bmm(world_xyz) + grid2world[:, :3, 3:4]

    # We process all samples simultaneously
    intrinsics = collapse_batch(intrinsics)
    poses_world2cam = collapse_batch(poses_world2cam)
    features = collapse_batch(features)
    grid2world = grid2world.repeat_interleave(num_images, dim=0)
    grid_xyz = (
        grid_xyz[None].repeat(batch_size * num_images, 1, 1).transpose(1, 2)
    )  # (batch_size * num_images, 3, num_voxels)

    # Project voxels to screen/ndc space.
    grid2cam = poses_world2cam.bmm(grid2world)
    sampler = project_batch(grid_xyz, intrinsics, grid2cam)
    sampler = sampler.permute(0, 2, 1)
    sampler = screen_to_ndc(sampler, height, width)

    # Mark valid pixels.
    valid_pixels = (
        (sampler[..., 0:1] >= -1)
        & (sampler[..., 0:1] <= 1)
        & (sampler[..., 1:2] >= -1)
        & (sampler[..., 1:2] <= 1)
        & (sampler[..., 2:3] > 0)
    )
    valid_pixels = valid_pixels.repeat(1, 1, 3)
    sampler[~valid_pixels] = -10
    sampler[~valid_pixels] = -10

    # Interpolate features.
    def make_query_pixels():
        query_pixels = torch.stack([sampler[..., 0], sampler[..., 1]], dim=-1)
        query_pixels = query_pixels.view(batch_size * num_images, 1, num_voxels, 2).contiguous()
        return query_pixels

    # Mark valid pixels based on GT depth
    if depth is not None:
        depth = collapse_batch(depth)
        queried_depth = torch.nn.functional.grid_sample(
            depth.view(batch_size * num_images, 1, height, width),
            make_query_pixels(),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze()
        diff = (queried_depth - sampler[..., 2]) ** 2
        invalid_depth_pixels = (diff < depth_threshold)[..., None].repeat(1, 1, 3)
        valid_pixels &= invalid_depth_pixels
        sampler[~valid_pixels] = -10
        sampler[~valid_pixels] = -10

    # Sample features at computed pixel locations.
    query_pixels = make_query_pixels()
    queried_features = torch.nn.functional.grid_sample(
        features,
        query_pixels,
        mode=feature_sampling_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    queried_features = queried_features.view(batch_size * num_images, feature_dim, num_voxels).contiguous()

    # Set invalid values.
    valid_pixels = valid_pixels[..., 0]
    queried_weights = valid_pixels.float()
    queried_features = queried_features.permute(1, 0, 2)
    queried_features[:, ~valid_pixels] = 0

    # Unflatten to xyz grid_dim shape.
    queried_weights = queried_weights.view(batch_size * num_images, grid_dim, grid_dim, grid_dim).contiguous()
    queried_weights = queried_weights.unsqueeze(1)

    voxel_depth = sampler[..., 2].view(batch_size * num_images, grid_dim, grid_dim, grid_dim).contiguous()
    voxel_depth = voxel_depth.unsqueeze(1)

    queried_features = queried_features.view(feature_dim, batch_size * num_images, grid_dim, grid_dim, grid_dim).contiguous()
    queried_features = queried_features.transpose(0, 1)

    # Expand to batch_size, num_images.
    queried_weights = expand_batch(queried_weights, num_images)
    voxel_depth = expand_batch(voxel_depth, num_images)
    queried_features = expand_batch(queried_features, num_images)

    # mask out queried_features for those voxels that are invalid in the contraction
    if contract_background:
        queried_features[:, :, :, invalid_contract_voxels_mask] = 0

    return queried_features, queried_weights, world_xyz, voxel_depth


def sparsify_cost_volume(features: torch.Tensor, weights: torch.Tensor, points: torch.Tensor):
    points = points.reshape(
        points.shape[0], points.shape[1], *features.shape[2:]
    )  # (batch_size, 3, grid_dim, grid_dim, grid_dim)

    feature_list = []
    points_list = []
    remaining_points = 0
    for i in range(features.shape[0]):
        # sparsify features
        m = weights[i].bool().squeeze()  # (grid_dim, grid_dim, grid_dim)
        feature_list.append(features[i, :, m])  # (C, P)

        # sparsify world_points
        sparse_points = points[i, :, m]
        points_list.append(sparse_points)  # (3, P)
        remaining_points += sparse_points.shape[1]

    return feature_list, points_list, remaining_points


def mean_aggregate_cost_volumes(
    features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Aggregates multiple per-frame feature_grids into a single feature_grid using the specified aggregation function.

    Args:
        features (torch.Tensor): tensor of shape (batch_size, num_images, feature_dim, grid_dim, grid_dim, grid_dim)
        weights (torch.Tensor): tensor of shape (batch_size, num_images, 1, grid_dim, grid_dim, grid_dim)

    Returns:
        torch.Tensor: the aggregated feature_grid of shape (batch_size, feature_dim, grid_dim, grid_dim, grid_dim)
    """
    features = features.sum(dim=1)  # (batch_size, C, grid_dim, grid_dim, grid_dim)
    weights = weights.sum(dim=1)  # (batch_size, 1, grid_dim, grid_dim, grid_dim)
    features = features / (weights + 1e-8)
    return features, weights


def get_rays_for_view(world2cam: torch.Tensor, K: torch.Tensor, canonical_rays: torch.Tensor):
    rays = canonical_rays[None].repeat(world2cam.shape[0], 1, 1)  # (batch_size, 3, h*w)

    R_inv = world2cam[..., :3, :3].transpose(1, 2)  # (batch_size, 3, 3)
    t_inv = -R_inv @ world2cam[..., :3, 3:4]  # (batch_size, 3, 1)

    rays = K.inverse().bmm(rays)
    rays = R_inv.bmm(rays) + t_inv
    centers = t_inv.expand(rays.shape)  # (batch_size, 3, h*w)
    rays = rays - centers

    rays = rays.permute(0, 2, 1)  # (batch_size, h*w, 3)
    centers = centers.permute(0, 2, 1)  # (batch_size, h*w, 3)

    return rays, centers


def ray_aabb_intersection(
    ray_d: torch.Tensor, ray_o: torch.Tensor, bbox: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ray_d, ray_o: (batch_size, h*w, 3)
    # bbox: (batch_size, 2, 3)
    # t_min, t_max: (batch_size, h*w,)
    vec = torch.where(ray_d == 0, torch.full_like(ray_d, 1e-6), ray_d)
    bbox_min = bbox[:, 0][:, None]
    bbox_max = bbox[:, 1][:, None]
    rate_a = (bbox_min - ray_o) / vec
    rate_b = (bbox_max - ray_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=-1e5, max=1e5)
    t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=-1e5, max=1e5)

    return t_min, t_max


@torch.autocast("cuda", enabled=False)
def get_rays_in_unit_cube(
    bbox: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    canonical_rays: torch.Tensor,
    use_ray_aabb: bool = True,
    default_near: float = 0.5,
    default_far: float = 5.0,
):
    """Always do it in high-precision, e.g. disable fp16 here.

    Args:
        bbox (torch.Tensor): _description_
        pose (torch.Tensor): _description_
        K (torch.Tensor): _description_
        canonical_rays (torch.Tensor): _description_
        default_near (float, optional): _description_. Defaults to 0.5.
        default_far (float, optional): _description_. Defaults to 5.0.
    """
    num_images = pose.shape[1]

    # our volume-renderer assumes that the world-space is [-1, 1]^3 and voxel-centers are at half-grid coordinates
    # our build_cost_volume() returns voxels within the bbox (sampled such that bbox-extrema are the boundaries of the voxels, e.g. voxel-centers are at half-grid coordinates)
    # we need to scale the bbox and poses to lie in (-1, 1). we can then re-use the voxel-features as they are!
    bbox, scale = scale_bbox(bbox)
    scale_pose = scale[:, None].repeat(1, num_images).view(-1).contiguous()
    pose = scale_camera_center(collapse_batch(pose), scale_pose)
    bbox = bbox.unsqueeze(1).repeat(1, num_images, 1, 1)
    bbox = collapse_batch(bbox)

    # get rays
    rays, centers = get_rays_for_view(pose, collapse_batch(K), canonical_rays)

    # get near/far sampling points along the ray
    if use_ray_aabb:
        near_t, far_t = ray_aabb_intersection(rays, centers, bbox)
        mask_out_of_bbox = far_t <= near_t
        near_t[mask_out_of_bbox] = default_near
        far_t[mask_out_of_bbox] = default_far
    else:
        R = pose[:, :3, :3]
        T = pose[:, :3, 3:4]
        camera_center = -R.transpose(-2, -1).bmm(T)
        dist_camera_center_origin = torch.linalg.vector_norm(camera_center, dim=1)
        near_t = (dist_camera_center_origin - 4.0).clamp_min(0.0).repeat(1, rays.shape[1])
        far_t = (dist_camera_center_origin + 4.0).repeat(1, rays.shape[1])

    rays = expand_batch(rays, num_images)
    centers = expand_batch(centers, num_images)
    near_t = expand_batch(near_t, num_images)
    far_t = expand_batch(far_t, num_images)

    return rays, centers, near_t, far_t, scale
