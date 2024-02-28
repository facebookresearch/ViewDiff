# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def get_pixel_grids(height, width, reverse=False):
    if reverse:
        # specify as +X left, +Y up (e.g. Pytorch3D convention, see here: https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md)
        x_linspace = torch.linspace(width - 1, 0, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(height - 1, 0, height).view(height, 1).expand(height, width)
    else:
        # specify as +X right, +Y down
        x_linspace = torch.linspace(0, width - 1, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height).view(height, 1).expand(height, width)
    x_coordinates = x_linspace.contiguous().view(-1).contiguous()
    y_coordinates = y_linspace.contiguous().view(-1).contiguous()

    ones = torch.ones(height * width)
    indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid


def project_batch(points: torch.Tensor, K: torch.Tensor, world2cam: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """

    Args:
        points (torch.Tensor): (batch_size, 3, P)
        world2cam (torch.Tensor): (batch_size, 4, 4)
        K (torch.Tensor): (batch_size, 3, 3)

    Returns:
        torch.Tensor: (xy in pixels, depth)
    """
    cam_points = world2cam[..., :3, :3].bmm(points) + world2cam[..., :3, 3:4]
    xy_proj = K.bmm(cam_points)

    zs = xy_proj[..., 2:3, :]
    mask = (zs.abs() < eps).detach()
    zs[mask] = eps
    sampler = torch.cat((xy_proj[..., 0:2, :] / zs, xy_proj[..., 2:3, :]), dim=1)

    # Remove invalid zs that cause nans
    sampler[mask.repeat(1, 3, 1)] = -10

    return sampler


def screen_to_ndc(x, h, w):
    # convert as specified here: https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
    sampler = torch.clone(x)
    if h > w:
        # W from [-1, 1], H from [-s, s] where s=H/W
        sampler[..., 0:1] = (sampler[..., 0:1] + 0.5) / w * 2.0 - 1.0
        sampler[..., 1:2] = ((sampler[..., 1:2] + 0.5) / h * 2.0 - 1.0) * h / w
    else:
        # H from [-1, 1], W from [-s, s] where s=W/H
        sampler[..., 0:1] = ((sampler[..., 0:1] + 0.5) / w * 2.0 - 1.0) * w / h
        sampler[..., 1:2] = (sampler[..., 1:2] + 0.5) / h * 2.0 - 1.0
    return sampler
