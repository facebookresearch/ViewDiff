# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum
import torch
import math
import random
from typing import Optional
import triton

try:
    from .fastplane_triton_sig import _fw_kernel, _bw_kernel, _int_to_randn_kernel
except ImportError:
    from fastplane_triton_sig import _fw_kernel, _bw_kernel, _int_to_randn_kernel


# The number of MLP layers.
#   - has to be hardcoded for now to avoid wasting GPU memory with
#     an N_MAX_LAYERS x C x C-sized MLP-weights buffer
N_LAYERS = 3

# Triton constraint - triton.language.dot requires
# both operands to be matrices with each size >= 16.
MIN_BLOCK_SIZE = 16


def _flat(x, n):
    if x is None:
        return None
    return x.reshape(-1, *x.shape[-n:]).contiguous()


def _flat_or_empty(x, n, device):
    if x is None:
        return torch.empty((1,), device=device)
    return _flat(x, n)


class FastplaneShapeRepresentation(Enum):
    TRIPLANE = 0
    VOXEL_GRID = 1


class FastplaneActivationFun(Enum):
    SOFTPLUS = 0
    RELU = 1


class FastplaneFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xy: torch.Tensor,  # either stores the first triplane, or the full voxel grid
        yz: Optional[torch.Tensor],
        zx: Optional[torch.Tensor],
        xy_color: Optional[torch.Tensor],
        yz_color: Optional[torch.Tensor],
        zx_color: Optional[torch.Tensor],
        weights: torch.Tensor,
        biases: torch.Tensor,
        weight_opacity: torch.Tensor,
        bias_opacity: torch.Tensor,
        weight_color: torch.Tensor,
        bias_color: torch.Tensor,
        rays: torch.Tensor,
        centers: torch.Tensor,
        rays_enc: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int,
        num_samples_inf: int,
        gain: float,
        BLOCK_SIZE: int,
        transmittance_thr: float,
        mask_out_of_bounds_samples: bool,
        inject_noise_sigma: float,
        inject_noise_seed: int,
        contract_coords: bool,
        contract_perc_foreground: float,
        disparity_at_inf: float,
        shape_representation: FastplaneShapeRepresentation,
        activation_fun: FastplaneActivationFun,
        use_separate_color_rep: bool,
    ):
        # Input shapes
        #
        # xy = [B x H x W x C] or [B x D x H x W x C]
        # yz = [B x W x D x C] or None
        # zx = [B x D x H x C] or None
        # weights = [N_LAYERS, C, C]
        # biases = [N_LAYERS, C]
        # weight_opacity = [C,]
        # bias_opacity = [1,]
        # weight_color = [C, C_OUT]
        # bias_color = [C_OUT,]
        # rays = [B x R x 3]
        # centers = [B x R x 3]

        assert BLOCK_SIZE >= MIN_BLOCK_SIZE

        device = xy.device
        num_channels_out = weight_color.shape[-1]
        num_rays = rays.shape[1:-1].numel()

        if shape_representation == FastplaneShapeRepresentation.TRIPLANE:
            batch_size, H, W, num_channels = xy.shape
            _, D, H, _ = yz.shape
            for plane_ in [xy, yz, zx] + ([xy_color, yz_color, zx_color] if use_separate_color_rep else []):
                assert plane_.shape[0] == batch_size
                assert plane_.shape[3] == num_channels
        elif shape_representation == FastplaneShapeRepresentation.VOXEL_GRID:
            batch_size, D, H, W, num_channels = xy.shape
            assert yz is None
            assert zx is None
            assert yz_color is None
            assert zx_color is None
            yz = torch.empty((1,), device=device)
            zx = torch.empty((1,), device=device)
            yz_color = torch.empty((1,), device=device)
            zx_color = torch.empty((1,), device=device)
            if use_separate_color_rep:
                assert xy.shape == xy_color.shape
            else:
                assert xy_color is None
                xy_color = torch.empty((1,), device=device)
        else:
            raise ValueError()

        assert math.log2(num_channels) % 1 == 0
        assert num_channels >= MIN_BLOCK_SIZE
        assert math.log2(num_channels_out) % 1 == 0
        assert num_channels_out >= MIN_BLOCK_SIZE

        assert weights.ndim == 3
        assert biases.ndim == 2
        assert weight_opacity.ndim == 1
        assert bias_opacity.ndim == 1
        assert weight_color.ndim == 2
        assert bias_color.ndim == 1
        assert inject_noise_seed >= 0

        assert weights.shape[0] == N_LAYERS + 1
        assert weights.shape[1] == num_channels
        assert weights.shape[2] == num_channels
        assert biases.shape[0] == N_LAYERS + 1
        assert biases.shape[1] == num_channels
        assert weight_opacity.shape[0] == num_channels
        assert weight_color.shape[0] == num_channels
        assert weight_color.shape[1] == num_channels_out
        assert bias_opacity.shape[0] == 1
        assert bias_color.shape[0] == num_channels_out
        assert rays.ndim == 3
        assert rays.shape[0] == batch_size
        assert rays.shape[1:-1].numel() == num_rays
        assert rays.shape[-1] == 3
        assert centers.ndim == 3
        assert centers.shape[0] == batch_size
        assert centers.shape[1:-1].numel() == num_rays
        assert centers.shape[-1] == 3
        assert rays_enc.shape[0] == batch_size
        assert rays_enc.shape[1:-1].numel() == num_rays
        assert rays_enc.shape[-1] == num_channels
        assert near.ndim == 2
        assert near.shape[0] == batch_size
        assert near.shape[1:].numel() == num_rays
        assert far.ndim == 2
        assert far.shape[0] == batch_size
        assert far.shape[1:].numel() == num_rays

        negative_log_transmittance = torch.zeros(batch_size, num_rays, device=device)
        expected_depth = torch.zeros(batch_size, num_rays, device=device)
        expected_features = torch.zeros(batch_size, num_rays, num_channels_out, device=device)
        inject_noise_seed_t = torch.full((batch_size, num_rays), inject_noise_seed, device=device, dtype=torch.long)

        n_blocks = math.ceil((num_rays * batch_size) / BLOCK_SIZE)
        grid = (n_blocks,)
        effective_num_samples = torch.zeros(n_blocks, device=device, dtype=torch.int32)

        _fw_kernel[grid](
            xy.contiguous(),
            yz.contiguous(),
            zx.contiguous(),
            xy_color.contiguous(),
            yz_color.contiguous(),
            zx_color.contiguous(),
            _flat(rays, 1),
            _flat(centers, 1),
            weights.contiguous(),
            biases.contiguous(),
            weight_opacity.contiguous(),
            bias_opacity.contiguous(),
            weight_color.contiguous(),
            bias_color.contiguous(),
            rays_enc.reshape(-1, num_channels).contiguous(),
            _flat(negative_log_transmittance, 0),
            _flat(expected_depth, 0),
            _flat(expected_features, 0),
            _flat(near, 0),
            _flat(far, 0),
            _flat(effective_num_samples, 0),
            num_samples,
            num_samples_inf,
            gain,
            batch_size,
            num_rays,
            num_channels,
            num_channels_out,
            H,
            W,
            D,
            BLOCK_SIZE,
            transmittance_thr,
            int(mask_out_of_bounds_samples),
            int(inject_noise_sigma > 0.0),
            inject_noise_sigma,
            inject_noise_seed_t,
            int(contract_coords),
            float(contract_perc_foreground),
            float(disparity_at_inf),
            int(shape_representation.value),
            int(activation_fun.value),
            int(use_separate_color_rep),
        )

        ctx.save_for_backward(
            xy,
            yz,
            zx,
            xy_color,
            yz_color,
            zx_color,
            negative_log_transmittance,
            weights,
            weight_opacity,
            weight_color,
            biases,
            bias_opacity,
            bias_color,
            rays,
            rays_enc,
            centers,
            near,
            far,
            effective_num_samples,
        )

        ctx.num_samples = num_samples
        ctx.num_samples_inf = num_samples_inf
        ctx.gain = gain
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.transmittance_thr = transmittance_thr
        ctx.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        ctx.inject_noise_sigma = inject_noise_sigma
        ctx.inject_noise_seed = inject_noise_seed
        ctx.contract_coords = contract_coords
        ctx.contract_perc_foreground = contract_perc_foreground
        ctx.disparity_at_inf = disparity_at_inf
        ctx.shape_representation = shape_representation
        ctx.activation_fun = activation_fun
        ctx.use_separate_color_rep = use_separate_color_rep

        # Denormalize the output
        expected_depth = expected_depth.reshape(rays.shape[:-1])
        negative_log_transmittance = negative_log_transmittance.reshape(rays.shape[:-1])
        expected_features = expected_features.reshape(*rays.shape[:-1], -1)

        return expected_depth, negative_log_transmittance, expected_features

    @staticmethod
    def backward(
        ctx,
        grad_expected_depth: torch.Tensor,
        grad_negative_log_transmittances: torch.Tensor,
        grad_expected_features: torch.Tensor,
    ):
        (
            xy,
            yz,
            zx,
            xy_color,
            yz_color,
            zx_color,
            negative_log_transmittance,
            weights,
            weight_opacity,
            weight_color,
            biases,
            bias_opacity,
            bias_color,
            rays,
            rays_enc,
            centers,
            near,
            far,
            effective_num_samples,
        ) = ctx.saved_tensors

        device = xy.device
        num_channels_out = weight_color.shape[1]
        num_rays = rays.shape[1:-1].numel()

        if ctx.shape_representation == FastplaneShapeRepresentation.TRIPLANE:
            batch_size, H, W, num_channels = xy.shape
            _, D, H, _ = yz.shape
            grad_xy = torch.zeros(batch_size, H, W, num_channels, device=device)
            grad_yz = torch.zeros(batch_size, D, H, num_channels, device=device)
            grad_zx = torch.zeros(batch_size, W, D, num_channels, device=device)
            if ctx.use_separate_color_rep:
                grad_xy_color = torch.zeros(batch_size, H, W, num_channels, device=device)
                grad_yz_color = torch.zeros(batch_size, D, H, num_channels, device=device)
                grad_zx_color = torch.zeros(batch_size, W, D, num_channels, device=device)
            else:
                grad_xy_color = None
                grad_yz_color = None
                grad_zx_color = None
        elif ctx.shape_representation == FastplaneShapeRepresentation.VOXEL_GRID:
            batch_size, D, H, W, num_channels = xy.shape
            grad_xy = torch.zeros(batch_size, D, H, W, num_channels, device=device)
            grad_yz = None
            grad_zx = None
            grad_yz_color = None
            grad_zx_color = None
            if ctx.use_separate_color_rep:
                grad_xy_color = torch.zeros(batch_size, D, H, W, num_channels, device=device)
            else:
                grad_xy_color = None
        else:
            raise ValueError()
        grad_weights = torch.zeros(N_LAYERS + 1, num_channels, num_channels, device=device)
        grad_biases = torch.zeros(N_LAYERS + 1, num_channels, device=device)
        grad_weight_opacity = torch.zeros(num_channels, device=device)
        grad_bias_opacity = torch.zeros(1, device=device)
        grad_weight_color = torch.zeros(num_channels_out, num_channels, device=device)
        grad_bias_color = torch.zeros(num_channels_out, device=device)
        grad_rays_enc = torch.zeros(batch_size, num_rays, num_channels, device=device)

        inject_noise_seed_t = torch.full(
            (batch_size, num_rays),
            ctx.inject_noise_seed,
            device=device,
            dtype=torch.long,
        )

        grid = (math.ceil((num_rays * batch_size) / ctx.BLOCK_SIZE),)

        result = _bw_kernel[grid](
            xy.contiguous(),
            yz.contiguous(),
            zx.contiguous(),
            xy_color.contiguous(),
            yz_color.contiguous(),
            zx_color.contiguous(),
            _flat(rays, 1),
            _flat(centers, 1),
            weights.contiguous(),
            biases.contiguous(),
            weight_opacity.contiguous(),
            bias_opacity.contiguous(),
            weight_color.contiguous(),
            bias_color.contiguous(),
            _flat(rays_enc, 1),
            _flat(negative_log_transmittance, 0),
            None,  # _flat(expected_depth, 0),
            None,  # _flat(expected_features, 0),
            _flat(near, 0),
            _flat(far, 0),
            _flat(effective_num_samples, 0),
            ctx.num_samples,
            ctx.num_samples_inf,
            ctx.gain,
            batch_size,
            num_rays,
            num_channels,
            num_channels_out,
            H,
            W,
            D,
            ctx.BLOCK_SIZE,
            ctx.transmittance_thr,
            int(ctx.mask_out_of_bounds_samples),
            int(ctx.inject_noise_sigma > 0.0),
            ctx.inject_noise_sigma,
            inject_noise_seed_t,
            int(ctx.contract_coords),
            float(ctx.contract_perc_foreground),
            float(ctx.disparity_at_inf),
            int(ctx.shape_representation.value),
            int(ctx.activation_fun.value),
            int(ctx.use_separate_color_rep),
            #
            grad_negative_log_transmittances.reshape(
                -1,
            ).contiguous(),
            grad_expected_depth.reshape(
                -1,
            ).contiguous(),
            grad_expected_features.reshape(-1, num_channels_out).contiguous(),
            #
            _flat_or_empty(grad_xy, 1, device),
            _flat_or_empty(grad_yz, 1, device),
            _flat_or_empty(grad_zx, 1, device),
            _flat_or_empty(grad_xy_color, 1, device),
            _flat_or_empty(grad_yz_color, 1, device),
            _flat_or_empty(grad_zx_color, 1, device),
            grad_weights,
            grad_biases,
            grad_weight_opacity,
            grad_bias_opacity,
            grad_weight_color,
            grad_bias_color,
            grad_rays_enc.reshape(-1, num_channels).contiguous(),
        )

        # Denormalize output
        grad_xy = grad_xy.reshape(xy.shape)
        if ctx.shape_representation == FastplaneShapeRepresentation.TRIPLANE:
            grad_yz = grad_yz.reshape(yz.shape)
            grad_zx = grad_zx.reshape(zx.shape)
            if ctx.use_separate_color_rep:
                grad_xy_color = grad_xy_color.reshape(xy_color.shape)
                grad_yz_color = grad_yz_color.reshape(yz_color.shape)
                grad_zx_color = grad_zx_color.reshape(zx_color.shape)
        elif ctx.shape_representation == FastplaneShapeRepresentation.VOXEL_GRID:
            grad_yz = None
            grad_zx = None
            grad_yz_color = None
            grad_zx_color = None
            if ctx.use_separate_color_rep:
                grad_xy_color = grad_xy_color.reshape(xy_color.shape)
            else:
                grad_xy_color = None
        else:
            raise ValueError()
        grad_rays_enc = grad_rays_enc.reshape(grad_rays_enc.shape)

        # grad_weight_color comes out transposed (Triton bugs ...)
        grad_weight_color = grad_weight_color.t().contiguous()

        # print("----")
        # for x_ in [
        #     grad_weights,
        #     grad_biases,
        #     grad_weight_opacity,
        #     grad_bias_opacity,
        #     grad_xy, grad_yz, grad_zx
        # ]:
        #     print(x_.norm())

        return (
            grad_xy,
            grad_yz,
            grad_zx,
            grad_xy_color,
            grad_yz_color,
            grad_zx_color,
            grad_weights,
            grad_biases,
            grad_weight_opacity,
            grad_bias_opacity,
            grad_weight_color,
            grad_bias_color,
            None,
            None,
            grad_rays_enc,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fastplane(
    xy: torch.Tensor,
    yz: Optional[torch.Tensor],
    zx: Optional[torch.Tensor],
    xy_color: Optional[torch.Tensor],
    yz_color: Optional[torch.Tensor],
    zx_color: Optional[torch.Tensor],
    weights: torch.Tensor,
    biases: torch.Tensor,
    weight_opacity: torch.Tensor,
    bias_opacity: torch.Tensor,
    weight_color: torch.Tensor,
    bias_color: torch.Tensor,
    rays: torch.Tensor,
    centers: torch.Tensor,
    rays_encoding: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    num_samples_inf: int,
    gain: float,
    BLOCK_SIZE: int,
    transmittance_thr: float,
    mask_out_of_bounds_samples: bool = False,
    inject_noise_sigma: float = 0.0,
    inject_noise_seed: Optional[int] = None,
    contract_coords: bool = False,
    contract_perc_foreground: float = 0.5,
    disparity_at_inf: float = 1e-5,
    shape_representation: FastplaneShapeRepresentation = FastplaneShapeRepresentation.TRIPLANE,
    activation_fun: FastplaneActivationFun = FastplaneActivationFun.SOFTPLUS,
):
    r"""Fast triplane"""

    if inject_noise_seed is None:
        if inject_noise_sigma > 0.0:
            inject_noise_seed = int(random.randint(0, 1000000))
        else:
            inject_noise_seed = 0

    use_separate_color_rep = xy_color is not None

    if not use_separate_color_rep:
        assert weights.shape[0] == N_LAYERS
        assert biases.shape[0] == N_LAYERS
        weights = torch.cat([weights, torch.zeros_like(weights[:1])], dim=0)
        biases = torch.cat([biases, torch.zeros_like(biases[:1])], dim=0)
    else:
        assert weights.shape[0] == N_LAYERS + 1
        assert biases.shape[0] == N_LAYERS + 1

    return FastplaneFunction.apply(
        xy,
        yz,
        zx,
        xy_color,
        yz_color,
        zx_color,
        weights,
        biases,
        weight_opacity,
        bias_opacity,
        weight_color,
        bias_color,
        rays,
        centers,
        rays_encoding,
        near,
        far,
        num_samples,
        num_samples_inf,
        gain,
        BLOCK_SIZE,
        transmittance_thr,
        mask_out_of_bounds_samples,
        inject_noise_sigma,
        inject_noise_seed,
        contract_coords,
        contract_perc_foreground,
        disparity_at_inf,
        shape_representation,
        activation_fun,
        use_separate_color_rep,
    )


def _contract_pi(x):
    n = x.abs().max(dim=-1).values[..., None]
    x_contract = torch.where(
        n <= 1.0,
        x,
        torch.where(
            (x.abs() - n).abs() <= 1e-7,
            (2 - 1 / x.abs()) * (x / x.abs()),
            x / n,
        ),
    )
    return x_contract / 2


def _depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)


def fastplane_naive(
    xy,
    yz,
    zx,
    xy_color,
    yz_color,
    zx_color,
    rays,
    centers,
    weights,
    biases,
    weight_opacity,
    bias_opacity,
    weight_color,  # C, 4
    bias_color,  # 4
    rays_encoding,  # ..., C
    near,
    far,
    num_samples,
    num_samples_inf,
    gain,
    transmittance_thr,
    BLOCK_SIZE,
    mask_out_of_bounds_samples,
    inject_noise_sigma,
    inject_noise_seed,
    contract_coords,
    disparity_at_inf,
    activation_fun,
):
    tot_num_samples = num_samples + num_samples_inf
    batch_size = xy.shape[0]
    num_rays = rays.shape[1]
    C_OUT = weight_color.shape[1]

    device = xy.device

    # depths = torch.linspace(near, far, num_samples).reshape(1, 1, -1).to(device)

    lsp = torch.linspace(0.0, 1.0, num_samples).to(device)
    depths = (near[:, None] + lsp[None, :] * (far - near)[:, None])[None]

    if num_samples_inf > 0:
        sph = torch.stack(
            [_depth_inv_sphere(far, disparity_at_inf, num_samples_inf, step) for step in range(num_samples_inf)],
            dim=-1,
        )
        depths = torch.cat([depths, sph[None]], dim=-1)

    samples = depths.unsqueeze(-1) * rays.unsqueeze(-2)
    samples = samples.reshape(batch_size, num_rays, tot_num_samples, 3)
    samples = samples + centers.unsqueeze(-2)

    if contract_coords:
        samples = _contract_pi(samples)

    # noise injection
    i1 = (
        tot_num_samples * torch.arange(num_rays, device=device)[:, None]
        + torch.arange(tot_num_samples, device=device)[None]
        + 1
    ).long()
    i2 = i1 + num_rays * tot_num_samples
    z = torch.zeros(i1.numel(), device=i1.device, dtype=torch.float32)
    grid = (triton.cdiv(i1.numel(), BLOCK_SIZE),)
    _int_to_randn_kernel[grid](i1.reshape(-1), i2.reshape(-1), z, i1.numel(), BLOCK_SIZE, inject_noise_seed)
    inject_noise = z * inject_noise_sigma

    delta_one = (far - near) / (num_samples - 1) if num_samples > 1 else 1
    delta = torch.cat([delta_one[None, :, None], depths.diff(dim=-1)], dim=-1)

    value, color = fastplane_eval_mlp(
        samples,
        xy,
        yz,
        zx,
        xy_color,
        yz_color,
        zx_color,
        weights,
        biases,
        weight_opacity,
        bias_opacity,
        weight_color,  # C, 4
        bias_color,  # 4
        rays_encoding,  # ..., C
        gain,
        mask_out_of_bounds_samples,
        activation_fun,
        inject_noise,
    )

    value = value[None, :, None] * delta
    value = torch.nn.functional.pad(value, (1, 0))

    negative_log_transmittances = torch.cumsum(value, dim=-1)
    transmittance = torch.exp(-negative_log_transmittances)

    # if True:
    #     # early ray stopping, need to take the BLOCK_SIZE into account as well
    #     with torch.no_grad():
    #         transmittance_re = transmittance.reshape(batch_size, -1, BLOCK_SIZE, num_samples+1)
    #         to_clamp = (
    #             transmittance_re <= transmittance_thr
    #         ).all(dim=2, keepdim=True).expand(
    #             transmittance_re.shape
    #         )
    #         to_clamp_mask = to_clamp.reshape(batch_size, -1, num_samples+1).float()

    #     import pdb; pdb.set_trace()

    #     value = (1 - to_clamp_mask) * value
    #     negative_log_transmittances = torch.cumsum(value, dim=-1)
    #     transmittance = torch.exp(-negative_log_transmittances)

    rweights = -transmittance.diff(dim=-1)
    # rweights = rweights * (1-to_clamp_mask[:, :, :-1])

    expected_depth = (depths * rweights).sum(dim=-1)
    expected_features = (color * rweights[..., None]).sum(-2)
    negative_log_transmittance = negative_log_transmittances[..., -1]

    return (
        expected_depth,
        negative_log_transmittance,
        expected_features,
    )


def _sample_shape_rep(
    xy,
    yz,
    zx,
    samples,
    mask_out_of_bounds_samples,
):
    if xy.ndim == 5:  # voxel grid
        assert yz is None
        assert zx is None
        vec = torch.nn.functional.grid_sample(xy, samples[..., None, :], align_corners=False)[..., 0]
        _, _, D, H, W = xy.shape

    else:
        _, _, H, W = xy.shape
        _, _, D, H = yz.shape
        a = torch.nn.functional.grid_sample(xy, samples[:, :, :, [0, 1]], align_corners=False)
        b = torch.nn.functional.grid_sample(yz, samples[:, :, :, [1, 2]], align_corners=False)
        c = torch.nn.functional.grid_sample(zx, samples[:, :, :, [2, 0]], align_corners=False)
        vec = a + b + c

    if mask_out_of_bounds_samples:
        vec = vec * _oob_mask(samples, D, H, W).to(vec)[:, None]

    return vec


def _oob_mask(samples, D, H, W):
    x, y, z = samples.unbind(dim=-1)
    ix = ((x + 1) / 2) * W - 0.5
    iy = ((y + 1) / 2) * H - 0.5
    iz = ((z + 1) / 2) * D - 0.5
    in_bounds = (iy >= 0) * (iy < H) * (ix >= 0) * (ix < W) * (iz >= 0) * (iz < D)
    return in_bounds


def fastplane_eval_mlp(
    samples,
    xy,
    yz,
    zx,
    xy_color,
    yz_color,
    zx_color,
    weights,
    biases,
    weight_opacity,
    bias_opacity,
    weight_color,  # C, 4
    bias_color,  # 4
    rays_encoding,  # ..., C
    gain,
    mask_out_of_bounds_samples,
    activation_fun,
    inject_noise=None,
):
    if xy.ndim == 5:  # voxel grid
        _, _, D, H, W = xy.shape
    else:
        _, _, H, W = xy.shape
        _, _, D, H = yz.shape

    vec = _sample_shape_rep(
        xy,
        yz,
        zx,
        samples,
        mask_out_of_bounds_samples,
    )

    vec = torch.relu(torch.nn.functional.conv2d(vec, weights[0][:, :, None, None], biases[0]))
    vec = torch.relu(torch.nn.functional.conv2d(vec, weights[1][:, :, None, None], biases[1]))

    value = (vec * weight_opacity[None, :, None, None]).sum(dim=1) + bias_opacity

    if inject_noise is not None:
        value = value + inject_noise.reshape(value.shape)

    if activation_fun == FastplaneActivationFun.SOFTPLUS:
        value = torch.nn.functional.softplus(value)
    elif activation_fun == FastplaneActivationFun.RELU:
        value = torch.nn.functional.relu(value)
    else:
        raise ValueError()
    value = gain * value

    if xy_color is not None:
        vec_color = _sample_shape_rep(xy_color, yz_color, zx_color, samples, mask_out_of_bounds_samples)
        vec_color = vec_color + rays_encoding.permute(0, 2, 1)[..., None]
        if mask_out_of_bounds_samples:
            vec_color = vec_color * _oob_mask(samples, D, H, W).to(vec_color)[:, None]
        vec_color = torch.relu(torch.nn.functional.conv2d(vec_color, weights[2][:, :, None, None], biases[2]))
        vec_color = torch.relu(torch.nn.functional.conv2d(vec_color, weights[3][:, :, None, None], biases[3]))
    else:
        vec_color = torch.relu(
            torch.nn.functional.conv2d(vec, weights[2][:, :, None, None], biases[2])
            + rays_encoding.permute(0, 2, 1)[..., None]
        )
    log_color = torch.nn.functional.conv2d(vec_color, weight_color.t()[:, :, None, None], bias_color)
    color = torch.sigmoid(log_color).permute(0, 2, 3, 1)

    return value, color
