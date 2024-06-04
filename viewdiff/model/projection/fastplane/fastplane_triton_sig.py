# Copyright 2024 (c) Meta Platforms, Inc. and affiliates.
# Inspired by Lightplane: https://github.com/facebookresearch/lightplane/tree/main
#
# BSD License
#
# For Lightplane software
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import triton
import triton.language as tl


ALLOW_TF32 = False  # A100s cause illegal memory access when this is enabled


@triton.jit
def _sample_2d(
    image,
    w,
    batch_index,
    ix,
    iy,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    val = tl.view(
        tl.load((image + batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :]),
        (BLOCK_SIZE, C),
    )
    return val * tl.view(
        (w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32))[:, None] * channel_bcast,
        (BLOCK_SIZE, C),
    )


@triton.jit
def _is_in_bounds(
    x,
    y,
    z,
    W: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((x + 1) / 2) * W - 0.5
    iy = ((y + 1) / 2) * H - 0.5
    iz = ((z + 1) / 2) * D - 0.5
    in_bounds = (iy >= 0) * (iy < H) * (ix >= 0) * (ix < W) * (iz >= 0) * (iz < D)
    in_bounds_mask = tl.broadcast_to(in_bounds.to(tl.float32)[:, None], (BLOCK_SIZE, C))
    return in_bounds_mask


@triton.jit
def _int_to_randn_kernel(
    x1,
    x2,
    out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    seed: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < N
    x1_buffer = tl.load(x1 + offs, mask=offs_mask).to(tl.uint32)
    x2_buffer = tl.load(x2 + offs, mask=offs_mask).to(tl.uint32)
    seed_buffer = tl.full((BLOCK_SIZE,), seed, dtype=tl.int64).to(tl.uint32)
    r = _int_to_randn(x1_buffer, x2_buffer, seed_buffer)
    tl.store(out + offs, r, mask=offs_mask)


@triton.jit
def _hash(x):  # x is tl.uint32
    # https://stackoverflow.com/a/12996028
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


@triton.jit
def _pair_hash(x, h):  # x, h is tl.uint32
    # https://stackoverflow.com/a/30057527
    h = h ^ x
    h = (h << 24) + h * 0x193
    return h


@triton.jit
def _int_to_randn(x1, x2, seed):  # x is tl.int32
    # convert two integers to a float which is randomly-normally distributed
    # 1) hash both ints to a uniformly distributed uint32
    # 2) divide by max uint32 to map to [0, 1] (uniform distribution due to hashing fun)
    # 3) box-muller transform to map U[0, 1] to N(0, 1)
    x_hash_1 = _hash(x1.to(tl.uint32))
    x_hash_2 = _hash(x2.to(tl.uint32))
    # alter the hashes with the seed: https://stackoverflow.com/a/30057527
    # x_hash_1 = _pair_hash(seed, x_hash_1)
    # x_hash_2 = _pair_hash(seed + 1, x_hash_2)
    x_hash_1 = _pair_hash(_pair_hash(2166136261, seed), x_hash_1)  # slower+stronger
    x_hash_2 = _pair_hash(_pair_hash(2166136261, seed + 1), x_hash_2)
    # transform to [0, 1]
    x_01_1 = (x_hash_1.to(tl.float32) + 10) / (4294967295.0 + 10)  # 4294967295.0 = max uint32
    x_01_2 = (x_hash_2.to(tl.float32) + 10) / (4294967295.0 + 10)
    # box-muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    z = tl.sqrt(-2 * tl.log(x_01_1)) * tl.cos(6.28318530718 * x_01_2)
    return z


@triton.jit()
def _get_sample_randn(pid, step, n_rays, n_steps, BLOCK_SIZE, seed_buffer):
    offs = pid * BLOCK_SIZE * n_steps + 1
    i1 = offs + step + tl.arange(0, BLOCK_SIZE) * n_steps
    i2 = n_rays * n_steps + i1
    return _int_to_randn(i1, i2, seed_buffer)


@triton.jit
def _grid_sample(
    image,
    batch_index,
    ix,
    iy,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5

    ix_nw = (ix - ix % 1).to(tl.float32)  # floor
    iy_nw = (iy - iy % 1).to(tl.float32)  # floor

    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    out_val = (
        _sample_2d(image, nw, batch_index, ix_nw, iy_nw, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, ne, batch_index, ix_ne, iy_ne, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, se, batch_index, ix_se, iy_se, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, sw, batch_index, ix_sw, iy_sw, IH, IW, C, BLOCK_SIZE)
    )

    return out_val


@triton.jit
def _splat_2d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    w = tl.view(
        (w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32))[:, None] * channel_bcast,
        (BLOCK_SIZE, C),
    )
    offs = tl.view(
        (batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _grid_splat(
    to_splat,
    grad_image,
    batch_index,
    ix,
    iy,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5

    ix_nw = (ix - ix % 1).to(tl.float32)  # floor
    iy_nw = (iy - iy % 1).to(tl.float32)  # floor

    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    _splat_2d(to_splat, grad_image, nw, batch_index, ix_nw, iy_nw, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, ne, batch_index, ix_ne, iy_ne, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, sw, batch_index, ix_sw, iy_sw, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, se, batch_index, ix_se, iy_se, IH, IW, C, BLOCK_SIZE)


@triton.jit
def _splat_3d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)
    w = tl.view(
        w
        * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)).to(tl.float32)[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )
    offs = tl.view(
        (batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _voxel_grid_splat(
    to_splat,
    grad_image,
    batch_index,
    ix,
    iy,
    iz,
    N: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: use this instead:
    #   - https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/torch_utils/ops/grid_sample_gradfix.py

    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5
    iz = ((iz + 1) / 2) * ID - 0.5

    # ijk = xyz.astype(np.int32)
    # i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]

    ix0 = (ix - ix % 1).to(tl.float32)  # floor
    iy0 = (iy - iy % 1).to(tl.float32)  # floor
    iz0 = (iz - iz % 1).to(tl.float32)  # floor

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    V000x = ix0
    V000y = iy0
    V000z = iz0

    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1

    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0

    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0

    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1

    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0

    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1

    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1

    # xyz = xyz - ijk

    x = ix - ix0
    y = iy - iy0
    z = iz - iz0

    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * (1 - y) * (1 - z),
        batch_index,
        V000x,
        V000y,
        V000z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * (1 - y) * z,
        batch_index,
        V100x,
        V100y,
        V100z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * y * (1 - z),
        batch_index,
        V010x,
        V010y,
        V010z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * (1 - y) * (1 - z),
        batch_index,
        V001x,
        V001y,
        V001z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * (1 - y) * z,
        batch_index,
        V101x,
        V101y,
        V101z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * y * (1 - z),
        batch_index,
        V011x,
        V011y,
        V011z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * y * z,
        batch_index,
        V110x,
        V110y,
        V110z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * y * z,
        batch_index,
        V111x,
        V111y,
        V111z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )


@triton.jit
def _sample_3d(
    image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)
    val = tl.view(
        tl.load(
            (image + batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None]
            + Coffs[None, :]
        ),
        (BLOCK_SIZE, C),
    )
    return val * tl.view(
        (w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)).to(tl.float32))[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )


@triton.jit
def _voxel_grid_sample(
    image,
    batch_index,
    ix,
    iy,
    iz,
    N: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5
    iz = ((iz + 1) / 2) * ID - 0.5

    # ijk = xyz.astype(np.int32)
    # i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    ix0 = (ix - ix % 1).to(tl.float32)  # floor
    iy0 = (iy - iy % 1).to(tl.float32)  # floor
    iz0 = (iz - iz % 1).to(tl.float32)  # floor

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    V000x = ix0
    V000y = iy0
    V000z = iz0

    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1

    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0

    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0

    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1

    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0

    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1

    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1

    # xyz = xyz - ijk

    x = ix - ix0
    y = iy - iy0
    z = iz - iz0

    # Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
    #         + V100 * x * (1 - y) * (1 - z) +
    #         + V010 * (1 - x) * y * (1 - z) +
    #         + V001 * (1 - x) * (1 - y) * z +
    #         + V101 * x * (1 - y) * z +
    #         + V011 * (1 - x) * y * z +
    #         + V110 * x * y * (1 - z) +
    #         + V111 * x * y * z)

    out_val = (
        _sample_3d(
            image,
            (1 - x) * (1 - y) * (1 - z),
            batch_index,
            V000x,
            V000y,
            V000z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * (1 - y) * z,
            batch_index,
            V100x,
            V100y,
            V100z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * y * (1 - z),
            batch_index,
            V010x,
            V010y,
            V010z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * (1 - y) * (1 - z),
            batch_index,
            V001x,
            V001y,
            V001z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * (1 - y) * z,
            batch_index,
            V101x,
            V101y,
            V101z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * y * (1 - z),
            batch_index,
            V011x,
            V011y,
            V011z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * y * z,
            batch_index,
            V110x,
            V110y,
            V110z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * y * z,
            batch_index,
            V111x,
            V111y,
            V111z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
    )

    return out_val


@triton.jit
def _sample_grid_rep(
    xy,
    yz,
    zx,
    batch_index,
    sample_x,
    sample_y,
    sample_z,
    batch_size: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    shape_representation: tl.constexpr,
):
    if shape_representation == 0:
        a = _grid_sample(xy, batch_index, sample_x, sample_y, batch_size, C, H, W, BLOCK_SIZE)
        b = _grid_sample(yz, batch_index, sample_y, sample_z, batch_size, C, D, H, BLOCK_SIZE)
        c = _grid_sample(zx, batch_index, sample_z, sample_x, batch_size, C, W, D, BLOCK_SIZE)
        vec = a + b + c
    else:  # shape_representation==1
        vec = _voxel_grid_sample(
            xy,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
        )

    vec = tl.view(vec, (BLOCK_SIZE, C))

    return vec


@triton.jit
def _splat_grid_rep(
    to_splat,
    xy,
    yz,
    zx,
    batch_index,
    sample_x,
    sample_y,
    sample_z,
    batch_size: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    shape_representation: tl.constexpr,
):
    if shape_representation == 0:
        _grid_splat(
            to_splat,
            xy,
            batch_index,
            sample_x,
            sample_y,
            batch_size,
            C,
            H,
            W,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            yz,
            batch_index,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            zx,
            batch_index,
            sample_z,
            sample_x,
            batch_size,
            C,
            W,
            D,
            BLOCK_SIZE,
        )
    else:  # shape_representation==1
        _voxel_grid_splat(
            to_splat,
            xy,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
        )


@triton.jit
def _color_activation(x):
    return tl.sigmoid(x)


@triton.jit
def _d_color_activation(dy, x):
    return dy * tl.sigmoid(x) * (1 - tl.sigmoid(x))


@triton.jit
def _softplus(x):
    return tl.where(x >= 0, x + tl.log(1 + tl.exp(-x)), tl.log(1 + tl.exp(x)))


@triton.jit
def _d_softplus(grad, x):
    #   z = (x >= 0) ? 1 / (1 + exp(-x)) : (1 - 1 / (1 + exp(x)));
    z = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), 1 - 1 / (1 + tl.exp(x)))
    return grad * z


@triton.jit
def _d_linear_relu(d_y, w, b, xwb, x):
    # gradients of `y = max(x @ w + b, 0); xwb = x @ w + b`
    d_y_relu = d_y * (xwb > 0.0).to(tl.float32)
    return _d_linear(d_y_relu, w, b, x)


@triton.jit
def _d_linear(d_y, w, b, x):
    # gradients of `y = x @ w + b;
    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)
    d_w = tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32)
    d_b = tl.sum(d_y, axis=0)
    return d_x, d_w, d_b


@triton.jit
def _load_mlp_bias_bcast(
    biases,
    C,
    offs,
    BLOCK_SIZE,
):
    return tl.view(
        tl.load((biases + offs + tl.arange(0, C))[None, :] + tl.zeros((BLOCK_SIZE, 1), dtype=tl.int32)),
        (BLOCK_SIZE, C),
    )


@triton.jit()
def _load_mlp_weights(
    weights,
    biases,
    weight_opacity,
    bias_opacity,
    weight_color,
    bias_color,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # MLP weights, biases
    #w1 = tl.view(tl.load(weights + tl.arange(0, C * C)), (C, C))
    ptrs = tl.view(tl.arange(0, C * C), (C, C))
    w1 = tl.load(weights + ptrs)
    b1 = _load_mlp_bias_bcast(biases, C, 0, BLOCK_SIZE)

    #w2 = tl.view(tl.load(weights + C * C + tl.arange(0, C * C)), (C, C))
    w2 = tl.load(weights + C * C + ptrs)
    b2 = _load_mlp_bias_bcast(biases, C, C, BLOCK_SIZE)

    #wr = tl.view(tl.load(weights + 2 * C * C + tl.arange(0, C * C)), (C, C))
    wr = tl.load(weights + 2 * C * C + ptrs)
    br = _load_mlp_bias_bcast(biases, C, 2 * C, BLOCK_SIZE)

    #w2c = tl.view(tl.load(weights + 3 * C * C + tl.arange(0, C * C)), (C, C))
    w2c = tl.load(weights + 3 * C * C + ptrs)
    b2c = _load_mlp_bias_bcast(biases, C, 3 * C, BLOCK_SIZE)

    # color head weights, biases
    wc = tl.view(
        tl.load(weight_color + tl.arange(0, C)[:, None] * C_OUT + tl.arange(0, C_OUT)[None, :]),
        (C, C_OUT),
    )
    bc = _load_mlp_bias_bcast(bias_color, C_OUT, 0, BLOCK_SIZE)

    # opacity head weights, biases
    wo = tl.view(  # super annoying we cannot broadcast this easily
        tl.load(weight_opacity + (tl.arange(0, C)[None, :] + tl.zeros((BLOCK_SIZE, 1), dtype=tl.int32))),
        (BLOCK_SIZE, C),
    )
    bo = tl.view(tl.load(bias_opacity + tl.zeros((BLOCK_SIZE, 1), dtype=tl.int32)), (BLOCK_SIZE,))

    return w1, w2, wr, wo, wc, b1, b2, br, bo, bc, w2c, b2c


@triton.jit
def _contract_pi(x, y, z, perc_foreground):
    # MERF contract_pi function
    # def contract_pi(x):
    #     n = x.abs().max(dim=-1).values[..., None]
    #     x_contract = torch.where(
    #         n <= 1.0,
    #         x,
    #         torch.where(
    #             (x.abs()-n).abs() <= 1e-7,
    #             (2 - 1/x.abs()) * (x / x.abs()),
    #             x / n,
    #         )
    #     )
    #     return x_contract
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n, perc_foreground)
    y_c = _contract_pi_one(y, n, perc_foreground)
    z_c = _contract_pi_one(z, n, perc_foreground)
    return x_c, y_c, z_c


@triton.jit
def _contract_pi_one(x, n, perc_foreground):
    x_c = tl.where(
        n <= 1.0,
        x,
        tl.where(
            tl.abs(tl.abs(x) - n) <= 1e-8,
            ((1.0 / perc_foreground) - ((1.0 / perc_foreground) - 1) / tl.abs(x)) * (x / tl.abs(x)),
            x / n,
        ),
    )
    # important: we map the contracted coords from [- (1 / perc_foreground), (1 / perc_foreground)] to [-1, 1]!
    x_c = x_c * perc_foreground
    return x_c


@triton.jit
def _depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)


@triton.jit
def _depth_lin(near, far, n, step):
    frac_step = step / (n - 1)
    return (far - near) * frac_step + near


@triton.jit
def _fw_kernel(
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
    weight_color,
    bias_color,
    rays_encoding,
    negative_log_transmittance,
    expected_depth,
    expected_features,
    near,
    far,
    effective_num_samples,
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    gain: tl.constexpr,
    batch_size: tl.constexpr,
    num_rays_per_batch: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    transmittance_thr: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
    inject_noise: tl.constexpr,
    inject_noise_sigma: tl.constexpr,
    inject_noise_seed,
    contract_coords: tl.constexpr,
    contract_perc_foreground: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    shape_representation: tl.constexpr,
    activation_fun: tl.constexpr,  # 0 - softplus, 1 - relu
    use_separate_color_rep: tl.constexpr,
):
    # shape_representation:
    #   0: triplane
    #   1: voxel grid, yz, zx are ignored
    tot_num_samples = num_samples + num_samples_inf
    num_rays = num_rays_per_batch * batch_size
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_features = pid * BLOCK_SIZE * OUT_C + OUT_C * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, OUT_C)[None, :]
    offs_features_mask = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]) < num_rays

    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays_per_batch

    near_buffer = tl.load(near + offs, mask=offs < num_rays).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays).to(tl.float32)
    effective_num_samples_buffer = tl.zeros((1,), dtype=tl.int32)

    # delta = (far_buffer - near_buffer) / (num_samples - 1)
    depth = near_buffer

    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays).to(tl.uint32)
    sample_index_buffer = tl.arange(0, BLOCK_SIZE) * tot_num_samples + pid * BLOCK_SIZE * tot_num_samples + 1

    expected_depth_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    expected_features_buffer = tl.zeros((BLOCK_SIZE, OUT_C), dtype=tl.float32)
    prev_transmittance = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    negative_log_transmittance_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    w1, w2, wr, wo, wc, b1, b2, br, bo, bc, w2c, b2c = _load_mlp_weights(
        weights,
        biases,
        weight_opacity,
        bias_opacity,
        weight_color,
        bias_color,
        C,
        OUT_C,
        BLOCK_SIZE,
    )

    rays_encoding_buffer = tl.load(
        rays_encoding + pid * BLOCK_SIZE * C + C * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, C)[None, :]
    )

    transmittance = tl.exp(-negative_log_transmittance_buffer)
    zero_value = tl.zeros((BLOCK_SIZE,), tl.float32)
    zero_color = tl.zeros((BLOCK_SIZE, OUT_C), tl.float32)

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
            depth_prev = _depth_lin(near_buffer, far_buffer, num_samples, step - 1)
        else:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )
            depth_prev = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples - 1,
            )
        delta = depth - depth_prev

        if tl.sum(transmittance > transmittance_thr, axis=0):
            # At least one transmittance is above eps, so we continue as normal.
            sample_x = center_x + depth * ray_x
            sample_y = center_y + depth * ray_y
            sample_z = center_z + depth * ray_z

            if contract_coords:
                sample_x, sample_y, sample_z = _contract_pi(sample_x, sample_y, sample_z, contract_perc_foreground)

            vec = _sample_grid_rep(
                xy,
                yz,
                zx,
                batch_index,
                sample_x,
                sample_y,
                sample_z,
                batch_size,
                C,
                D,
                H,
                W,
                BLOCK_SIZE,
                shape_representation,
            )

            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
                vec = vec * in_bounds_mask

            # mlp
            vec = tl.maximum(tl.dot(vec, w1, allow_tf32=ALLOW_TF32) + b1, 0.0)
            vec = tl.maximum(tl.dot(vec, w2, allow_tf32=ALLOW_TF32) + b2, 0.0)

            # final opacity value
            value = tl.view(tl.sum(wo * vec, axis=1), (BLOCK_SIZE,)) + bo

            if inject_noise:
                r = _int_to_randn(
                    sample_index_buffer,
                    sample_index_buffer + num_rays * tot_num_samples,
                    seed_buffer,
                )
                value = value + r * inject_noise_sigma

            if activation_fun == 0:
                value_act = _softplus(value)
            else:  # activation_fun==1
                value_act = tl.maximum(value, 0.0)

            value = delta * gain * value_act

            # colors
            if use_separate_color_rep:
                vec_color = _sample_grid_rep(
                    xy_color,
                    yz_color,
                    zx_color,
                    batch_index,
                    sample_x,
                    sample_y,
                    sample_z,
                    batch_size,
                    C,
                    D,
                    H,
                    W,
                    BLOCK_SIZE,
                    shape_representation,
                )

                # color mlp
                vec_color = vec_color + rays_encoding_buffer

                if mask_out_of_bounds_samples:
                    in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
                    vec_color = vec_color * in_bounds_mask

                vec_color1 = tl.maximum(tl.dot(vec_color, wr, allow_tf32=ALLOW_TF32) + br, 0.0)
                vec_color2 = tl.maximum(tl.dot(vec_color1, w2c, allow_tf32=ALLOW_TF32) + b2c, 0.0)
                log_color = tl.dot(vec_color2, wc, allow_tf32=ALLOW_TF32) + bc
            else:
                vecr = tl.maximum(tl.dot(vec, wr, allow_tf32=ALLOW_TF32) + br + rays_encoding_buffer, 0.0)
                log_color = tl.dot(vecr, wc, allow_tf32=ALLOW_TF32) + bc

            color = _color_activation(log_color)
            effective_ns_increment = 1

        else:
            # Transmittance is very low everywhere -> we skip all MLP passes
            # and render 0 colors/opacity.
            value = zero_value
            color = zero_color
            effective_ns_increment = 0
            # tl.printf("skipping samples ", step)

        # neg log transmittance
        negative_log_transmittance_buffer = negative_log_transmittance_buffer + value
        transmittance = tl.exp(-negative_log_transmittance_buffer)
        render_weights = prev_transmittance - transmittance

        # exp depth
        expected_depth_buffer = expected_depth_buffer + render_weights * depth

        # render weights
        render_weights_bcast = tl.broadcast_to(prev_transmittance[:, None], (BLOCK_SIZE, OUT_C)) - tl.broadcast_to(
            transmittance[:, None], (BLOCK_SIZE, OUT_C)
        )
        feature_render = color * render_weights_bcast

        expected_features_buffer = expected_features_buffer + feature_render
        prev_transmittance = transmittance
        # depth = depth + delta
        # if step >= (num_samples - 1):
        #     # inverse-sphere depth
        #     depth = _depth_inv_sphere(
        #         far_buffer,
        #         disparity_at_inf,
        #         num_samples_inf,
        #         step - num_samples + 1,
        #     )
        sample_index_buffer = sample_index_buffer + 1

        # add the effective far value
        effective_num_samples_buffer = effective_num_samples_buffer + effective_ns_increment

    # tl.printf("T ", tl.exp(-negative_log_transmittance_buffer))

    tl.store(
        negative_log_transmittance + offs,
        negative_log_transmittance_buffer,
        mask=offs < num_rays,
    )
    tl.store(expected_depth + offs, expected_depth_buffer, mask=offs < num_rays)
    tl.store(effective_num_samples + pid + tl.arange(0, 1), effective_num_samples_buffer)
    tl.store(
        expected_features + offs_features,
        expected_features_buffer,
        mask=offs_features_mask,
    )


@triton.jit
def _bw_kernel(
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
    weight_color,
    bias_color,
    rays_encoding,
    negative_log_transmittance,
    expected_depth,
    expected_features,
    near,
    far,
    effective_num_samples,
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    gain: tl.constexpr,
    batch_size: tl.constexpr,
    num_rays_per_batch: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    transmittance_thr: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
    inject_noise: tl.constexpr,
    inject_noise_sigma: tl.constexpr,
    inject_noise_seed,
    contract_coords: tl.constexpr,
    contract_perc_foreground: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    shape_representation: tl.constexpr,
    activation_fun: tl.constexpr,  # 0 - softplus, 1 - relu
    use_separate_color_rep: tl.constexpr,
    # ----
    grad_negative_log_transmittance,
    grad_expected_depth,
    grad_expected_features,
    # ----
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
    grad_rays_enc,
):
    tot_num_samples = num_samples + num_samples_inf
    num_rays = num_rays_per_batch * batch_size
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_features = pid * BLOCK_SIZE * OUT_C + OUT_C * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, OUT_C)[None, :]
    offs_features_mask = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]) < num_rays

    offs_CC = tl.arange(0, C)[:, None] * C + tl.arange(0, C)[None, :]

    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    rays_enc_offs = pid * BLOCK_SIZE * C + C * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, C)[None, :]
    rays_enc_mask = rays_enc_offs < num_rays * C
    rays_encoding_buffer = tl.load(rays_encoding + rays_enc_offs, mask=rays_enc_mask)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays_per_batch

    near_buffer = tl.load(near + offs, mask=offs < num_rays).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays).to(tl.float32)

    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays).to(tl.uint32)

    # seed_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.int64).to(tl.uint32)
    sample_index_buffer = (
        tl.arange(0, BLOCK_SIZE) * tot_num_samples
        + pid * BLOCK_SIZE * tot_num_samples
        + 1
        + tot_num_samples
        - 1  # we count from the top
    )

    depth = far_buffer

    # input grad buffers
    grad_negative_log_transmittance_buffer = tl.load(
        grad_negative_log_transmittance + offs, mask=offs_mask, other=0.0
    ).to(tl.float32)
    grad_expected_features_buffer = tl.load(
        grad_expected_features + offs_features, mask=offs_features_mask, other=0.0
    ).to(tl.float32)
    grad_expected_depth_buffer = tl.load(grad_expected_depth + offs, mask=offs_mask, other=0.0).to(tl.float32)

    # intermediate buffers
    prev_proj_depth = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    prev_proj_features = tl.zeros((BLOCK_SIZE, OUT_C), dtype=tl.float32)
    negative_log_transmittance_buffer = tl.load(negative_log_transmittance + offs, mask=offs_mask, other=0.0).to(
        tl.float32
    )
    transmittance = tl.exp(-negative_log_transmittance_buffer)
    prev_grad_opacity = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # grad weight buffers
    d_w2 = tl.zeros((C, C), dtype=tl.float32)
    d_w1 = tl.zeros((C, C), dtype=tl.float32)
    d_b1 = tl.zeros((C,), dtype=tl.float32)
    d_b2 = tl.zeros((C,), dtype=tl.float32)
    d_w2c = tl.zeros((C, C), dtype=tl.float32)
    d_b2c = tl.zeros((C,), dtype=tl.float32)
    d_wr = tl.zeros((C, C), dtype=tl.float32)
    d_br = tl.zeros((C,), dtype=tl.float32)
    d_wo = tl.zeros((C,), dtype=tl.float32)
    d_bo = tl.zeros((1,), dtype=tl.float32)
    d_wc = tl.zeros((C, OUT_C), dtype=tl.float32)
    d_wc = tl.zeros((OUT_C, C), dtype=tl.float32)
    d_bc = tl.zeros((OUT_C,), dtype=tl.float32)
    d_rays_enc = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    zero_w = tl.zeros((C, C), dtype=tl.float32)
    zero_b = tl.zeros((C,), dtype=tl.float32)
    zero_vec = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)

    w1, w2, wr, wo, wc, b1, b2, br, bo, bc, w2c, b2c = _load_mlp_weights(
        weights,
        biases,
        weight_opacity,
        bias_opacity,
        weight_color,
        bias_color,
        C,
        OUT_C,
        BLOCK_SIZE,
    )

    prev_transmittance = transmittance
    for step in range(tot_num_samples):
        if step < num_samples_inf:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                num_samples_inf - step - 1,
            )
            depth_prev = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                num_samples_inf - step - 2,
            )
        else:
            depth = _depth_lin(
                near_buffer,
                far_buffer,
                num_samples,
                num_samples - (step - num_samples_inf) - 1,
            )
            depth_prev = _depth_lin(
                near_buffer,
                far_buffer,
                num_samples,
                num_samples - (step - num_samples_inf) - 2,
            )
        delta = depth - depth_prev

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(sample_x, sample_y, sample_z, contract_perc_foreground)

        vec = _sample_grid_rep(
            xy,
            yz,
            zx,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
            shape_representation,
        )

        if mask_out_of_bounds_samples:
            in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
            vec = vec * in_bounds_mask

        # mlp
        vec1 = tl.maximum(tl.dot(vec, w1, allow_tf32=ALLOW_TF32) + b1, 0.0)
        vec2 = tl.maximum(tl.dot(vec1, w2, allow_tf32=ALLOW_TF32) + b2, 0.0)

        # final opacity value
        value = tl.sum(vec2 * wo, axis=1) + bo

        if inject_noise:
            # the steps count downwards ...
            r = _int_to_randn(
                sample_index_buffer,
                sample_index_buffer + num_rays * tot_num_samples,
                seed_buffer,
            )
            value = value + r * inject_noise_sigma

        if activation_fun == 0:
            value_act = _softplus(value)
        else:  # activation_fun==1
            value_act = tl.maximum(value, 0.0)

        delta_value = gain * value_act * delta

        # colors
        if use_separate_color_rep:
            vec_color = _sample_grid_rep(
                xy_color,
                yz_color,
                zx_color,
                batch_index,
                sample_x,
                sample_y,
                sample_z,
                batch_size,
                C,
                D,
                H,
                W,
                BLOCK_SIZE,
                shape_representation,
            )

            # plug in the rays encodings here
            vec_color = vec_color + rays_encoding_buffer

            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
                vec_color = vec_color * in_bounds_mask

            vec_color1 = tl.maximum(tl.dot(vec_color, wr, allow_tf32=ALLOW_TF32) + br, 0.0)
            vecr = tl.maximum(tl.dot(vec_color1, w2c, allow_tf32=ALLOW_TF32) + b2c, 0.0)
        else:
            vecr = tl.maximum(tl.dot(vec2, wr, allow_tf32=ALLOW_TF32) + br + rays_encoding_buffer, 0.0)

        log_color = tl.dot(vecr, wc, allow_tf32=ALLOW_TF32) + bc
        color = _color_activation(log_color)

        # grads
        proj_features = color * grad_expected_features_buffer
        proj_depth = depth * grad_expected_depth_buffer

        prev_transmittance = transmittance

        opacity_grad_now = prev_transmittance * (
            (proj_depth - prev_proj_depth) + tl.sum(proj_features - prev_proj_features, axis=1)
        )

        prev_grad_opacity = prev_grad_opacity + opacity_grad_now

        grad_value_act = delta * (prev_grad_opacity + grad_negative_log_transmittance_buffer)
        if activation_fun == 0:
            grad_value_act = _d_softplus(grad_value_act, value)
        else:  # activation_fun==1
            grad_value_act = grad_value_act * (value > 0.0).to(tl.float32)

        #grad_value = tl.view(gain * grad_value_act, (BLOCK_SIZE, 1))
        grad_value = gain * grad_value_act
        grad_value = tl.expand_dims(grad_value, 1)

        # grad opacity head
        d_wo_ = tl.sum(vec2 * tl.broadcast_to(grad_value, (BLOCK_SIZE, C)), axis=0)
        d_bo_ = tl.sum(grad_value, axis=0)
        d_vec2_1 = wo * grad_value

        # update to the transmittance of the prev step
        negative_log_transmittance_buffer = negative_log_transmittance_buffer - delta_value
        transmittance = tl.exp(-negative_log_transmittance_buffer)

        """
        transmittance_diff = tl.broadcast_to(
            tl.view(transmittance, (BLOCK_SIZE, 1)), (BLOCK_SIZE, OUT_C)
        ) - tl.broadcast_to(
            tl.view(prev_transmittance, (BLOCK_SIZE, 1)), (BLOCK_SIZE, OUT_C)
        )  # = rendering weights for the given step
        """
        transmittance_diff = tl.broadcast_to(tl.expand_dims(transmittance, 1), (BLOCK_SIZE, OUT_C)) - tl.broadcast_to(
            tl.expand_dims(prev_transmittance, 1), (BLOCK_SIZE, OUT_C)
        )  # = rendering weights for the given step

        # add the feature grad again
        d_color = grad_expected_features_buffer * transmittance_diff
        d_log_color = _d_color_activation(d_color, log_color)
        d_vecr, d_wc_, d_bc_ = _d_linear(d_log_color, wc, bc, vecr)

        if use_separate_color_rep:
            # the color branch does not enter here
            d_vec2_12 = tl.view(d_vec2_1, (BLOCK_SIZE, C))

        else:
            d_vec2_2, d_wr_, d_br_ = _d_linear_relu(d_vecr, wr, br, vecr, vec2)

            # no idea why but these reshapes are utterly necessary
            d_vec2_12 = tl.view(d_vec2_1, (BLOCK_SIZE, C)) + tl.view(d_vec2_2, (BLOCK_SIZE, C))

        # grad MLP
        d_vec1, d_w2_, d_b2_ = _d_linear_relu(d_vec2_12, w2, b2, vec2, vec1)
        d_vec, d_w1_, d_b1_ = _d_linear_relu(d_vec1, w1, b1, vec1, vec)

        if mask_out_of_bounds_samples:
            in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
            d_vec = d_vec * in_bounds_mask

        _splat_grid_rep(
            d_vec,
            grad_xy,
            grad_yz,
            grad_zx,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
            shape_representation,
        )

        if use_separate_color_rep:
            d_vec_color1, d_w2c_, d_b2c_ = _d_linear_relu(d_vecr, w2c, b2c, vecr, vec_color1)
            d_vec_color, d_wr_, d_br_ = _d_linear_relu(d_vec_color1, wr, br, vec_color1, vec_color)

            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE)
                d_vec_color = d_vec_color * in_bounds_mask

            d_rays_enc_ = tl.view(d_vec_color, (BLOCK_SIZE, C))

            _splat_grid_rep(
                d_vec_color,
                grad_xy_color,
                grad_yz_color,
                grad_zx_color,
                batch_index,
                sample_x,
                sample_y,
                sample_z,
                batch_size,
                C,
                D,
                H,
                W,
                BLOCK_SIZE,
                shape_representation,
            )

        else:
            d_vec_color = zero_vec
            d_w2c_ = zero_w
            d_b2c_ = zero_b
            d_rays_enc_ = d_vecr * (vecr > 0.0).to(tl.float32)

        d_wc += d_wc_
        d_bc += d_bc_
        d_wr += d_wr_
        d_br += d_br_
        d_w2 += d_w2_
        d_w1 += d_w1_
        d_b1 += d_b1_
        d_b2 += d_b2_
        d_wo += d_wo_
        d_bo += d_bo_
        d_w2c += d_w2c_
        d_b2c += d_b2c_
        d_rays_enc += d_rays_enc_

        prev_proj_depth = proj_depth
        prev_proj_features = proj_features
        sample_index_buffer = sample_index_buffer - 1

    # update the weight, bias grads
    tl.atomic_add(grad_weights + offs_CC, d_w1)
    tl.atomic_add(grad_weights + C * C + offs_CC, d_w2)
    tl.atomic_add(grad_weights + 2 * C * C + offs_CC, d_wr)
    tl.atomic_add(grad_weights + 3 * C * C + offs_CC, d_w2c)
    tl.atomic_add(grad_biases + tl.arange(0, C), d_b1)
    tl.atomic_add(grad_biases + C + tl.arange(0, C), d_b2)
    tl.atomic_add(grad_biases + 2 * C + tl.arange(0, C), d_br)
    tl.atomic_add(grad_biases + 3 * C + tl.arange(0, C), d_b2c)
    tl.atomic_add(grad_weight_opacity + tl.arange(0, C), d_wo)
    tl.atomic_add(grad_bias_opacity + tl.arange(0, 1), d_bo)
    tl.atomic_add(
        grad_weight_color + tl.arange(0, OUT_C)[:, None] * C + tl.arange(0, C)[None, :],
        d_wc,
    )
    tl.atomic_add(grad_bias_color + tl.arange(0, OUT_C), d_bc)
    tl.store(
        grad_rays_enc + rays_enc_offs,
        d_rays_enc,
        mask=rays_enc_mask,
    )
