# Copyright (c) Meta Platforms, Inc. and affiliates.

import tyro
from .co3d_dataset import CO3DConfig, CO3DDataset


def save_recentered(
    dataset_config: CO3DConfig,
    recompute: bool = False,
):
    # make sure the important fields are set correctly
    dataset_config.dataset_args.load_point_clouds = True
    dataset_config.batch.load_recentered = False
    dataset_config.batch.need_mask_augmentations = False

    # Get the dataset: parse CO3Dv2
    dataset = CO3DDataset(dataset_config)

    # save recentered data
    dataset.recenter_sequences(recompute=recompute)


if __name__ == "__main__":
    tyro.cli(save_recentered)
