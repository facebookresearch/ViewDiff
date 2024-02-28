# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import lpips
import skimage


def load_lpips_vgg_model(lpips_vgg_model_path: str):
    return lpips.LPIPS(net='vgg', model_path=lpips_vgg_model_path)


def calc_psnr_ssim_lpips(src: torch.Tensor, target: torch.Tensor, lpips_vgg_model: lpips.LPIPS):
    """

    :param src: (n_batches, n_images_per_batch, C, H, W)
    :param target: (n_batches, n_images_per_batch, C, H, W)
    :param lpips_vgg_model:
    :return:
    """
    l2_criterion = torch.nn.MSELoss(reduction='none')

    psnrs = []
    lpipses = []
    ssims = []
    for batch_idx in range(src.shape[0]):
        # ================ PSNR measurement ================
        # don't average across frames
        l2_loss = l2_criterion(src[batch_idx], target[batch_idx]).mean(dim=[1, 2, 3])
        psnr = -10 * torch.log10(l2_loss)
        psnrs.extend(list(x.item() for x in psnr))

        # ================ LPIPS measurement ================
        lpips = lpips_vgg_model(src[batch_idx], target[batch_idx], normalize=True)
        lpipses.extend(list(x.item() for x in lpips))

        # ================ SSIM measurement =============
        for view_idx in range(src.shape[1]):
            ssim = skimage.metrics.structural_similarity(
                src[batch_idx, view_idx].cpu().numpy(),
                target[batch_idx, view_idx].cpu().numpy(),
                data_range=1,
                channel_axis=0
            )
            ssims.append(float(ssim))

    return psnrs, lpipses, ssims
