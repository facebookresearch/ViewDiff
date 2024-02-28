# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, asdict
from typing import Union, Optional, Literal, Tuple
import os
from pathlib import Path
import json
import tyro
import copy
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.utils.checkpoint

from torch.utils.tensorboard import SummaryWriter

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler, DDPMScheduler, DDIMScheduler

from .model.custom_unet_2d_condition import (
    UNet2DConditionCrossFrameInExistingAttnModel,
)
from .model.util import (
    replace_self_attention_with_cross_frame_attention,
    add_pose_cond_to_attention_layers,
    update_cross_frame_attention_config,
    update_last_layer_mode,
    update_vol_rend_inject_noise_sigma,
    update_n_novel_images,
    CrossFrameAttentionConfig,
    ModelConfig,
)
from .model.custom_stable_diffusion_pipeline import CustomStableDiffusionPipeline

from .io_util import (
    setup_output_directories,
    make_output_directories,
    convert_to_tensorboard_dict,
    SaveConfig
)

from .metrics.image_metrics import load_lpips_vgg_model

from .scripts.misc.export_nerf_transforms import main as export_nerf_transforms
from .scripts.misc.export_nerf_transforms import save_smooth_video
from .scripts.misc.calculate_mean_image_stats import main as calculate_mean_image_stats
from .scripts.misc.create_masked_images import load_carvekit_bkgd_removal_model

from .data.co3d.co3d_dataset import CO3DConfig, CO3DDataset

from .train_util import FinetuneConfig
from diffusers.loaders import LoraLoaderMixin

from dacite import from_dict, Config

from .train import test_step

logger = get_logger(__name__, log_level="INFO")


@dataclass
class SlidingWindowConfig:
    is_active: bool = False
    """If sliding window generation is activated or not."""

    max_degrees: float = 360.0
    """How large is the rotation around hemisphere in a single batch."""

    degree_increment: float = 5.0
    """How much should the position of each image be rotated around hemisphere for each next batch."""

    not_alternating: bool = False
    """If False, change sign of degree_increment every step, i.e. first rotate right, then rotate left."""

    first_theta: float = 45.0
    """The theta value to use for first batch (==condition for all other images)."""

    first_radius: float = 4.5
    """The radius value to use for first batch (==condition for all other images)."""

    min_theta: float = 30.0
    """The minimum theta for all cameras."""

    max_theta: float = 100.0
    """The maximum theta for all cameras."""

    min_radius: float = 3.5
    """The minimum radius for all cameras."""

    max_radius: float = 4.5
    """The maximum radius for all cameras."""

    n_full_batches_to_save: int = 1
    """How many full batches should be the condition for next batches."""

    perc_add_images_to_save: float = 0.5
    """Percentage of how many images of the last (if --not_alternating=True) or last two (if --not_alternating=False) batches should be saved for next batches."""

    guidance_scale_from_second: float = 1.0
    """After first batch was generated, set the guidance-scale to this value for all subsequent generations."""

    repeat_first_n_steps: int = 1
    """Repeat this many steps in sliding window fashion (to build up complete condition)."""

    vol_rend_cache_per_frame_grids_percetange: float = 0.0
    """If >0, will save that many per-frame vol-rend grids the _previous_ batch and re-use it for the next batches (== sliding window generation). Default: 0.0"""

    input_condition_mode: Literal["dataset", "file", "none"] = "none"
    """Specifies if the first batch already has some condition.
        'dataset': sample a batch from the dataset and use as input.
        'none': do not provide condition for first batch.
    """

    input_condition_n_images: int = 0
    """How many images to use as condition. Only relevant for --input_condition_mode='dataset'."""

    create_smooth_video: bool = False
    """Whether the data should be converted to a smooth video at the end."""


@dataclass
class RunConfig:
    """Arguments for this run."""

    cross_frame_attention: CrossFrameAttentionConfig

    model: ModelConfig

    sliding_window: SlidingWindowConfig

    save: SaveConfig

    create_nerf_exports: bool = False
    """Whether the data should be exported to nerf-compatible format at the end."""

    pretrained_model_name_or_path: str = None
    """Path to pretrained model or model identifier from huggingface.co/models"""

    lpips_vgg_model_path: Optional[str] = None
    """Location to search for the pretrained lpips vgg model."""

    n_repeat_generation: int = 1
    """How often to repeat generation for each batch for image metrics (see Viewset-Diffusion: better to report average psnr/ssim scores)."""

    carvekit_checkpoint_dir: str = None
    """Path to the carvekit model used to remove backgrounds for metric calculations and nerf export. If set to None, will download the model."""

    revision: str = None
    """Revision of pretrained model identifier from huggingface.co/models."""

    output_dir: str = "output"
    """The output directory where the model predictions and checkpoints will be written."""

    experiment_name: Optional[str] = None
    """If this is set, will use this instead of the datetime string as identifier for the experiment."""

    device: str = "cuda"
    """The device for the pipeline"""

    batch_size: int = 1
    """The batch-size for the dataloader. Image-Grids will be saved with this batch-size."""

    guidance_scale: float = 7.5
    """Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
       `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
       Guidance scale is enabled by setting `guidance_scale > 1`.
       Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
       usually at the expense of lower image quality."""

    n_input_images: int = -1
    """Overwrite the n_input_images value stored in the pretrained model with this value."""

    allow_train_sequences: bool = False
    """If True, will also allow to select sequences from training distribution for testing. Default: False"""

    scheduler_type: Literal["ddpm", "dpm-multi-step", "unipc", "ddim"] = "ddpm"
    """Which scheduler to choose."""

    num_inference_steps: int = 50
    """How many steps to take with the scheduler."""

    vol_rend_deactivate_view_dependent_effects: bool = False
    """If True, will not use view-dependent effects in vol-rend layers."""

    max_steps: int = 30
    """How many steps of generation to do."""


def setup_run(
    dataset_config: CO3DConfig,
    run_config: RunConfig
):
    # create subfolder path
    setup_output_directories(
        io_config=run_config, model_config=run_config, dataset_config=dataset_config, is_train=False
    )

    # create directories
    make_output_directories(io_config=run_config)

    # Writer will output to <output_dir>/logs/ directory
    writer = SummaryWriter(log_dir=os.path.join(run_config.output_dir, "logs"))

    # write hparams for quick search in tb/file
    config = {**asdict(dataset_config), **asdict(run_config)}
    with open(os.path.join(run_config.output_dir, f"config.json"), "w") as f:
        json.dump(config, f, indent=4)
    writer.add_text("Hparams", json.dumps(convert_to_tensorboard_dict(config), indent=4), global_step=0)

    return writer


def set_sliding_window_config(
    config: SlidingWindowConfig,
    dataset: CO3DDataset,
    step: int,
    repeat_first_n_steps: int = 1,
):
    if step <= repeat_first_n_steps:
        # first n batches: generate at these fixed positions
        dataset.config.batch.replace_pose_with_spherical_theta = config.first_theta
        dataset.config.batch.replace_pose_with_spherical_radius = config.first_radius
        if not config.not_alternating:
            if step == 0:
                dataset.config.batch.replace_pose_with_spherical_start_phi = 0
                dataset.config.batch.replace_pose_with_spherical_end_phi = 360
            else:
                dataset.config.batch.replace_pose_with_spherical_start_phi = -config.max_degrees / 2.0
                dataset.config.batch.replace_pose_with_spherical_end_phi = config.max_degrees / 2.0
        else:
            dataset.config.batch.replace_pose_with_spherical_start_phi = 0
            dataset.config.batch.replace_pose_with_spherical_end_phi = config.max_degrees
    else:
        actual_step = step - repeat_first_n_steps

        # other batches: generate at random (theta, radius)
        dataset.config.batch.replace_pose_with_spherical_theta = np.random.uniform(
            low=config.min_theta,
            high=config.max_theta,
            size=1
        )[0]
        dataset.config.batch.replace_pose_with_spherical_radius = np.random.uniform(
            low=config.min_radius,
            high=config.max_radius,
            size=1
        )[0]

        # other batches: generate at different phi
        if not config.not_alternating:
            # rotate left for odd steps and rotate right for even steps
            residual = actual_step % 2
            delta_degree = (actual_step // 2 - (residual == 0)) * config.degree_increment + config.max_degrees / 2.0
            delta_degree = delta_degree % 360  # keep the increment in 360 degrees
            dataset.config.batch.replace_pose_with_spherical_start_phi = delta_degree
            dataset.config.batch.replace_pose_with_spherical_end_phi = config.degree_increment + delta_degree
            if residual:
                dataset.config.batch.replace_pose_with_spherical_start_phi *= -1
                dataset.config.batch.replace_pose_with_spherical_end_phi *= -1
        else:
            delta_degree = actual_step * config.degree_increment  # how many increments to take
            delta_degree = delta_degree % 360  # keep the increment in 360 degrees
            dataset.config.batch.replace_pose_with_spherical_start_phi = delta_degree
            dataset.config.batch.replace_pose_with_spherical_end_phi = config.max_degrees + delta_degree


def combine_batches(known_batch, new_batch):
    def duplicate_known(key: str):
        N = known_batch[key].shape[1] + new_batch[key].shape[1]
        x = known_batch[key][:, 0:1]
        x = x.repeat(1, N, *[1] * len(new_batch[key].shape[2:]))
        return x

    batch = {
        "prompt": known_batch["prompt"],  # override, can be different, but should be same as old # [N,]
        "images": torch.cat([known_batch["images"], new_batch["images"]], dim=1),  # first the known, then new
        "pose": torch.cat([known_batch["pose"], new_batch["pose"]], dim=1),  # first the known, then new # (N, n_known_images + n_input_images, 4, 4)
        "intensity_stats": duplicate_known("intensity_stats"),
        "K": duplicate_known("K"),
        "known_images": known_batch["known_images"] if "known_images" not in new_batch else torch.cat([known_batch["known_images"], new_batch["known_images"]], dim=1),
        "bbox": new_batch["bbox"],  # should be the same as in known_batch
        "root": new_batch["root"],  # should be the same as in known_batch
        "file_names": [*known_batch["file_names"], *new_batch["file_names"]]  # first the known, then new
    }

    if "foreground_prob" in known_batch and "foreground_prob" in new_batch:
        batch["foreground_prob"] = torch.cat([known_batch["foreground_prob"], new_batch["foreground_prob"]], dim=1)  # first the known, then new

    return batch


def subsample_batch(x, indices, output=None, keep_known_images: bool = False, mark_filename_as_cond: bool = True):
    batch = {
        "prompt": x["prompt"],
        "images": x["images"][:, indices],
        "pose": x["pose"][:, indices],
        "intensity_stats": x["intensity_stats"][:, indices],
        "K": x["K"][:, indices],
        "bbox": x["bbox"],
        "file_names": [tuple(f"cond_{xi}" if mark_filename_as_cond and "cond" not in xi else xi for xi in x) for x in [x["file_names"][i] for i in indices]],
        "root": x["root"],
    }

    if "foreground_prob" in x:
        batch["foreground_prob"] = x["foreground_prob"][:, indices]

    if "known_images" in x or output is not None:
        batch["known_images"] = x["known_images"][:, indices] if keep_known_images else output.images[:, indices].cpu() * 2 - 1

    return batch


def collate_batch(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        elif k == "root" or k == "prompt":
            batch[k] = [v]
        elif k == "file_names":
            batch[k] = [(vi,) for vi in v]
        else:
            raise ValueError(k, v)

    return batch


def select_as_known(batch, n_images: int = 1):
    known_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            known_batch[k] = v[:, 0:n_images].clone()
            if k == "images":
                known_batch["known_images"] = v[:, 0:n_images].clone()
        elif k == "root" or k == "prompt":
            known_batch[k] = v
        elif k == "file_names":
            known_batch[k] = [tuple(f"cond_{xi}" for xi in x) for x in v[0:n_images]]
        else:
            raise ValueError(k, v)

    return known_batch


def update_cfa_config(
    run_config: RunConfig,
    pipeline: CustomStableDiffusionPipeline
):
    if run_config.cross_frame_attention.mode == "add_in_existing_block":
        update_cross_frame_attention_config(
            pipeline.unet,
            run_config.n_input_images,
            run_config.cross_frame_attention.to_k_other_frames,
            run_config.cross_frame_attention.with_self_attention,
            run_config.cross_frame_attention.random_others,
            change_self_attention_layers=False,  # should have custom cfa layers
        )
    elif run_config.cross_frame_attention.mode == "pretrained":
        update_cross_frame_attention_config(
            pipeline.unet,
            run_config.n_input_images,
            run_config.cross_frame_attention.to_k_other_frames,
            run_config.cross_frame_attention.with_self_attention,
            run_config.cross_frame_attention.random_others,
            change_self_attention_layers=True,  # should have cfa is sa layers
        )
    else:
        raise NotImplementedError(
            f"did not implement different n_input_images for cfa.mode={run_config.cross_frame_attention.mode}"
        )


def process_batch(
    run_config: RunConfig,
    dataset_config: CO3DConfig,
    pipeline: CustomStableDiffusionPipeline,
    step: int,
    batch,
    known_batch=None,
    previous_cached_vol_rend_grids=None,
    writer=None,
    lpips_vgg_model=None,
    carvekit_model=None,
):
    if run_config.sliding_window.is_active:
        # use identical generator for different batches --> start/add the same random noise all the time
        generator = torch.Generator(device=run_config.device).manual_seed(dataset_config.seed)
    else:
        # use different generator for different batches --> start/add different random noise to get variety of results
        generator = torch.Generator(device=run_config.device).manual_seed(dataset_config.seed + step)

    # combine
    if known_batch is not None:
        batch = combine_batches(known_batch, batch)

    # check if need to change n_input_images
    if run_config.n_input_images != batch["pose"].shape[1]:
        run_config.n_input_images = batch["pose"].shape[1]
        run_config.cross_frame_attention.to_k_other_frames = batch["pose"].shape[1] - 1
        run_config.model.n_input_images = batch["pose"].shape[1]
        update_cfa_config(run_config, pipeline)

    # alwasy set to 0
    batch["intensity_stats"] *= 0

    # create images
    output = test_step(
        pipeline=pipeline,
        batch=batch,
        model_config=run_config.model,
        cfa_config=run_config.cross_frame_attention,
        io_config=run_config,
        orig_hw=(dataset_config.batch.image_height, dataset_config.batch.image_width),
        guidance_scale=run_config.guidance_scale,
        generator=generator,
        global_step=step,
        writer=writer,
        deactivate_view_dependent_rendering=run_config.vol_rend_deactivate_view_dependent_effects,
        num_inference_steps=run_config.num_inference_steps,
        n_repeat_generation=run_config.n_repeat_generation,
        lpips_vgg_model=lpips_vgg_model,
        carvekit_model=carvekit_model,
    )

    # save generated images as condition for next batch
    if run_config.sliding_window.is_active:
        assert run_config.batch_size == 1, "sliding window currently only supported for batch-size 1"
        # TODO need to handle padding in collapse for bs>1

        # how many images to save for this batch
        n_save_additionally = int(dataset_config.batch.n_parallel_images * run_config.sliding_window.perc_add_images_to_save)
        n_full_batches_to_save = run_config.sliding_window.n_full_batches_to_save
        n_add_batches = 2 if (not run_config.sliding_window.not_alternating) and run_config.sliding_window.input_condition_mode == "none" else 1

        if step < n_full_batches_to_save:
            # assign all generated images to known_batch
            save_indices = np.array([step * dataset_config.batch.n_parallel_images + i for i in range(dataset_config.batch.n_parallel_images)])
            save_indices += run_config.sliding_window.input_condition_n_images
            add_known_batch = subsample_batch(batch, save_indices, output)
            known_batch = add_known_batch if known_batch is None else combine_batches(known_batch, add_known_batch)
        elif n_save_additionally > 0:
            # check if we should remove images from known_batch
            n_images_in_known_batch = known_batch["pose"].shape[1] if known_batch is not None else 0
            max_images_with_full_batches = n_full_batches_to_save * dataset_config.batch.n_parallel_images + run_config.sliding_window.input_condition_n_images
            max_images_in_known_batch = max_images_with_full_batches + n_add_batches * n_save_additionally
            if n_images_in_known_batch >= max_images_in_known_batch:
                remove_indices = np.array([i for i in range(max_images_with_full_batches, max_images_with_full_batches + n_save_additionally)])
                keep_indices = np.array([i for i in range(max_images_in_known_batch) if i not in remove_indices and i < n_images_in_known_batch])
                known_batch = subsample_batch(known_batch, keep_indices, None, keep_known_images=True) if len(keep_indices) > 0 else None

            # add new images to known_batch
            x = n_images_in_known_batch + dataset_config.batch.n_parallel_images
            save_indices = np.array([i for i in range(x - n_save_additionally, x)])
            add_known_batch = subsample_batch(batch, save_indices, output)
            known_batch = combine_batches(known_batch, add_known_batch) if known_batch is not None else add_known_batch

    if run_config.sliding_window.vol_rend_cache_per_frame_grids_percetange > 0:
        previous_cached_vol_rend_grids = output.cached_vol_rend_grids

    return known_batch, previous_cached_vol_rend_grids


def test(
    dataset_config: CO3DConfig,
    run_config: RunConfig,
):
    # give it an experiment name.
    # the pretrained_model_name_or_path is always ".../XXXX/model" or "..../XXXX/model/"
    # we are interested to set the experiment name to XXXX
    run_config.experiment_name = Path(run_config.pretrained_model_name_or_path).parent.stem

    # load config from checkpoint_path
    config_path = os.path.join(run_config.pretrained_model_name_or_path, "config.json")
    if not os.path.isfile(str(config_path)):
        raise ValueError("cannot find config.json in ", config_path)
    with open(config_path, "r") as f:
        config_data = json.load(f)
    finetune_config = from_dict(FinetuneConfig, data=config_data, config=Config(cast=[tuple, int]))

    # copy from saved config
    run_config.cross_frame_attention = finetune_config.cross_frame_attention
    run_config.model = finetune_config.model

    # overwrite n_input_images
    need_update_cfa_config = False
    if run_config.n_input_images > -1:
        need_update_cfa_config = run_config.model.n_input_images != run_config.n_input_images
        run_config.model.n_input_images = run_config.n_input_images
        run_config.model.n_output_noise = run_config.n_input_images
        run_config.cross_frame_attention.to_k_other_frames = run_config.n_input_images - 1

    # manually update required fields
    run_config.n_input_images = run_config.model.n_input_images
    run_config.n_output_noise = run_config.model.n_output_noise
    dataset_config.batch.n_parallel_images = run_config.model.n_input_images
    if not run_config.allow_train_sequences:
        dataset_config.dataset_args.exclude_sequence = tuple(config_data["selected_sequences"]["train"])

    # parse selected_sequences: if greater than one entry is specified, it is assumed to be multiple sequences (i.e., used as expected)
    # else we make some more checks (see below) - while a bit confusing, this is a simple way to hack the datasets with less command line arguments to be specified ... :)
    if len(dataset_config.dataset_args.pick_sequence) == 1:
        if dataset_config.dataset_args.pick_sequence[0] == "":
            # if it is not specified at all, we default to picking one random sequence
            dataset_config.dataset_args.pick_sequence = ()
            dataset_config.max_sequences = 1
        elif dataset_config.dataset_args.pick_sequence[0].isdigit():
            # if it is a number, we assume we really wanted to specify max_sequences
            dataset_config.max_sequences = int(dataset_config.dataset_args.pick_sequence[0])
            dataset_config.dataset_args.pick_sequence = ()
    n_sequences = dataset_config.max_sequences

    # setup run
    orig_run_config = copy.deepcopy(run_config)
    writer = setup_run(dataset_config, run_config)
    set_seed(dataset_config.seed)

    # Load model.
    if run_config.pretrained_model_name_or_path[-1] != "/":
        # need trailing slash in checkpoint path
        run_config.pretrained_model_name_or_path += "/"
    pipeline = CustomStableDiffusionPipeline.from_pretrained(
        run_config.pretrained_model_name_or_path, revision=run_config.revision
    )
    pipeline.set_progress_bar_config(**{
        "leave": True,
        "desc": "Run Denoising Inference"
    })

    # set scheduler for inference
    if run_config.scheduler_type == "dpm-multi-step":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif run_config.scheduler_type == "unipc":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    elif run_config.scheduler_type == "ddpm":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    elif run_config.scheduler_type == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError('unsupported scheduler_type', run_config.scheduler_type)

    # update model architecture
    pipeline.scheduler.config.prediction_type = finetune_config.training.noise_prediction_type
    if run_config.cross_frame_attention.mode != "none":
        if run_config.cross_frame_attention.mode == "pretrained":
            if not isinstance(pipeline.unet, UNet2DConditionCrossFrameInExistingAttnModel):
                raise ValueError(
                    f"set cross_frame_attention.mode to {run_config.cross_frame_attention.mode}, but the model path loaded unet of type",
                    type(pipeline.unet),
                )

            # need to update the unet, unfortunately this info does not get saved -- but no new weights are missed, it just changes the implementation behaviour
            replace_self_attention_with_cross_frame_attention(
                unet=pipeline.unet,
                n_input_images=run_config.n_input_images,
                to_k_other_frames=run_config.cross_frame_attention.to_k_other_frames,
                with_self_attention=run_config.cross_frame_attention.with_self_attention,
                random_others=run_config.cross_frame_attention.random_others,
                use_lora_in_cfa="cfa" in run_config.model.pose_cond_mode or "sa" in run_config.model.pose_cond_mode,
                use_temb_in_lora=run_config.cross_frame_attention.use_temb_cond,
                temb_out_size=8,
                pose_cond_dim=run_config.model.pose_cond_dim,
                rank=run_config.model.pose_cond_lora_rank,
            )

        elif run_config.cross_frame_attention.mode == "add_in_existing_block":
            if not isinstance(pipeline.unet, UNet2DConditionCrossFrameInExistingAttnModel):
                raise ValueError(
                    f"set cross_frame_attention.mode to {run_config.cross_frame_attention.mode}, but the model path loaded unet of type",
                    type(pipeline.unet),
                )

    # check if we should update last_layer_mode
    if finetune_config.training.changed_cfa_last_layer != run_config.cross_frame_attention.last_layer_mode:
        print("Change last-layer-mode to", finetune_config.training.changed_cfa_last_layer)
        update_last_layer_mode(
            pipeline.unet,
            finetune_config.training.changed_cfa_last_layer,
        )

    # disable vol-rend noise
    update_vol_rend_inject_noise_sigma(
        pipeline.unet, 0.0
    )
    # disable n_novel_images
    update_n_novel_images(
        pipeline.unet, 0
    )

    # set to correct n_images
    if need_update_cfa_config:
        update_cfa_config(run_config, pipeline)

    # load lora pose-cond weights
    if run_config.model.pose_cond_mode != "none":
        # Set correct lora layers
        unet_lora_attn_procs, unet_lora_parameters = add_pose_cond_to_attention_layers(
            pipeline.unet,
            rank=run_config.model.pose_cond_lora_rank,
            pose_cond_dim=run_config.model.pose_cond_dim,
            only_cross_attention="sa" not in run_config.model.pose_cond_mode,
        )

        if unet_lora_parameters is not None:
            in_dir = os.path.join(run_config.pretrained_model_name_or_path, "unet")
            try:
                lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.safetensors")
            except:
                lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.bin")
            lora_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
            pipeline.unet.load_state_dict(lora_state_dict, strict=False)
            print("Loaded LoRA weights into model")

    # now the model is complete --> put to required device
    pipeline = pipeline.to(run_config.device)

    # load lpips vgg model for image metrics
    # load carvekit model for image metrics
    lpips_vgg_model = None
    carvekit_model = None
    if run_config.sliding_window.input_condition_mode != "none":
        lpips_vgg_model = load_lpips_vgg_model(run_config.lpips_vgg_model_path)
        lpips_vgg_model = lpips_vgg_model.to(run_config.device)

        carvekit_model = load_carvekit_bkgd_removal_model(run_config.carvekit_checkpoint_dir)

    #############################
    # run inference on test set #
    #############################
    if run_config.sliding_window.is_active:
        # save original config for restarting of next sequence
        orig_run_config_model = copy.deepcopy(run_config.model)
        orig_run_config_cfa = copy.deepcopy(run_config.cross_frame_attention)
        orig_n_input_images = run_config.n_input_images
        orig_model_n_input_images = run_config.model.n_input_images

        # load test set
        assert isinstance(dataset_config, CO3DConfig)
        dataset = CO3DDataset(dataset_config)
        n_total_sequences = len(dataset.get_all_sequences())
        seq_counter = 0

        # generate for each sequence separately
        for category_offset, category in dataset.offset_to_category:
            offset_to_sequence = dataset.category_dict[category]["offset_to_sequence"]
            sequence_lengths = dataset.category_dict[category]["sequence_lengths"]
            for (sequence_offset, sequence), sequence_length in zip(offset_to_sequence, sequence_lengths):
                start_idx = category_offset + sequence_offset
                known_batch = None
                previous_cached_vol_rend_grids = None

                # loop for sequence -- generate all the images
                for step in tqdm(range(run_config.max_steps + run_config.sliding_window.repeat_first_n_steps + 1), desc="Generate Image Batch"):
                    # update step for dataset lookup
                    actual_step = step
                    if run_config.sliding_window.input_condition_mode == "dataset":
                        if step <= run_config.sliding_window.repeat_first_n_steps:
                            # sample first batch from dataset as long as we want to repeat first step
                            actual_step = 0
                        else:
                            # sample the next set of images by going forward n_parallel_images at once
                            # this way, we sample consecutive frames, e.g. if n_parallel_images=5:
                            #       first batch at idx=0 contains frames 0-4
                            #       second batch at idx=5 contains frames 5-9
                            #       third batch at idx=10 contains frames 10-14
                            actual_step = step - run_config.sliding_window.repeat_first_n_steps
                            actual_step = actual_step * dataset_config.batch.n_parallel_images

                    # update dataset sampling mode
                    if run_config.sliding_window.input_condition_mode == "none":
                        set_sliding_window_config(run_config.sliding_window, dataset, actual_step, repeat_first_n_steps=run_config.sliding_window.repeat_first_n_steps)
                    elif run_config.sliding_window.input_condition_mode == "dataset" and step == 0:
                        # for first batch: set known_batch (for all next batches it is already defined)
                        batch = collate_batch(dataset[start_idx])
                        known_batch = select_as_known(batch, n_images=run_config.sliding_window.input_condition_n_images)

                        # for first batch: other images are spread out equally (similar to DFM)
                        dataset.config.batch.sequence_offset = 40
                        actual_step = 40

                    # get next batch
                    data_index = start_idx + actual_step
                    if data_index >= sequence_length:
                        print("Cancel generation early because data_index is out-of-bounds", data_index, sequence_length)
                        break
                    batch = collate_batch(dataset[data_index])

                    # for subsequent batches: other images are sampled sequentially
                    if run_config.sliding_window.input_condition_mode == "dataset" and step == 0:
                        dataset.config.batch.sequence_offset = 1

                    # process next batch
                    known_batch, previous_cached_vol_rend_grids = process_batch(
                        run_config=run_config,
                        dataset_config=dataset_config,
                        pipeline=pipeline,
                        step=step,
                        batch=batch,
                        known_batch=known_batch,
                        previous_cached_vol_rend_grids=previous_cached_vol_rend_grids,
                        writer=writer,
                        lpips_vgg_model=lpips_vgg_model,
                        carvekit_model=carvekit_model
                    )
                    run_config.guidance_scale = run_config.sliding_window.guidance_scale_from_second

                if run_config.create_nerf_exports:
                    # foreground segment images + export to (instant-ngp, nerfstudio, sdfstudio)-compatible formats
                    export_nerf_transforms(
                        input_path=os.path.join(run_config.output_dir, "images"),
                        carvekit_checkpoint_dir=run_config.carvekit_checkpoint_dir,
                        combine_all=True,
                        create_smooth_video=run_config.sliding_window.create_smooth_video,
                        smooth_video_framerate=15,
                        smooth_video_sort_type="interleaving" if run_config.sliding_window.not_alternating else "alternating",
                        n_images_per_batch=dataset_config.batch.n_parallel_images,
                        skip_first_n_steps=run_config.sliding_window.repeat_first_n_steps,
                    )
                elif run_config.sliding_window.create_smooth_video:
                    save_smooth_video(
                        image_folder=os.path.join(run_config.output_dir, "images"),
                        n_images_per_batch=dataset_config.batch.n_parallel_images,
                        framerate=15,
                        sort_type="interleaving" if run_config.sliding_window.not_alternating else "alternating",
                        skip_first_n_steps=run_config.sliding_window.repeat_first_n_steps,
                    )

                print("finished generation for sequence", sequence, "at directory", run_config.output_dir)

                # prepare for next scene
                seq_counter += 1
                if seq_counter < n_total_sequences:
                    # restore original config
                    run_config = copy.deepcopy(orig_run_config)
                    run_config.model = copy.deepcopy(orig_run_config_model)
                    run_config.cross_frame_attention = copy.deepcopy(orig_run_config_cfa)
                    run_config.n_input_images = orig_n_input_images
                    run_config.model.n_input_images = orig_model_n_input_images

                    # restore original model config
                    update_cfa_config(run_config, pipeline)

                    # create new output folder / writer / ...
                    writer = setup_run(dataset_config, run_config)
    else:
        dataset = CO3DDataset(dataset_config)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=n_sequences != 1,
            batch_size=run_config.batch_size,
            num_workers=0,
        )

        for step, batch in enumerate(dataloader):
            if step >= run_config.max_steps:
                break

            if run_config.sliding_window.input_condition_mode == "dataset":
                # set the first few images as condition and remove from the actual batch
                known_batch = select_as_known(batch, n_images=run_config.sliding_window.input_condition_n_images)
                keep_indices = np.array([x for x in range(run_config.sliding_window.input_condition_n_images, dataset_config.batch.n_parallel_images)])
                batch = subsample_batch(batch, keep_indices, mark_filename_as_cond=False)
                batch = combine_batches(known_batch, batch)

            process_batch(
                run_config=run_config,
                dataset_config=dataset_config,
                pipeline=pipeline,
                step=step,
                batch=batch,
                known_batch=None,
                previous_cached_vol_rend_grids=None,
                writer=writer,
                lpips_vgg_model=lpips_vgg_model,
                carvekit_model=carvekit_model
            )

        # calc accumulated image statistics
        if run_config.sliding_window.input_condition_mode == "dataset":
            calculate_mean_image_stats(
                input_path=run_config.stats_out_dir
            )

        if run_config.create_nerf_exports:
            # foreground segment images + export to (instant-ngp, nerfstudio, sdfstudio)-compatible formats
            export_nerf_transforms(
                input_path=os.path.join(run_config.output_dir, "images"),
                carvekit_checkpoint_dir=run_config.carvekit_checkpoint_dir,
                combine_all=False
            )

        print("finished generation at directory", run_config.output_dir)


if __name__ == "__main__":
    tyro.cli(test)
