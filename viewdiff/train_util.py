# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import yaml
import time
import math
import json
import shutil
import logging
from dataclasses import dataclass, asdict
from typing import Literal, List, Optional, Union, Dict
from packaging import version

import torch
import torch.utils.checkpoint

from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.operations import recursively_apply
from accelerate.tracking import GeneralTracker, on_main_process, logger

from .io_util import convert_to_tensorboard_dict, IOConfig

from .data.co3d.co3d_dataset import CO3DConfig, CO3DDataset, CO3DDreamboothDataset


from .model.util import ModelConfig, CrossFrameAttentionConfig
from .model.custom_unet_2d_condition import (
    UNet2DConditionCrossFrameInExistingAttnModel
)
from diffusers.loaders import LoraLoaderMixin


@dataclass
class TrainingConfig:
    """Arguments for training."""

    validation_epochs: int = 1
    """Run fine-tuning validation every X epochs."""

    train_batch_size: int = 1
    """Batch size (per device) for the training dataloader."""

    num_train_epochs: int = 500
    """epochs"""

    max_train_steps: Optional[int] = None
    """Total number of training steps to perform.  If provided, overrides num_train_epochs."""

    dataloader_num_workers: int = 4
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."""

    local_rank: int = -1
    """For distributed training: local_rank"""

    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = None
    """Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.
       and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the
        flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."""

    noise_prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon"
    """How to calculate the diffusion denoising MSE loss. The target is the applied noise ('epsilon'), the noise-free sample ('sample'), or the velocity ('v_prediction')."""

    remove_cfa_skip_connections_at_iter: int = -1
    """If >-1, it will change the last_layer_mode in all cross-frame-attention transformer-blocks to --changed_cfa_last_layer after that many training iterations."""

    changed_cfa_last_layer: Literal["none", "no_residual_connection"] = "no_residual_connection"
    """Change the last_layer_mode in all cross-frame-attention transformer-blocks to this value."""

    dreambooth_prior_preservation_loss_weight: float = 0.0
    """If >0, will use prior preservation loss during training similar to Dreambooth (see https://dreambooth.github.io/)."""

    dreambooth_prior_preservation_every_nth: int = 5
    """Calculates the prior preservation loss every nth step."""

    prob_images_not_noisy: float = 0.25
    """With this probability, some of the images in a batch will be not noisy (e.g. input image conditioning)."""

    max_num_images_not_noisy: int = 2
    """If some of the images in a batch should not be noisy: this defines the maximum number of images to which this applies."""


@dataclass
class OptimizerConfig:
    """Arguments for optimizer"""

    learning_rate: float = 5e-5
    """Initial learning rate (after the potential warmup period) to use."""

    vol_rend_learning_rate: float = 1e-3
    """Initial learning rate (after the potential warmup period) to use for the volume-rendering components in the model."""

    vol_rend_adam_weight_decay: float = 0.0
    """Weight decay to use for the volume-rendering components in the model."""

    scale_lr: bool = False
    """Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."""

    lr_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "constant"
    """The scheduler type to use."""

    lr_warmup_steps: int = 500
    """Number of steps for the warmup in the lr scheduler."""

    use_8bit_adam: bool = False
    """Whether or not to use 8-bit Adam from bitsandbytes."""

    allow_tf32: bool = False
    """Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
        https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"""

    adam_beta1: float = 0.9
    """The beta1 parameter for the Adam optimizer."""

    adam_beta2: float = 0.999
    """The beta2 parameter for the Adam optimizer."""

    adam_weight_decay: float = 1e-2
    """Weight decay to use."""

    adam_epsilon: float = 1e-08
    """Epsilon value for the Adam optimizer"""

    max_grad_norm: float = 0.1
    """Max gradient norm."""

    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""

    only_train_new_layers: bool = False
    """If set, will only optimize over new layers (e.g. additional cross-frame attention layers)."""


@dataclass
class FinetuneConfig:
    training: TrainingConfig
    optimizer: OptimizerConfig
    model: ModelConfig
    cross_frame_attention: CrossFrameAttentionConfig
    io: IOConfig


class CustomTensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "custom_tensorboard"
    requires_logging_directory = True

    @staticmethod
    def listify(data):
        """
        Recursively finds tensors in a nested list/tuple/dictionary and converts them to a list of numbers.

        Args:
            data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to convert to regular numbers.

        Returns:
            The same data structure as `data` with lists of numbers instead of `torch.Tensor`.
        """

        def _convert_to_list(tensor):
            tensor = tensor.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
                # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
                # Until Numpy adds bfloat16, we must convert float32.
                tensor = tensor.to(torch.float32)
            return tensor.tolist()

        return recursively_apply(_convert_to_list, data)

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir)
        logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        project_run_name = time.time()
        dir_name = os.path.join(self.logging_dir, str(project_run_name))
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(values, outfile)
            except yaml.representer.RepresenterError:
                logger.error("Serialization to store hyperparameters failed")
                raise
        logger.debug("Stored initial configuration hyperparameters to TensorBoard and hparams yaml file")

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        values = CustomTensorBoardTracker.listify(values)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()
        logger.debug("Successfully logged to TensorBoard")

    @on_main_process
    def log_images(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `SummaryWriter.add_image` method.
        """
        for k, v in values.items():
            self.writer.add_images(k, v, global_step=step, **kwargs)
        logger.debug("Successfully logged images to TensorBoard")

    @on_main_process
    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.debug("TensorBoard writer closed")


def check_local_rank(training_config: TrainingConfig):
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != training_config.local_rank:
        training_config.local_rank = env_local_rank

    return training_config


def setup_accelerate(finetune_config: FinetuneConfig, dataset_config: CO3DConfig, logger):
    logging_dir = os.path.join(finetune_config.io.output_dir, finetune_config.io.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=finetune_config.io.output_dir,
        logging_dir=logging_dir,
    )
    if finetune_config.io.report_to == "custom_tensorboard":
        log_with = CustomTensorBoardTracker(run_name=f"view-diff", logging_dir=logging_dir)
    else:
        log_with = finetune_config.io.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=finetune_config.optimizer.gradient_accumulation_steps,
        mixed_precision=finetune_config.training.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # set the training seed now.
    generator = torch.Generator(device=accelerator.device).manual_seed(dataset_config.seed)
    set_seed(dataset_config.seed)

    return accelerator, generator


def load_models(pretrained_model_name_or_path: str, revision: str = None):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def unet_attn_processors_state_dict(
    unet: Union[UNet2DConditionModel, UNet2DConditionCrossFrameInExistingAttnModel]
) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        if hasattr(attn_processor, "state_dict"):
            for parameter_key, parameter in attn_processor.state_dict().items():
                attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def setup_model_and_optimizer(
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    unet: Union[
        UNet2DConditionCrossFrameInExistingAttnModel,
    ],
    accelerator: Accelerator,
    finetune_config: FinetuneConfig,
    logger,
    unet_lora_parameters: List[torch.nn.Parameter] = None,
):
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # parse orig_unet
    model_cls = type(unet)

    # Create EMA for the unet.
    ema_unet = None
    if finetune_config.model.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=model_cls, model_config=unet.config)

    if finetune_config.model.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        unet_lora_layers_to_save = None
        ema_unet_lora_layers_to_save = None

        for model in models:
            if isinstance(model, model_cls):
                out_dir = os.path.join(output_dir, "unet")
                model.save_pretrained(out_dir)
                if unet_lora_parameters is not None:
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                    LoraLoaderMixin.save_lora_weights(out_dir, unet_lora_layers=unet_lora_layers_to_save)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        if finetune_config.model.use_ema:
            out_dir = os.path.join(output_dir, "unet_ema")
            ema_unet.save_pretrained(out_dir)
            if unet_lora_parameters is not None:
                ema_unet_lora_layers_to_save = unet_attn_processors_state_dict(ema_unet)
                LoraLoaderMixin.save_lora_weights(out_dir, unet_lora_layers=ema_unet_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        if finetune_config.model.use_ema:
            in_dir = os.path.join(input_dir, "unet_ema")
            load_model = EMAModel.from_pretrained(in_dir, model_cls)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

            if unet_lora_parameters is not None:
                try:
                    lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.safetensors")
                except:
                    lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.bin")
                lora_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
                ema_unet.load_state_dict(lora_state_dict, strict=False)

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(model, model_cls):
                in_dir = os.path.join(input_dir, "unet")
                load_model = model_cls.from_pretrained(in_dir)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict(), strict=unet_lora_parameters is None)
                del load_model

                if unet_lora_parameters is not None:
                    try:
                        lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.safetensors")
                    except:
                        lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.bin")
                    lora_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
                    model.load_state_dict(lora_state_dict, strict=False)
                    logger.info("Loaded lora parameters into model")
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if finetune_config.model.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if finetune_config.optimizer.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if finetune_config.optimizer.scale_lr:
        finetune_config.optimizer.learning_rate = (
            finetune_config.optimizer.learning_rate
            * finetune_config.optimizer.gradient_accumulation_steps
            * finetune_config.training.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if finetune_config.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # add model parameters
    params = []
    if finetune_config.optimizer.only_train_new_layers:
        if (
            finetune_config.cross_frame_attention.mode == "add_in_existing_block"
            or finetune_config.cross_frame_attention.mode == "pretrained"
        ):
            # add cfa params (w/o vol-rend)
            cross_frame_params_without_vol_rend = unet.get_cross_frame_params(vol_rend_mode="without")
            if len(cross_frame_params_without_vol_rend) > 0:
                # there are cases where these params could be empty, e.g. when we use forward_model and not use cfa in unet or when we use "pretrained"
                params.append(
                    {"params": cross_frame_params_without_vol_rend, "lr": finetune_config.optimizer.learning_rate}
                )

            # add cfa vol-rend params
            cross_frame_vol_rend_params = unet.get_cross_frame_params(vol_rend_mode="only")
            if len(cross_frame_vol_rend_params) > 0:
                # there are cases where these params could be empty, e.g. when we do not use vol-rend layer or not use cfa in unet
                params.append(
                    {
                        "params": cross_frame_vol_rend_params,
                        "lr": finetune_config.optimizer.vol_rend_learning_rate,
                        "weight_decay": finetune_config.optimizer.vol_rend_adam_weight_decay,
                    }
                )

            # add lora pose-cond params
            if unet_lora_parameters is not None:
                params.append({"params": unet_lora_parameters, "lr": finetune_config.optimizer.learning_rate})

    else:
        if (
            finetune_config.cross_frame_attention.mode == "add_in_existing_block"
            or finetune_config.cross_frame_attention.mode == "pretrained"
        ):
            # add non-vol-rend params (including the existing unet params)
            params_without_vol_rend = unet.get_params_without_volume_rendering()
            params.append(
                {
                    "params": params_without_vol_rend,
                    "lr": finetune_config.optimizer.learning_rate,
                    "weight_decay": finetune_config.optimizer.adam_weight_decay,
                }
            )

            # add vol-rend params (both for cfa layers and forward-model)
            vol_rend_params = unet.get_cross_frame_params(vol_rend_mode="only")
            params.append(
                {
                    "params": vol_rend_params,
                    "lr": finetune_config.optimizer.vol_rend_learning_rate,
                    "weight_decay": finetune_config.optimizer.vol_rend_adam_weight_decay,
                }
            )
        else:
            # can add all parameters, since currently only add_in_existing_block and pretrained does support the vol-rend layer.
            params = [
                {
                    "params": unet.parameters(),
                    "lr": finetune_config.optimizer.learning_rate,
                    "weight_decay": finetune_config.optimizer.adam_weight_decay,
                }
            ]

    # sanity check
    if len(params) == 0:
        raise ValueError("no parameters found for training")

    optimizer = optimizer_cls(
        params,
        betas=(finetune_config.optimizer.adam_beta1, finetune_config.optimizer.adam_beta2),
        weight_decay=finetune_config.optimizer.adam_weight_decay,
        eps=finetune_config.optimizer.adam_epsilon,
    )

    return ema_unet, optimizer


def setup_train_val_dataloaders(
    finetune_config: FinetuneConfig,
    dataset_config: CO3DConfig,
    validation_dataset_config: CO3DConfig,
    accelerator: Accelerator,
):
    # Get the train dataset
    if isinstance(dataset_config, CO3DConfig):
        train_dataset = CO3DDataset(dataset_config)
    else:
        raise NotImplementedError("unsupported dataset_config", type(dataset_config))

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=finetune_config.training.train_batch_size,
        num_workers=finetune_config.training.dataloader_num_workers,
    )

    validation_dataloader = None
    if accelerator.is_main_process:
        if isinstance(validation_dataset_config, CO3DConfig):
            if dataset_config.max_sequences > -1:
                # exclude train sequences from being picked if we do generalization training
                validation_dataset_config.dataset_args.exclude_sequence += train_dataset.get_all_sequences()
            validation_dataset = CO3DDataset(validation_dataset_config)
        else:
            raise NotImplementedError("unsupported validation_dataset_config", type(validation_dataset_config))

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=True,
            batch_size=1,
            num_workers=1,
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        selected_sequences = {
            "selected_sequences": {
                "train": train_dataset.get_all_sequences(),
                "val": validation_dataset.get_all_sequences(),
            }
        }
        config = {**asdict(dataset_config), **asdict(finetune_config), **selected_sequences}
        accelerator.init_trackers(f"view-diff", config=convert_to_tensorboard_dict(config))
        with open(os.path.join(finetune_config.io.output_dir, f"config.json"), "w") as f:
            json.dump(config, f, indent=4)
            accelerator.trackers[0].writer.add_text(
                "Hparams", json.dumps(convert_to_tensorboard_dict(config), indent=4), global_step=0
            )

    # dreambooth dataset creation
    if isinstance(dataset_config, CO3DConfig) and finetune_config.training.dreambooth_prior_preservation_loss_weight > 0:
        dreambooth_train_dataset = CO3DDreamboothDataset(
            co3d_root=dataset_config.co3d_root,
            selected_categories=dataset_config.category.split(","),
            height=dataset_config.batch.image_height,
            width=dataset_config.batch.image_width,
        )

        dreambooth_train_dataloader = torch.utils.data.DataLoader(
            dreambooth_train_dataset,
            shuffle=True,
            batch_size=finetune_config.training.train_batch_size,
            num_workers=finetune_config.training.dataloader_num_workers,
        )
    else:
        dreambooth_train_dataloader = None

    return train_dataloader, validation_dataloader, len(train_dataset), dreambooth_train_dataloader


def setup_training(
    finetune_config: FinetuneConfig,
    accelerator: Accelerator,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer,
    ema_unet: EMAModel,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: Union[
        UNet2DConditionModel,
        UNet2DConditionCrossFrameInExistingAttnModel,
    ],
    train_dreambooth_dataloader: torch.utils.data.DataLoader = None,
):
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / finetune_config.optimizer.gradient_accumulation_steps
    )
    if finetune_config.training.max_train_steps is None:
        finetune_config.training.max_train_steps = (
            finetune_config.training.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        finetune_config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=finetune_config.optimizer.lr_warmup_steps
        * finetune_config.optimizer.gradient_accumulation_steps,
        num_training_steps=finetune_config.training.max_train_steps
        * finetune_config.optimizer.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if train_dreambooth_dataloader is not None:
        train_dreambooth_dataloader = accelerator.prepare(train_dreambooth_dataloader)

    if finetune_config.model.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / finetune_config.optimizer.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        finetune_config.training.max_train_steps = (
            finetune_config.training.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    finetune_config.training.num_train_epochs = math.ceil(
        finetune_config.training.max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        finetune_config.training.train_batch_size
        * accelerator.num_processes
        * finetune_config.optimizer.gradient_accumulation_steps
    )

    return (
        total_batch_size,
        num_update_steps_per_epoch,
        lr_scheduler,
        weight_dtype,
        unet,
        optimizer,
        train_dataloader,
        train_dreambooth_dataloader,
    )


def load_checkpoint(finetune_config: FinetuneConfig, accelerator: Accelerator, num_update_steps_per_epoch: int):
    if finetune_config.io.resume_from_checkpoint != "latest":
        path = finetune_config.io.resume_from_checkpoint
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(finetune_config.io.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = os.path.join(finetune_config.io.output_dir, dirs[-1]) if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{finetune_config.io.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        finetune_config.io.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[-1])
        if path[-1] != "/":
            path = path + "/"
        accelerator.load_state(path)

        resume_global_step = global_step * finetune_config.optimizer.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * finetune_config.optimizer.gradient_accumulation_steps
        )

    return global_step, first_epoch, resume_step


def save_checkpoint(finetune_config: FinetuneConfig, accelerator: Accelerator, logger, global_step: int):
    if accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if finetune_config.io.checkpoints_total_limit is not None:
            checkpoints = os.listdir(finetune_config.io.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = [
                d for d in checkpoints if int(d.split("-")[1]) % 5000 != 0
            ]  # never remove checkpoints that are saved every 5K steps
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= finetune_config.io.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - finetune_config.io.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(finetune_config.io.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(finetune_config.io.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


def maybe_continue_training(
    accelerator: Accelerator,
    finetune_config: FinetuneConfig,
    num_update_steps_per_epoch,
    logger,
    global_step,
):
    if finetune_config.io.resume_from_checkpoint:
        global_step, first_epoch, resume_step = load_checkpoint(
            finetune_config, accelerator, num_update_steps_per_epoch
        )
    elif accelerator.is_main_process:
        # sanity check: saving checkpoint is working
        save_checkpoint(finetune_config, accelerator, logger, global_step)
        finetune_config.io.resume_from_checkpoint = "latest"
        global_step, first_epoch, resume_step = load_checkpoint(
            finetune_config, accelerator, num_update_steps_per_epoch
        )
        assert (
            first_epoch == 0 and resume_step == 0
        ), "sanity loading should only happen at the beginning of a new training"
    else:
        first_epoch = 0
        resume_step = 0

    return global_step, first_epoch, resume_step
