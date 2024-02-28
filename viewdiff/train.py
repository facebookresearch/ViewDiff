# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, List, Dict

import os
import tyro
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from .model.custom_unet_2d_condition import (
    UNet2DConditionCrossFrameInExistingAttnModel,
    get_down_block_types,
    get_mid_block_type,
    get_up_block_types,
)
from .model.util import (
    replace_self_attention_with_cross_frame_attention,
    update_last_layer_mode,
    update_vol_rend_inject_noise_sigma,
    update_n_novel_images,
    update_cross_frame_attention_config,
    add_pose_cond_to_attention_layers,
    collapse_prompt_to_batch_dim,
    collapse_tensor_to_batch_dim,
    expand_output_to_k,
    expand_tensor_to_k,
    tokenize_captions,
    ModelConfig,
    CrossFrameAttentionConfig,
    build_cross_attention_kwargs,
)
from .model.custom_stable_diffusion_pipeline import CustomStableDiffusionPipeline

from .io_util import (
    make_image_grid,
    norm_0_1,
    setup_output_directories,
    make_output_directories,
    save_inference_outputs,
    IOConfig,
)
from .train_util import (
    check_local_rank,
    FinetuneConfig,
    load_models,
    setup_accelerate,
    setup_model_and_optimizer,
    setup_train_val_dataloaders,
    setup_training,
    save_checkpoint,
    maybe_continue_training,
)

from .metrics.image_metrics import calc_psnr_ssim_lpips

from .data.co3d.co3d_dataset import CO3DConfig

from .scripts.misc.create_masked_images import remove_background


logger = get_logger(__name__, log_level="INFO")


def train_and_test(
    dataset_config: CO3DConfig,
    finetune_config: FinetuneConfig,
    validation_dataset_config: CO3DConfig,
):
    # manually update required fields in the config
    finetune_config.training = check_local_rank(finetune_config.training)
    dataset_config.batch.n_parallel_images = finetune_config.model.n_input_images
    validation_dataset_config.batch.n_parallel_images = finetune_config.model.n_input_images

    # setup run
    setup_output_directories(
        io_config=finetune_config.io, model_config=finetune_config.model, dataset_config=dataset_config, is_train=True
    )
    accelerator, generator = setup_accelerate(
        finetune_config=finetune_config, dataset_config=dataset_config, logger=logger
    )
    if accelerator.is_main_process:
        make_output_directories(io_config=finetune_config.io)

    # resume from latest checkpoint
    if finetune_config.io.experiment_name is not None and finetune_config.io.automatic_checkpoint_resume and os.path.exists(finetune_config.io.output_dir):
        checkpoints = os.listdir(finetune_config.io.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            last_checkpoint = checkpoints[-1]
            finetune_config.io.resume_from_checkpoint = os.path.join(finetune_config.io.output_dir, last_checkpoint)
            print("Found a checkpoint to resume training from automatically:", finetune_config.io.resume_from_checkpoint)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_models(
        finetune_config.io.pretrained_model_name_or_path, revision=finetune_config.io.revision
    )

    # Set denoising type
    noise_scheduler.config.prediction_type = finetune_config.training.noise_prediction_type

    # update U-Net architecture as specified in the config
    logger.info("Initializing the StableDiffusion3D UNet from the pretrained UNet.")
    unet, unet_lora_parameters = update_model(finetune_config, unet)

    # setup the rest for training
    ema_unet, optimizer = setup_model_and_optimizer(
        vae,
        text_encoder,
        unet,
        accelerator,
        finetune_config,
        logger,
        unet_lora_parameters=unet_lora_parameters,
    )
    train_dataloader, validation_dataloader, n_train_examples, train_dreambooth_dataloader = setup_train_val_dataloaders(
        finetune_config, dataset_config, validation_dataset_config, accelerator
    )
    (
        total_batch_size,
        num_update_steps_per_epoch,
        lr_scheduler,
        weight_dtype,
        unet,
        optimizer,
        train_dataloader,
        train_dreambooth_dataloader
    ) = setup_training(
        finetune_config,
        accelerator,
        train_dataloader,
        optimizer,
        ema_unet,
        text_encoder,
        vae,
        unet,
        train_dreambooth_dataloader=train_dreambooth_dataloader
    )
    if accelerator.is_main_process:
        validation_batch = next(iter(validation_dataloader))  # always use fixed validation_batch
    if train_dreambooth_dataloader is not None:
        train_dreambooth_dataloader_iter = iter(train_dreambooth_dataloader)
        dreambooth_iter_sentinel = object()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {n_train_examples}")
    logger.info(f"  Num Epochs = {finetune_config.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {finetune_config.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {finetune_config.optimizer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {finetune_config.training.max_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    global_step, first_epoch, resume_step = maybe_continue_training(
        accelerator, finetune_config, num_update_steps_per_epoch, logger, global_step
    )

    # check model modifications
    if finetune_config.training.remove_cfa_skip_connections_at_iter >= global_step:
        update_last_layer_mode(
            unet=accelerator.unwrap_model(unet), last_layer_mode=finetune_config.training.changed_cfa_last_layer
        )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, finetune_config.training.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # check if train_dataloader should be modified to show the
    first_epoch_use_different_dataloader = False
    first_epoch_train_dataloader = None
    if finetune_config.io.resume_from_checkpoint and resume_step > 0:
        logger.info(f"Will skip the first {resume_step} steps in dataloader.")
        first_epoch_use_different_dataloader = True
        first_epoch_train_dataloader = skip_first_batches(dataloader=train_dataloader, num_batches=resume_step)
        progress_bar.update(resume_step // finetune_config.optimizer.gradient_accumulation_steps)

    for epoch in range(first_epoch, finetune_config.training.num_train_epochs):
        # ################
        # Val Loop
        # ################
        unet.eval()
        update_vol_rend_inject_noise_sigma(accelerator.unwrap_model(unet), 0.0)  # disable vol-rend noise
        update_n_novel_images(accelerator.unwrap_model(unet), 0)  # disable skipping frame in inference mode
        torch.cuda.empty_cache()
        if (
            accelerator.is_main_process
            and finetune_config.training.validation_epochs > 0
            and (epoch % finetune_config.training.validation_epochs) == 0
        ):
            logger.info(f"Running validation...")
            # create pipeline
            if finetune_config.model.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            # The models need unwrapping because for compatibility in distributed training mode.
            pipeline = CustomStableDiffusionPipeline.from_pretrained(
                finetune_config.io.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                revision=finetune_config.io.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=False)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.scheduler.config.prediction_type = finetune_config.training.noise_prediction_type

            # run inference on one batch of the validation set
            test_step(
                pipeline=pipeline,
                batch=validation_batch,
                model_config=finetune_config.model,
                cfa_config=finetune_config.cross_frame_attention,
                io_config=finetune_config.io,
                generator=generator,
                prefix="Validation",
                global_step=global_step,
                writer=accelerator.trackers[0].writer,
                orig_hw=(validation_dataset_config.batch.image_height, validation_dataset_config.batch.image_width),
            )

            if finetune_config.model.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())

            del pipeline
            torch.cuda.empty_cache()

        # ################
        # Train Loop
        # ################
        unet.train()
        update_vol_rend_inject_noise_sigma(accelerator.unwrap_model(unet), 1.0)  # enable vol-rend noise
        update_n_novel_images(accelerator.unwrap_model(unet), finetune_config.cross_frame_attention.n_novel_images)  # enable skipping frame in inference mode
        torch.cuda.empty_cache()
        train_losses = {}
        train_accs = {}
        acc_per_timestep = {}
        logger.info(f"Running training...")
        dataloader_to_use = train_dataloader if epoch != first_epoch or not first_epoch_use_different_dataloader else first_epoch_train_dataloader
        for step, batch in enumerate(dataloader_to_use):
            # Skip steps until we reach the resumed step
            if not first_epoch_use_different_dataloader and finetune_config.io.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % finetune_config.optimizer.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # check model modifications
            if (
                finetune_config.training.remove_cfa_skip_connections_at_iter > -1
                and finetune_config.training.remove_cfa_skip_connections_at_iter == global_step
            ):
                update_last_layer_mode(
                    unet=accelerator.unwrap_model(unet), last_layer_mode=finetune_config.training.changed_cfa_last_layer
                )

            # get dreambooth batch
            if train_dreambooth_dataloader is not None and (global_step % finetune_config.training.dreambooth_prior_preservation_every_nth) == 0:
                # get batch
                dreambooth_batch = next(train_dreambooth_dataloader_iter, dreambooth_iter_sentinel)
                if dreambooth_batch is dreambooth_iter_sentinel:
                    # end of dreambooth dataloader reached -- restart with new iterator from the beginning
                    train_dreambooth_dataloader_iter = iter(train_dreambooth_dataloader)
                    dreambooth_batch = next(train_dreambooth_dataloader_iter, dreambooth_iter_sentinel)
                    assert dreambooth_batch is not dreambooth_iter_sentinel
                dreambooth_batch["is_dreambooth"] = True
            else:
                dreambooth_batch = None

            # do training step
            avg_step_losses, acc_step, loss = train_step(
                accelerator=accelerator,
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                finetune_config=finetune_config,
                batch=batch,
                generator=generator,
                weight_dtype=weight_dtype,
                global_step=global_step,
                orig_hw=(dataset_config.batch.image_height, dataset_config.batch.image_width),
                dreambooth_batch=dreambooth_batch,
            )

            # accumulate losses for logging across gradient_accumulation_steps
            for k, v in avg_step_losses.items():
                if k not in train_losses:
                    train_losses[k] = 0.0
                train_losses[k] += v

            # accumulate accs for logging across gradient_accumulation_steps
            if "timesteps" in acc_step:
                timesteps = acc_step["timesteps"]
                for k, v in acc_step.items():
                    if "timesteps" in k:
                        continue

                    # mean acc (averaged across timesteps for logging per step)
                    if k not in train_accs:
                        train_accs[k] = 0.0
                    train_accs[k] += v.mean().item() / accelerator.gradient_accumulation_steps

                    # per-timestep acc (averaged across steps for logging per timestep)
                    if k not in acc_per_timestep:
                        acc_per_timestep[k] = {t: (0, 0) for t in range(1000)}
                    for idx in range(v.shape[0]):
                        t = timesteps[idx].item()
                        prev_val, prev_count = acc_per_timestep[k][t]
                        acc_per_timestep[k][t] = (prev_val + v[idx].item(), prev_count + 1)

            # Checks if the accelerator has performed an optimization step behind the scenes (e.g. when gradient_accumulation_steps are reached)
            if accelerator.sync_gradients:
                if finetune_config.model.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                for k, v in train_losses.items():
                    if finetune_config.io.log_images_every_nth > -1:
                        try:
                            accelerator.log({f"Train/Loss/{k}": v}, step=global_step)
                        except Exception as e:
                            print("logging failed", e)
                    train_losses[k] = 0.0
                for k, v in train_accs.items():
                    if finetune_config.io.log_images_every_nth > -1:
                        try:
                            accelerator.log({f"Train/Acc/{k}": v}, step=global_step)
                        except Exception as e:
                            print("logging failed", e)
                    train_accs[k] = 0.0

                if global_step % finetune_config.io.checkpointing_steps == 0:
                    save_checkpoint(finetune_config, accelerator, logger, global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= finetune_config.training.max_train_steps:
                break

        # log acc_per_timestep after each epoch
        if accelerator.is_main_process and finetune_config.io.log_images_every_nth > -1:
            for k, v in acc_per_timestep.items():
                for t, (acc_sum, acc_count) in v.items():
                    if acc_count > 0:
                        acc = acc_sum / acc_count
                        try:
                            accelerator.log({f"Train/Acc-Per-Timestep/{k}/epoch-{epoch}": acc}, step=t)
                        except Exception as e:
                            print("logging failed", e)

    accelerator.end_training()


def update_model(finetune_config: FinetuneConfig, unet: UNet2DConditionModel):
    if finetune_config.cross_frame_attention.mode != "none":
        use_lora_in_cfa = "cfa" in finetune_config.model.pose_cond_mode
        if (
            finetune_config.cross_frame_attention.mode == "pretrained"
            or finetune_config.cross_frame_attention.mode == "add_in_existing_block"
        ):
            if finetune_config.cross_frame_attention.mode == "pretrained":
                # overwrite the settings for cfa to not create the cfa layers
                # instead we want to re-use the sa layers for it
                if finetune_config.cross_frame_attention.unproj_reproj_mode == "with_cfa":
                    finetune_config.cross_frame_attention.unproj_reproj_mode = "only_unproj_reproj"

            unet = UNet2DConditionCrossFrameInExistingAttnModel.from_source(
                src=unet,
                load_weights=True,
                down_block_types=get_down_block_types(finetune_config.cross_frame_attention.n_cfa_down_blocks),
                mid_block_type=get_mid_block_type(not finetune_config.cross_frame_attention.no_cfa_in_mid_block),
                up_block_types=get_up_block_types(finetune_config.cross_frame_attention.n_cfa_up_blocks),
                n_input_images=finetune_config.model.n_input_images,
                to_k_other_frames=finetune_config.cross_frame_attention.to_k_other_frames,
                random_others=finetune_config.cross_frame_attention.random_others,
                last_layer_mode=finetune_config.cross_frame_attention.last_layer_mode,
                use_lora_in_cfa=use_lora_in_cfa,
                use_temb_in_lora=finetune_config.cross_frame_attention.use_temb_cond,
                temb_out_size=8,
                pose_cond_dim=finetune_config.model.pose_cond_dim,
                rank=finetune_config.model.pose_cond_lora_rank,
                unproj_reproj_mode=finetune_config.cross_frame_attention.unproj_reproj_mode,
                dim_3d_grid=finetune_config.cross_frame_attention.dim_3d_grid,
                dim_3d_latent=finetune_config.cross_frame_attention.dim_3d_latent,
                n_novel_images=finetune_config.cross_frame_attention.n_novel_images,
                num_3d_layers=finetune_config.cross_frame_attention.num_3d_layers,
                vol_rend_proj_in_mode=finetune_config.cross_frame_attention.vol_rend_proj_in_mode,
                vol_rend_aggregator_mode=finetune_config.cross_frame_attention.vol_rend_aggregator_mode,
                vol_rend_proj_out_mode=finetune_config.cross_frame_attention.vol_rend_proj_out_mode,
                vol_rend_model_background=finetune_config.cross_frame_attention.vol_rend_model_background,
                vol_rend_background_grid_percentage=finetune_config.cross_frame_attention.vol_rend_background_grid_percentage,
                vol_rend_disparity_at_inf=finetune_config.cross_frame_attention.vol_rend_disparity_at_inf,
            )

            if finetune_config.cross_frame_attention.mode == "pretrained":
                # TODO: allow to only replace the layers as specified in finetune_config.cross_frame_attention.n_cfa_down_blocks
                replace_self_attention_with_cross_frame_attention(
                    unet=unet,
                    n_input_images=finetune_config.model.n_input_images,
                    to_k_other_frames=finetune_config.cross_frame_attention.to_k_other_frames,
                    with_self_attention=finetune_config.cross_frame_attention.with_self_attention,
                    random_others=finetune_config.cross_frame_attention.random_others,
                    use_lora_in_cfa=use_lora_in_cfa or "sa" in finetune_config.model.pose_cond_mode,
                    use_temb_in_lora=finetune_config.cross_frame_attention.use_temb_cond,
                    temb_out_size=8,
                    pose_cond_dim=finetune_config.model.pose_cond_dim,
                    rank=finetune_config.model.pose_cond_lora_rank,
                )
        else:
            raise NotImplementedError(
                "unsupported cross_frame_attention.mode", finetune_config.cross_frame_attention.mode
            )

    unet_lora_parameters = None
    if finetune_config.model.pose_cond_mode != "none":
        # Set correct lora layers
        unet_lora_attn_procs, unet_lora_parameters = add_pose_cond_to_attention_layers(
            unet,
            rank=finetune_config.model.pose_cond_lora_rank,
            pose_cond_dim=finetune_config.model.pose_cond_dim,
            only_cross_attention="sa" not in finetune_config.model.pose_cond_mode,
        )

    return unet, unet_lora_parameters


def train_step(
    accelerator: Accelerator,
    unet: UNet2DConditionCrossFrameInExistingAttnModel,
    vae: AutoencoderKL,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    optimizer,
    lr_scheduler,
    finetune_config: FinetuneConfig,
    batch,
    generator,
    weight_dtype,
    global_step,
    orig_hw,
    dreambooth_batch = None
):
    def process_batch(batch):
        # parse batch
        # collapse K dimension into batch dimension (no concatenation happening)
        batch["prompt"] = collapse_prompt_to_batch_dim(batch["prompt"], finetune_config.model.n_input_images)
        batch_size, pose = collapse_tensor_to_batch_dim(batch["pose"])
        _, K = collapse_tensor_to_batch_dim(batch["K"])
        _, intensity_stats = collapse_tensor_to_batch_dim(batch["intensity_stats"])
        pose = pose.squeeze(1)
        K = K.squeeze(1)[..., :3, :3]
        intensity_stats = intensity_stats.squeeze(1)
        bbox = batch["bbox"]
        log_prefix = ""
        tb_log_prefix = ""
        is_dreambooth = batch.get("is_dreambooth", False)
        if is_dreambooth:
            log_prefix = "Dreambooth_"
            tb_log_prefix = "/Dreambooth"

        # build cross_attention_kwargs
        cross_attention_kwargs = build_cross_attention_kwargs(
            model_config=finetune_config.model,
            cfa_config=finetune_config.cross_frame_attention,
            pose=pose,
            K=K,
            intensity_stats=intensity_stats if finetune_config.model.pose_cond_dim > 8 else None,
            bbox=bbox,
            orig_hw=orig_hw,
        )

        if "images" in batch:
            # convert images to latent space.
            _, images = collapse_tensor_to_batch_dim(batch["images"])
            images = images.squeeze(1)
            images = images[:, :3].to(weight_dtype)  # remove alpha channel
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        else:
            raise ValueError("images not found in batch")

        # Sample a random timestep for each batch
        N = latents.shape[0]
        f_min = 0.0  # 0.02
        f_max = 1.0  # 0.98
        t_min = int(f_min * noise_scheduler.config.num_train_timesteps)
        t_max = int(f_max * noise_scheduler.config.num_train_timesteps)
        timesteps = torch.randint(t_min, t_max, (batch_size,), device=latents.device, dtype=torch.long)

        # per default: use same timestep within batch
        timesteps = timesteps.repeat_interleave(finetune_config.model.n_input_images)

        # check if some timesteps within the batch will be replaced with 0 (== non-noisy image)
        if not is_dreambooth and finetune_config.training.prob_images_not_noisy > 0:
            random_p_non_noisy = torch.rand((batch_size, finetune_config.model.n_input_images), device=latents.device, generator=generator)
            non_noisy_mask = random_p_non_noisy < finetune_config.training.prob_images_not_noisy
            non_noisy_mask[:, finetune_config.training.max_num_images_not_noisy:] = False
            non_noisy_mask = non_noisy_mask.flatten()
            timesteps = torch.where(non_noisy_mask, torch.zeros_like(timesteps), timesteps)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # convert prompt to input_ids
        batch["input_ids"] = tokenize_captions(tokenizer, batch["prompt"]).to(latents.device)

        # Get the text embedding for conditioning.
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        # Conditioning dropout to support classifier-free guidance during inference.
        if finetune_config.model.conditioning_dropout_prob is not None:
            random_p = torch.rand(N, device=latents.device, generator=generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * finetune_config.model.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(N, 1, 1)
            # Final text conditioning.
            null_conditioning = text_encoder(tokenize_captions(tokenizer, [""]).to(accelerator.device))[0]
            encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

        # Get the target for unet-pred loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "sample":
            unet_pred_target = latents
        elif noise_scheduler.config.prediction_type == "epsilon":
            unet_pred_target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            unet_pred_target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict w/ unet
        output = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        unet_pred = output.unet_sample
        rendered_depth_per_layer = output.rendered_depth
        rendered_mask_per_layer = output.rendered_mask

        # only compute losses for those batches that have non-zero timestep
        # the other images are perfect (no noise), so predicting noise is ambiguous and we do not want to receive gradients for it
        # instead, those images are only used as conditioning input in the cfa and proj layers
        if not is_dreambooth and finetune_config.training.prob_images_not_noisy > 0:
            unet_pred_target[non_noisy_mask] = unet_pred[non_noisy_mask].detach().clone().to(unet_pred_target)

        # compute unet-pred-loss
        unet_pred_acc = F.mse_loss(unet_pred.float(), unet_pred_target.float(), reduction="none")
        loss = unet_pred_acc.mean()
        unet_pred_acc = unet_pred_acc.mean(dim=(1, 2, 3))

        if is_dreambooth:
            loss = finetune_config.training.dreambooth_prior_preservation_loss_weight * loss

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(finetune_config.training.train_batch_size)).mean()
        avg_loss = avg_loss.item() / finetune_config.optimizer.gradient_accumulation_steps
        avg_losses = {
            f"{log_prefix}Loss": avg_loss,
        }

        # Gather the acc across all processes for logging (if we use distributed training).
        acc = {
            f"{log_prefix}timesteps": accelerator.gather(timesteps),
            f"{log_prefix}UNet-Pred": accelerator.gather(unet_pred_acc).detach(),
        }

        # logging
        if accelerator.is_main_process and finetune_config.io.log_images_every_nth > -1 and (global_step % finetune_config.io.log_images_every_nth) == 0:
            try:
                # add input images to tensorboard
                if "images" in batch:
                    input_image_grid = make_image_grid(
                        expand_tensor_to_k(images.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                    )
                    accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Input", input_image_grid, global_step=global_step)

                    image_noise = torch.randn_like(images)
                    noisy_images = noise_scheduler.add_noise(images, image_noise, timesteps)

                    image_noise_grid = make_image_grid(
                        expand_tensor_to_k(image_noise.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                    )
                    accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Input-Noise", image_noise_grid, global_step=global_step)

                    input_noisy_image_grid = make_image_grid(
                        expand_tensor_to_k(noisy_images.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                    )
                    accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Input-Noisy", input_noisy_image_grid, global_step=global_step)

                # add non-noisy latents to tensorboard
                latent_grid = make_image_grid(
                    expand_tensor_to_k(latents.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                )
                accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Latent", latent_grid, global_step=global_step)

                # add noisy latents to tensorboard
                noisy_latent_grid = make_image_grid(
                    expand_tensor_to_k(noisy_latents.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                )
                accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Noisy-Latent", noisy_latent_grid, global_step=global_step)

                # add noise to tensorboard
                noise_grid = make_image_grid(
                    expand_tensor_to_k(noise.unsqueeze(1), batch_size, finetune_config.model.n_input_images)
                )
                accelerator.trackers[0].writer.add_image(f"Train{tb_log_prefix}/Noise", noise_grid, global_step=global_step)

                # add rendered-depth and rendered-mask
                if (
                    rendered_depth_per_layer is not None
                    and rendered_mask_per_layer is not None
                ):
                    for i, (d, m) in enumerate(zip(rendered_depth_per_layer, rendered_mask_per_layer)):
                        rendered_depth_mask_grid = make_image_grid(norm_0_1(d), norm_0_1(m))
                        accelerator.trackers[0].writer.add_image(
                            f"Train{tb_log_prefix}/Rendered-Depth-Mask/{i}", rendered_depth_mask_grid, global_step=global_step
                        )
            except Exception as e:
                print("logging failed", e)

        # Backpropagate
        accelerator.backward(loss)

        return avg_losses, acc, loss

    # forward pass * backprop batch
    with accelerator.accumulate(unet):
        avg_losses, acc, loss = process_batch(batch)

    if dreambooth_batch is not None:
        # update model cfa config to process a single image
        old_n_input_images = finetune_config.model.n_input_images
        old_to_k_other_frames = finetune_config.cross_frame_attention.to_k_other_frames
        old_n_novel_images = finetune_config.cross_frame_attention.n_novel_images
        old_with_self_attention = finetune_config.cross_frame_attention.with_self_attention
        finetune_config.model.n_input_images = 1
        finetune_config.cross_frame_attention.to_k_other_frames = 0
        finetune_config.cross_frame_attention.n_novel_images = 0
        finetune_config.cross_frame_attention.with_self_attention = True

        update_cross_frame_attention_config(
            accelerator.unwrap_model(unet),
            n_input_images=finetune_config.model.n_input_images,
            to_k_other_frames=finetune_config.cross_frame_attention.to_k_other_frames,
            with_self_attention=finetune_config.cross_frame_attention.with_self_attention,
            random_others=finetune_config.cross_frame_attention.random_others,
            change_self_attention_layers=finetune_config.cross_frame_attention.mode == "pretrained",
        )
        update_n_novel_images(accelerator.unwrap_model(unet), n_novel_images=finetune_config.cross_frame_attention.n_novel_images)

        # forward pass * backprop dreambooth batch
        with accelerator.accumulate(unet):
            dreambooth_avg_losses, dreambooth_acc, dreambooth_loss = process_batch(dreambooth_batch)
            loss += dreambooth_loss
            acc = {**acc, **dreambooth_acc}
            avg_losses = {**avg_losses, **dreambooth_avg_losses}

        # reset model cfa config to process multiple images
        finetune_config.model.n_input_images = old_n_input_images
        finetune_config.cross_frame_attention.to_k_other_frames = old_to_k_other_frames
        finetune_config.cross_frame_attention.n_novel_images = old_n_novel_images
        finetune_config.cross_frame_attention.with_self_attention = old_with_self_attention

        update_cross_frame_attention_config(
            accelerator.unwrap_model(unet),
            n_input_images=finetune_config.model.n_input_images,
            to_k_other_frames=finetune_config.cross_frame_attention.to_k_other_frames,
            with_self_attention=finetune_config.cross_frame_attention.with_self_attention,
            random_others=finetune_config.cross_frame_attention.random_others,
            change_self_attention_layers=finetune_config.cross_frame_attention.mode == "pretrained",
        )
        update_n_novel_images(accelerator.unwrap_model(unet), n_novel_images=finetune_config.cross_frame_attention.n_novel_images)

    # Optim Step
    with accelerator.accumulate(unet):
        if accelerator.sync_gradients:
            clip_grad_parameters = list(unet.parameters())
            accelerator.clip_grad_norm_(clip_grad_parameters, finetune_config.optimizer.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return avg_losses, acc, loss


@torch.autocast("cuda")
def test_step(
    pipeline: CustomStableDiffusionPipeline,
    batch,
    model_config: ModelConfig,
    cfa_config: CrossFrameAttentionConfig,
    io_config: IOConfig,
    orig_hw,
    guidance_scale: float = 7.5,
    generator=None,
    prefix: str = None,
    global_step: int = 0,
    writer=None,
    deactivate_view_dependent_rendering: bool = False,
    num_inference_steps: int = 50,
    n_repeat_generation: int = 1,
    lpips_vgg_model=None,
    carvekit_model=None,
):
    batch_size = len(batch["prompt"])

    # parse batch
    # collapse K dimension into batch dimension (no concatenation happening)
    prompt = collapse_prompt_to_batch_dim(batch["prompt"], model_config.n_input_images)
    _, pose = collapse_tensor_to_batch_dim(batch["pose"])
    _, K = collapse_tensor_to_batch_dim(batch["K"])
    _, intensity_stats = collapse_tensor_to_batch_dim(batch["intensity_stats"])
    bbox = batch["bbox"]

    if "known_images" in batch and batch["known_images"] is not None:
        assert batch_size == 1, "sliding window currently only supported for batch-size 1"
        # TODO need to handle padding in collapse for bs>1
        _, known_images = collapse_tensor_to_batch_dim(batch["known_images"])
        known_images = known_images.to(pipeline.device)
        known_images = known_images.squeeze(1)
    else:
        known_images = None

    pose = pose.to(pipeline.device)
    K = K.to(pipeline.device)
    intensity_stats = intensity_stats.to(pipeline.device)
    bbox = bbox.to(pipeline.device)

    K = K.squeeze(1)[..., :3, :3]
    pose = pose.squeeze(1)
    intensity_stats = intensity_stats.squeeze(1)

    # build cross_attention_kwargs
    cross_attention_kwargs = build_cross_attention_kwargs(
        model_config=model_config,
        cfa_config=cfa_config,
        pose=pose,
        K=K,
        intensity_stats=intensity_stats,
        bbox=bbox,
        orig_hw=orig_hw,
    )

    if deactivate_view_dependent_rendering:
        cross_attention_kwargs["unproj_reproj_kwargs"]["deactivate_view_dependent_rendering"] = True

    # check classifier-free-guidance
    if guidance_scale > 1:
        if "pose_cond" in cross_attention_kwargs:
            cross_attention_kwargs["pose_cond"] = torch.cat([cross_attention_kwargs["pose_cond"]] * 2)
        if "unproj_reproj_kwargs" in cross_attention_kwargs:
            proj_kwargs = cross_attention_kwargs["unproj_reproj_kwargs"]
            proj_kwargs["pose"] = torch.cat([proj_kwargs["pose"]] * 2)
            proj_kwargs["K"] = torch.cat([proj_kwargs["K"]] * 2)
            proj_kwargs["bbox"] = torch.cat([proj_kwargs["bbox"]] * 2)

    # run denoising prediction + calc psnr/lpips/ssim
    outputs = []
    all_psnrs = []
    all_lpipses = []
    all_ssims = []
    for _ in range(n_repeat_generation):
        output = pipeline(
            prompt=prompt,
            height=orig_hw[0],
            width=orig_hw[1],
            known_images=known_images,
            output_type="pt",  # return tensor normalized to [0, 1]
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_scale=guidance_scale,
            decode_all_timesteps=True,
            num_inference_steps=num_inference_steps,
            n_images_per_batch=model_config.n_input_images,
        )

        # re-create K dimension from batch dimension
        output.images = output.images.unsqueeze(1)
        expand_output_to_k(output, batch_size, model_config.n_input_images)

        outputs.append(output)

        # calculate PSNR/SSIM/LPIPS metrics
        if lpips_vgg_model is not None and carvekit_model is not None:
            assert known_images is not None, "it does not make sense to calc psnr if we do not have known_images in the batch!"
            n_known_images_per_batch = batch["known_images"].shape[1]

            # calculate unmasked version
            src = output.images[:, n_known_images_per_batch:]
            target = (batch["images"][:, n_known_images_per_batch:].to(pipeline.device) + 1) * 0.5
            psnrs, lpipses, ssims = calc_psnr_ssim_lpips(
                src=src,
                target=target,
                lpips_vgg_model=lpips_vgg_model
            )

            # calculate masked version
            mask = batch["foreground_prob"].to(pipeline.device).unsqueeze(2).repeat(1, 1, 3, 1, 1)[:, n_known_images_per_batch:]
            src_masked = src * mask + torch.ones_like(src) * (1 - mask)
            target_masked = target * mask + torch.ones_like(target) * (1 - mask)
            masked_psnrs, masked_lpipses, masked_ssims = calc_psnr_ssim_lpips(
                src=src_masked,
                target=target_masked,
                lpips_vgg_model=lpips_vgg_model
            )

            # calculate carvekit-masked version
            mask = torch.zeros_like(src_masked, dtype=torch.bool)
            for i in range(src.shape[0]):
                for k in range(src.shape[1]):
                    x = (src[i, k].permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)
                    src_masked_, src_mask = remove_background(image=x, mask_predictor=carvekit_model)
                    mask[i, k] = torch.from_numpy(src_mask).permute(2, 0, 1).repeat(3, 1, 1).to(src_masked.device)
            mask = mask.float()
            src_masked = src * (1 - mask) + torch.ones_like(src) * mask
            target_masked = target * (1 - mask) + torch.ones_like(target) * mask
            carvekit_masked_psnrs, carvekit_masked_lpipses, carvekit_masked_ssims = calc_psnr_ssim_lpips(
                src=src_masked,
                target=target_masked,
                lpips_vgg_model=lpips_vgg_model
            )

            # add
            all_psnrs.append({"with-bkgd": psnrs, "only-fg": masked_psnrs, "only-fg-carvekit": carvekit_masked_psnrs})
            all_lpipses.append({"with-bkgd": lpipses, "only-fg": masked_lpipses, "only-fg-carvekit": carvekit_masked_lpipses})
            all_ssims.append({"with-bkgd": ssims, "only-fg": masked_ssims, "only-fg-carvekit": carvekit_masked_ssims})

    # select first for saving & opt-flow
    output = outputs[0]

    if lpips_vgg_model is not None and carvekit_model is not None:
        # calc metrics from average version of image
        average_image = torch.stack([x.images for x in outputs]).mean(dim=0, keepdim=False)[:, n_known_images_per_batch:]
        avg_psnrs, avg_lpipses, avg_ssims = calc_psnr_ssim_lpips(
            src=average_image,
            target=target,
            lpips_vgg_model=lpips_vgg_model
        )

        # calculate masked version
        masked_average_image = average_image * mask + torch.ones_like(average_image) * (1 - mask)
        masked_avg_psnrs, masked_avg_lpipses, masked_avg_ssims = calc_psnr_ssim_lpips(
            src=masked_average_image,
            target=target_masked,
            lpips_vgg_model=lpips_vgg_model
        )

        # store in output for later saving
        output.image_metrics = {
            "n_repeat_generation": n_repeat_generation,
            "batch-size": batch_size,
            "n_images_per_batch": model_config.n_input_images,
            "n_known_images_per_batch": n_known_images_per_batch,
            "psnr": {
                "all": all_psnrs,
                "from-avg-image": {
                    "with-bkgd": avg_psnrs,
                    "only-fg": masked_avg_psnrs
                },
            },
            "lpips": {
                "all": all_lpipses,
                "from-avg-image": {
                    "with-bkgd": avg_lpipses,
                    "only-fg": masked_avg_lpipses
                },
            },
            "ssim": {
                "all": all_ssims,
                "from-avg-image": {
                    "with-bkgd": avg_ssims,
                    "only-fg": masked_avg_ssims
                },
            }
        }

    try:
        save_inference_outputs(
            batch=batch,
            output=output,
            io_config=io_config,
            writer=writer,
            step=global_step,
            prefix=prefix,
        )
    except Exception as e:
        print("writing inference outputs failed", e)

    return output


if __name__ == "__main__":
    tyro.cli(train_and_test)
