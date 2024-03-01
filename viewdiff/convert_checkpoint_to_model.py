# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import tyro
import shutil

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from .model.custom_stable_diffusion_pipeline import CustomStableDiffusionPipeline

from .train_util import FinetuneConfig, unet_attn_processors_state_dict, load_models
from diffusers.loaders import LoraLoaderMixin

from dacite import from_dict, Config

from .train import update_model


def convert_checkpoint_to_model(
    checkpoint_path: str, keep_config_output_dir: bool = False, pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
):
    # load config from checkpoint_path
    if checkpoint_path[-1] == "/":
        # cannot have trailing slash in checkpoint path for dirname
        checkpoint_path = checkpoint_path[:-1]
    root_dir = os.path.dirname(checkpoint_path)
    ckpt_name = os.path.basename(checkpoint_path)
    if checkpoint_path[-1] != "/":
        # need trailing slash in checkpoint path
        checkpoint_path += "/"
    config_path = os.path.join(root_dir, "config.json")

    if not os.path.isfile(str(config_path)):
        raise ValueError("cannot find config.json in ", config_path)

    with open(config_path, "r") as f:
        config_data = json.load(f)
    finetune_config = from_dict(FinetuneConfig, data=config_data, config=Config(cast=[tuple, int]))

    if not keep_config_output_dir:
        finetune_config.io.output_dir = os.path.join(root_dir, f"saved_model_from_{ckpt_name}")

    if pretrained_model_name_or_path is not None:
        finetune_config.io.pretrained_model_name_or_path = pretrained_model_name_or_path

    # setup run
    accelerator_project_config = ProjectConfiguration(
        project_dir=finetune_config.io.output_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=finetune_config.optimizer.gradient_accumulation_steps,
        mixed_precision=finetune_config.training.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Load models.
    _, _, text_encoder, vae, orig_unet = load_models(
        finetune_config.io.pretrained_model_name_or_path, revision=finetune_config.io.revision
    )
    unet, unet_lora_parameters = update_model(finetune_config, orig_unet)
    model_cls = type(unet)
    orig_model_cls = type(orig_unet)

    def load_model_hook(models, input_dir):
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
                    print("Loaded LoRA weights into model")
            elif isinstance(model, orig_model_cls):
                in_dir = os.path.join(input_dir, "orig_unet")
                load_model = orig_model_cls.from_pretrained(in_dir)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
                print("Loaded orig_unet model")
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

    accelerator.register_load_state_pre_hook(load_model_hook)
    unet = accelerator.prepare(unet)

    # load in the weights and states from a previous save
    accelerator.print(f"Load checkpoint {checkpoint_path}")
    accelerator.load_state(checkpoint_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Save model at", finetune_config.io.output_dir)
        # Create the pipeline using the trained modules and save it.
        unet = accelerator.unwrap_model(unet)

        pipeline = CustomStableDiffusionPipeline.from_pretrained(
            finetune_config.io.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=finetune_config.io.revision,
        )
        pipeline.save_pretrained(finetune_config.io.output_dir)

    # save lora layers
    unet_lora_layers = unet_attn_processors_state_dict(unet)
    LoraLoaderMixin.save_lora_weights(save_directory=os.path.join(finetune_config.io.output_dir, "unet"), unet_lora_layers=unet_lora_layers)

    # copy config to new dir
    shutil.copy(config_path, os.path.join(finetune_config.io.output_dir, "config.json"))

    accelerator.print("Finished.")


if __name__ == "__main__":
    tyro.cli(convert_checkpoint_to_model)
