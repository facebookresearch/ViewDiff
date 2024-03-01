#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

export CO3DV2_DATASET_ROOT=$1

python -m viewdiff.test \
--run-config.pretrained_model_name_or_path "$2" \
--run-config.output_dir $3 \
--run-config.n_input_images 5 \
--run-config.sliding_window.input_condition_mode "dataset" \
--run_config.sliding_window.input_condition_n_images 1 \
--run-config.num_inference_steps 10 \
--run_config.scheduler_type "unipc" \
--run_config.max_steps 200 \
--run_config.guidance_scale 1.0 \
--run_config.n_repeat_generation 20 \
--dataset-config.co3d-root $CO3DV2_DATASET_ROOT \
--dataset-config.category "$4" \
--dataset-config.batch.other_selection "random" \
--dataset-config.batch.load_recentered \
--dataset-config.batch.crop "foreground" \
--dataset-config.batch.prompt "a $4" \
--dataset-config.batch.image_width 256 \
--dataset-config.batch.image_height 256 \
--dataset-config.dataset_args.load_masks
