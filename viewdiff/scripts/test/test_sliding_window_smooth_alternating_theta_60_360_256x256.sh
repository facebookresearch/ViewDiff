#!/bin/bash

export CO3DV2_DATASET_ROOT=$1

python -m viewdiff.test \
--run-config.pretrained_model_name_or_path $2 \
--run-config.output_dir $3 \
--run-config.n_input_images "10" \
--run-config.create_nerf_exports \
--run-config.save.pred_gif \
--run-config.sliding_window.is_active \
--run-config.sliding_window.create_smooth_video \
--run-config.sliding_window.repeat_first_n_steps 1 \
--run-config.sliding_window.n_full_batches_to_save 1 \
--run-config.sliding_window.perc_add_images_to_save 0.5 \
--run-config.sliding_window.max_degrees 60 \
--run-config.sliding_window.degree_increment 50 \
--run-config.sliding_window.first_theta 60.0 \
--run-config.sliding_window.min_theta 60.0 \
--run-config.sliding_window.max_theta 60.0 \
--run-config.sliding_window.first_radius 4.0 \
--run-config.sliding_window.min_radius 4.0 \
--run-config.sliding_window.max_radius 4.0 \
--run-config.num_inference_steps 10 \
--run_config.scheduler_type "unipc" \
--run_config.max_steps 6 \
--dataset-config.co3d-root $CO3DV2_DATASET_ROOT \
--dataset-config.category $4 \
--dataset-config.batch.other_selection "sequence" \
--dataset-config.batch.sequence_offset 1 \
--dataset-config.batch.load_recentered \
--dataset-config.batch.use_blip_prompt \
--dataset-config.batch.crop "foreground" \
--dataset-config.batch.image_width 256 \
--dataset-config.batch.image_height 256 \
--dataset-config.seed 500