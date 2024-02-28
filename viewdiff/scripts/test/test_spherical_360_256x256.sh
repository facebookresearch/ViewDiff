#!/bin/bash

export CO3DV2_DATASET_ROOT=$1

python -m viewdiff.test \
--run-config.pretrained_model_name_or_path "$2" \
--run-config.output_dir $3 \
--run-config.n_input_images "$4" \
--run-config.num_inference_steps 10 \
--run_config.scheduler_type "unipc" \
--dataset-config.co3d-root $CO3DV2_DATASET_ROOT \
--dataset-config.category "$5" \
--dataset-config.dataset_args.pick_sequence "$6" \
--dataset-config.batch.other_selection "sequence" \
--dataset-config.batch.sequence_offset 1 \
--dataset-config.batch.load_recentered \
--dataset-config.batch.use_blip_prompt \
--dataset-config.batch.replace_pose_with_spherical_start_phi 0 \
--dataset-config.batch.replace_pose_with_spherical_end_phi 360 \
--dataset-config.batch.crop "foreground" \
--dataset-config.batch.image_width 256 \
--dataset-config.batch.image_height 256 \
