# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import os
import tyro
import json


@torch.no_grad()
def main(
    input_path: str
):
    # get all metrics from several batches
    metrics_files = [os.path.join(input_path, x) for x in os.listdir(input_path) if "metrics" in x and "combined" not in x]

    # create per-frame average and maximum metrics, as well as the metrics from the "average-image" (see viewset-diffusion for more details...)
    avg_metrics_dict = {}
    best_metrics_dict = {}
    avg_image_metrics_dict = {}
    invalid_indices_dict = {}
    all_offset = 0
    for mf in metrics_files:
        # open metric file & read header infos
        with open(mf, "r") as f:
            mf = json.load(f)
        batch_size = mf["batch-size"]
        n_repeat_generation = mf["n_repeat_generation"]
        if "n_images_per_batch" in mf and "n_known_images_per_batch" in mf:
            n_images_per_batch = mf["n_images_per_batch"] - mf["n_known_images_per_batch"]
        else:
            n_images_per_batch = 1

        # these metrics should be in the file, loop over them...
        for metric_type in ["psnr", "lpips", "ssim"]:
            # create empty dicts for each metric in the final dicts
            if metric_type not in avg_metrics_dict:
                avg_metrics_dict[metric_type] = {}
            if metric_type not in best_metrics_dict:
                best_metrics_dict[metric_type] = {}
            if metric_type not in avg_image_metrics_dict:
                avg_image_metrics_dict[metric_type] = {}

            # go through all per-frame versions
            all_scores = mf[metric_type]["all"]
            assert len(all_scores) == n_repeat_generation
            for score_dict in all_scores:
                for key, val_list in score_dict.items():
                    assert len(val_list) == batch_size * n_images_per_batch

                    # mark inf values as invalid
                    for val_idx in range(len(val_list)):
                        if val_list[val_idx] == float("inf"):
                            if metric_type not in invalid_indices_dict:
                                invalid_indices_dict[metric_type] = {}
                            if key not in invalid_indices_dict[metric_type]:
                                invalid_indices_dict[metric_type][key] = []
                            invalid_indices_dict[metric_type][key].append(val_idx + all_offset)

                    # select best result per-frame from "all"
                    if key not in best_metrics_dict[metric_type]:
                        assert all_offset == 0
                        best_metrics_dict[metric_type][key] = []
                    if len(best_metrics_dict[metric_type][key]) < (all_offset + batch_size * n_images_per_batch):
                        best_metrics_dict[metric_type][key].extend([x for x in val_list])
                    else:
                        for val_idx in range(len(val_list)):
                            if metric_type == "psnr" or metric_type == "ssim":
                                if val_list[val_idx] > best_metrics_dict[metric_type][key][all_offset + val_idx]:
                                    best_metrics_dict[metric_type][key][all_offset + val_idx] = val_list[val_idx]
                            else:
                                if val_list[val_idx] < best_metrics_dict[metric_type][key][all_offset + val_idx]:
                                    best_metrics_dict[metric_type][key][all_offset + val_idx] = val_list[val_idx]

                    # sum results per-frame from "all"
                    if key not in avg_metrics_dict[metric_type]:
                        assert all_offset == 0
                        avg_metrics_dict[metric_type][key] = []
                    if len(avg_metrics_dict[metric_type][key]) < (all_offset + batch_size * n_images_per_batch):
                        avg_metrics_dict[metric_type][key].extend([x for x in val_list])
                    else:
                        for val_idx in range(len(val_list)):
                            avg_metrics_dict[metric_type][key][all_offset + val_idx] += val_list[val_idx]

            # calc per-frame average
            for key in avg_metrics_dict[metric_type].keys():
                for val_idx in range(batch_size * n_images_per_batch):
                    avg_metrics_dict[metric_type][key][all_offset + val_idx] /= n_repeat_generation

            # concat together the scores from "from-avg-image"
            avg_scores = mf[metric_type]["from-avg-image"]
            for key, val_list in avg_scores.items():
                if key not in avg_image_metrics_dict[metric_type]:
                    avg_image_metrics_dict[metric_type][key] = []
                val_list = [x for x in val_list if x != float("inf")]  # filter inf values from val_list
                avg_image_metrics_dict[metric_type][key].extend([x for x in val_list])

        # next values from next metrics dict should be written at this offset
        all_offset += batch_size * n_images_per_batch

    # calc average across frames for all dicts
    for dict_key, metrics_dict in zip(["avg", "best", "avg-image"], [avg_metrics_dict, best_metrics_dict, avg_image_metrics_dict]):
        total_dict = {}
        for metric_type, metric_dict in metrics_dict.items():
            out_metric_type = f"0-total-{metric_type}"
            total_dict[out_metric_type] = {}
            for key, val_list in metric_dict.items():
                # filter inf values
                if dict_key == "avg" or dict_key == "best":
                    if metric_type in invalid_indices_dict and key in invalid_indices_dict[metric_type]:
                        print(f"ignore indices for {dict_key}-{metric_type}-{key}: {invalid_indices_dict[metric_type][key]}. values: {[val_list[x] for x in invalid_indices_dict[metric_type][key]]}")
                        val_list = [x for i, x in enumerate(val_list) if i not in invalid_indices_dict[metric_type][key]]

                # calc mean/std
                val_list = np.array(val_list)
                mean_val = val_list.mean()
                std_val = val_list.std()
                total_dict[out_metric_type][key] = {
                    "mean": mean_val,
                    "std": std_val
                }
        metrics_dict = {**total_dict, **metrics_dict}

        # save file
        output_file_path = os.path.join(input_path, f"combined_{dict_key}_metrics.json")
        with open(output_file_path, "w") as f:
            json.dump(metrics_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    tyro.cli(main)
