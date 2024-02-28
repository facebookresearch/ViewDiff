# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import cv2
import numpy as np
import PIL
from pathlib import Path
from PIL import Image
from torchvision import transforms


def main(args):
    """
    Given data that follows the nerfstudio format such as the output from colmap or polycam,
    convert to a format that sdfstudio will ingest
    """
    output_dir = args.output_dir
    input_dir = args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "transforms.json"), "r") as f:
        cam_params = json.load(f)

    # === load camera intrinsics and poses ===
    cam_intrinsics = []
    frames = cam_params["frames"]
    poses = []
    image_paths = []
    mask_paths = []
    # only load images with corresponding pose info
    # currently in random order??, probably need to sort
    for frame in frames:
        # load intrinsics
        cam_intrinsics.append(np.array([
            [frame["fl_x"], 0, frame["cx"]],
            [0, frame["fl_y"], frame["cy"]],
            [0, 0, 1]]))

        # load poses
        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        # IGNORED for now
        c2w = np.array(frame["transform_matrix"]).reshape(4, 4)
        c2w[0:3, 1:3] *= -1
        poses.append(c2w)

        # load images
        file_path = Path(frame["file_path"])
        img_path = os.path.join(input_dir, "images", file_path.name)
        assert os.path.exists(img_path)
        image_paths.append(img_path)

        # load masks
        mask_path = os.path.join(input_dir, "masks", f"{file_path.stem}.png")
        assert os.path.exists(mask_path)
        mask_paths.append(mask_path)

    # Check correctness
    assert len(poses) == len(image_paths)
    assert len(mask_paths) == len(image_paths)
    assert len(poses) == len(cam_intrinsics)

    # Filter invalid poses
    poses = np.array(poses)
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    # === Normalize the scene ===
    if args.scene_type in ["indoor", "object"]:
        # Enlarge bbox by 1.05 for object scene and by 5.0 for indoor scene
        # TODO: Adaptively estimate `scene_scale_mult` based on depth-map or point-cloud prior
        if not args.scene_scale_mult:
            args.scene_scale_mult = 1.05 if args.scene_type == "object" else 5.0
        scene_scale = 2.0 / (np.max(max_vertices - min_vertices) * args.scene_scale_mult)
        scene_center = (min_vertices + max_vertices) / 2.0
        # normalize pose to unit cube
        poses[:, :3, 3] -= scene_center
        poses[:, :3, 3] *= scene_scale
        # calculate scale matrix
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3, 3] -= scene_center
        scale_mat[:3] *= scene_scale
        scale_mat = np.linalg.inv(scale_mat)
    else:
        scene_scale = 1.0
        scale_mat = np.eye(4).astype(np.float32)

    # === Construct the scene box ===
    if args.scene_type == "indoor":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.5,
            "radius": 1.0,
            "collider_type": "box",
        }
    elif args.scene_type == "object":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.6, # 0.05
            "far": 2.0,
            "radius": 1.0,
            "collider_type": "near_far",
        }
    elif args.scene_type == "unbound":
        # TODO: case-by-case near far based on depth prior
        #  such as colmap sparse points or sensor depths
        scene_box = {
            "aabb": [min_vertices.tolist(), max_vertices.tolist()],
            "near": 0.05,
            "far": 2.5 * np.max(max_vertices - min_vertices),
            "radius": np.min(max_vertices - min_vertices) / 2.0,
            "collider_type": "box",
        }
    else:
        raise NotImplementedError("unknown scene_type", args.scene_type)

    # === Resize the images and intrinsics ===
    # Only resize the images when we want to use mono prior
    with open(image_paths[0], "rb") as f:
        array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        sample_img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        h, w, _ = sample_img.shape
    if args.mono_prior:
        # get smallest side to generate square crop
        target_crop = min(h, w)
        tar_h = tar_w = 384 * args.crop_mult
        rgb_trans = transforms.Compose(
            [
                transforms.CenterCrop(target_crop),
                transforms.Resize((tar_h, tar_w), interpolation=PIL.Image.BILINEAR)
            ]
        )
        mask_trans = transforms.Compose(
            [
                transforms.CenterCrop(target_crop),
                transforms.Resize((tar_h, tar_w), interpolation=PIL.Image.NEAREST)
            ]
        )

        # Update camera intrinsics
        offset_x = (w - target_crop) * 0.5
        offset_y = (h - target_crop) * 0.5
        resize_factor = tar_h / target_crop
        for intrinsics in cam_intrinsics:
            # center crop by min_dim
            intrinsics[0, 2] -= offset_x
            intrinsics[1, 2] -= offset_y
            # resize from min_dim x min_dim -> to 384 x 384
            intrinsics[:2, :] *= resize_factor

    # Do nothing if we don't want to use mono prior
    else:
        tar_h, tar_w = h, w
        rgb_trans = transforms.Compose([])
        mask_trans = transforms.Compose([])

    # === Construct the frames in the meta_data.json ===
    frames = []
    out_index = 0
    for idx, (valid, pose, image_path) in enumerate(zip(valid_poses, poses, image_paths)):
        if not valid:
            continue

        # save rgb image
        out_img_name = f"{out_index:06d}_rgb.png"
        out_img_path = os.path.join(output_dir, out_img_name)
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img_tensor = rgb_trans(img)
            with open(out_img_path, "wb") as f2:
                img_tensor.save(f2)

        frame = {
            "rgb_path": out_img_name,
            "camtoworld": pose.tolist(),
            "intrinsics": cam_intrinsics[idx].tolist()
        }

        # load mask
        mask_path = mask_paths[idx]
        out_mask_name = f"{out_index:06d}_foreground_mask.png"
        out_mask_path = os.path.join(output_dir, out_mask_name)
        with open(mask_path, "rb") as f:
            mask_PIL = Image.open(f)
            new_mask = mask_trans(mask_PIL)
            with open(out_mask_path, "wb") as f2:
                new_mask.save(f2)
        frame["foreground_mask"] = out_mask_name

        if args.mono_prior:
            frame["mono_depth_path"] = out_img_name.replace("_rgb.png", "_depth.npy")
            frame["mono_normal_path"] = out_img_name.replace("_rgb.png", "_normal.npy")

        frames.append(frame)
        out_index += 1

    # === Construct and export the metadata ===
    meta_data = {
        "camera_model": "OPENCV",
        "height": tar_h,
        "width": tar_w,
        "has_mono_prior": args.mono_prior,
        "has_sensor_depth": False,
        "has_foreground_mask": True,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
        "frames": frames,
    }
    with open(os.path.join(output_dir, "meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    # === Generate mono priors using omnidata ===
    if args.mono_prior:
        assert os.path.exists(args.pretrained_models), "Pretrained model path not found"
        assert os.path.exists(args.omnidata_path), "omnidata l path not found"
        # generate mono depth and normal
        print("Generating mono depth...")
        os.system(
            f"python extract_monocular_cues.py \
            --omnidata_path {args.omnidata_path} \
            --pretrained_model {args.pretrained_models} \
            --img_path {output_dir} --output_path {output_dir} \
            --task depth"
        )
        print("Generating mono normal...")
        os.system(
            f"python extract_monocular_cues.py \
            --omnidata_path {args.omnidata_path} \
            --pretrained_model {args.pretrained_models} \
            --img_path {output_dir} --output_path {output_dir} \
            --task normal"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess nerfstudio dataset to sdfstudio dataset, "
                                                 "currently support colmap and polycam")

    parser.add_argument("--data", dest="input_dir", required=True, help="path to nerfstudio data directory")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="path to output data directory")
    parser.add_argument("--scene-type", dest="scene_type", required=True, choices=["indoor", "object", "unbound"],
                        help="The scene will be normalized into a unit sphere when selecting indoor or object.")
    parser.add_argument("--scene-scale-mult", dest="scene_scale_mult", type=float, default=None,
                        help="The bounding box of the scene is firstly calculated by the camera positions, "
                             "then multiply with scene_scale_mult")
    parser.add_argument("--mono-prior", dest="mono_prior", action="store_true",
                        help="Whether to generate mono-prior depths and normals. "
                             "If enabled, the images will be cropped to 384*384")
    parser.add_argument("--crop-mult", dest="crop_mult", type=int, default=1,
                        help="image size will be resized to crop_mult*384, only take effect when enabling mono-prior")
    parser.add_argument("--omnidata-path", dest="omnidata_path",
                        default="<YOUR_DIR>/omnidata/omnidata_tools/torch",
                        help="path to omnidata model")
    parser.add_argument("--pretrained-models", dest="pretrained_models",
                        default="<YOUR_DIR>/omnidata_tools/torch/pretrained_models/",
                        help="path to pretrained models")

    args = parser.parse_args()

    main(args)
