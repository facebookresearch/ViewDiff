# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import ffmpeg


def main(args):
    ffmpeg_bin = "/usr/bin/ffmpeg"
    (
        ffmpeg.input(
            f"{args.image_folder}/{args.file_name_pattern_glob}", pattern_type="glob", framerate=args.framerate
        )
        .output(args.output_path, **{"codec:v": "libx264", "crf": 19}, loglevel="quiet")
        .run(cmd=ffmpeg_bin, overwrite_output=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL CONFIG
    group = parser.add_argument_group("general")
    group.add_argument("--image_folder", required=True)
    group.add_argument("--file_name_pattern_glob", required=False, default="*.png")
    group.add_argument("--framerate", required=False, type=int, default=15)
    group.add_argument("--output_path", required=False, type=str, default="video.mp4")

    args = parser.parse_args()

    main(args)
