#!/usr/bin/env python3
"""
combine_planner_videos.py

Combines planner simulation videos side-by-side into a single MP4.

Each "round" (e.g. _1, _2, ...) is rendered with all planners side-by-side.
If one planner's video is shorter, its last frame is held until all are done.
Rounds are then concatenated in order into a final output video.

Usage:
    python combine_planner_videos.py planner1.mp4 planner2.mp4 [planner3.mp4 ...] -o output.mp4

    python save_sidebyside_videos.py "Unimodal Rew. Data (Class. Guid.).mp4" "Multimodal Rew. Data (Class. Guid.).mp4" "Pedestrian Data (Class. Guid.).mp4" -o output.mp4

The script infers the number of test runs automatically from the files present
on disk (looks for planner1_1.mp4, planner1_2.mp4, etc.).

Optional arguments:
    -o / --output   Output filename (default: combined_output.mp4)
    -n / --num      Force number of rounds instead of auto-detecting
    --width         Width of each planner tile in pixels (default: 640)
    --height        Height of each planner tile in pixels (default: 480)
    --fps           Output frames per second (default: taken from first video)
    --label         Add planner name labels to each tile (flag, default: on)
    --no-label      Disable labels
"""

import argparse
import os
import subprocess
import sys
import tempfile
import re


def run(cmd, check=True):
    """Run a shell command, printing it first."""
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def get_video_info(path):
    """Return (duration_seconds, fps, width, height) for a video file."""
    result = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "csv=p=0",
        path
    ])
    parts = result.stdout.strip().split(",")
    width = int(parts[0])
    height = int(parts[1])
    # r_frame_rate is like "30/1" or "60000/1001"
    num, den = parts[2].split("/")
    fps = float(num) / float(den)
    duration = float(parts[3])
    return duration, fps, width, height


def detect_num_rounds(base_paths):
    """Auto-detect the number of rounds by checking which _N files exist."""
    max_round = 0
    for base in base_paths:
        b, ext = os.path.splitext(base)
        i = 1
        while os.path.exists(f"{b}_{i}{ext}"):
            max_round = max(max_round, i)
            i += 1
    return max_round


def build_side_by_side(video_paths, output_path, tile_w, tile_h, fps, add_labels, planner_names, tmp_dir):
    """
    Combine multiple videos side-by-side for a single round.
    Shorter videos have their last frame held.
    """
    n = len(video_paths)

    # Get max duration across all videos in this round
    durations = []
    for vp in video_paths:
        dur, _, _, _ = get_video_info(vp)
        durations.append(dur)
    max_dur = max(durations)

    # Build ffmpeg filter_complex
    # Each input: scale to tile size, pad/extend to max_dur by looping last frame
    inputs = []
    for vp in video_paths:
        inputs += ["-i", vp]

    filter_parts = []
    stream_labels = []

    for idx in range(n):
        dur = durations[idx]
        # Scale and pad each video to tile_w x tile_h
        scale = f"[{idx}:v]scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black"

        if add_labels:
            name = planner_names[idx]
            # Escape special chars for drawtext
            name_safe = name.replace("'", "\\'").replace(":", "\\:")
            label = f",drawtext=text='{name_safe}':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5"
        else:
            label = ""

        # Hold last frame if this video is shorter than max_dur
        if dur < max_dur - 0.05:
            tpad = f",tpad=stop_mode=clone:stop_duration={max_dur - dur:.4f}"
        else:
            tpad = ""

        filter_parts.append(f"{scale}{label}{tpad}[v{idx}]")
        stream_labels.append(f"[v{idx}]")

    # hstack all streams
    hstack = f"{''.join(stream_labels)}hstack=inputs={n}[out]"
    filter_parts.append(hstack)

    filter_complex = ";".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    )
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Combine planner sim videos side-by-side.")
    parser.add_argument("base_videos", nargs="+",
                        help="Base video filenames, e.g. planner1.mp4 planner2.mp4. "
                             "The script looks for planner1_1.mp4, planner1_2.mp4, etc.")
    parser.add_argument("-o", "--output", default="combined_output.mp4",
                        help="Output filename (default: combined_output.mp4)")
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="Number of rounds (auto-detected if not set)")
    parser.add_argument("--width", type=int, default=640,
                        help="Width of each tile in pixels (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Height of each tile in pixels (default: 480)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output FPS (default: inferred from first video)")
    parser.add_argument("--no-label", dest="label", action="store_false",
                        help="Disable planner name labels")
    parser.set_defaults(label=True)
    args = parser.parse_args()

    base_videos = args.base_videos

    # Derive planner names from filenames (strip extension)
    planner_names = [os.path.splitext(os.path.basename(v))[0] for v in base_videos]

    # Auto-detect number of rounds
    num_rounds = args.num if args.num else detect_num_rounds(base_videos)
    if num_rounds == 0:
        print("ERROR: Could not find any numbered video files (e.g. planner1_1.mp4).")
        print("Make sure the numbered files exist alongside the base names you provided.")
        sys.exit(1)

    print(f"Found {num_rounds} round(s) across {len(base_videos)} planner(s).")

    # Infer FPS from first available video
    fps = args.fps
    if fps is None:
        first_vid = f"{os.path.splitext(base_videos[0])[0]}_1{os.path.splitext(base_videos[0])[1]}"
        _, fps, _, _ = get_video_info(first_vid)
        print(f"Using FPS={fps:.3f} from {first_vid}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        round_files = []

        for i in range(1, num_rounds + 1):
            print(f"\n--- Processing round {i}/{num_rounds} ---")
            round_videos = []
            missing = []

            for base in base_videos:
                b, ext = os.path.splitext(base)
                vpath = f"{b}_{i}{ext}"
                if os.path.exists(vpath):
                    round_videos.append(vpath)
                else:
                    missing.append(vpath)

            if missing:
                print(f"  WARNING: Missing files for round {i}: {missing}")
                print("  Skipping this round.")
                continue

            round_out = os.path.join(tmp_dir, f"round_{i:04d}.mp4")
            build_side_by_side(
                round_videos, round_out,
                args.width, args.height, fps,
                args.label, planner_names, tmp_dir
            )
            round_files.append(round_out)

        if not round_files:
            print("ERROR: No rounds were successfully processed.")
            sys.exit(1)

        # Concatenate all rounds
        print(f"\n--- Concatenating {len(round_files)} round(s) into {args.output} ---")
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for rf in round_files:
                f.write(f"file '{rf}'\n")

        run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            args.output
        ])

    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()