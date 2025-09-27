import argparse
import time

from kandinsky import get_T2V_pipeline


def validate_args(args):
    size = (args.width, args.height)
    supported_sizes = [(512, 512), (512, 768), (768, 512)]
    if not size in supported_sizes:
        raise NotImplementedError(
            f"Provided size of video is not supported: {size}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video using Kandinsky 5"
    )
    parser.add_argument(
        '--local-rank',
        type=int,
        help='local rank'
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_5s_sft.yaml",
        help="The config file of the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cat in a blue hat",
        help="The prompt to generate video"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        help="Negative prompt for classifier-free guidance"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        choices=[768, 512],
        help="Width of the video in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        choices=[768, 512],
        help="Height of the video in pixels"
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=5,
        help="Duratioin of the video in seconds"
    )
    parser.add_argument(
        "--expand_prompt",
        type=int,
        default=1,
        help="Whether to use prompt expansion."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="The sampling steps number."
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=5.0,
        help="Guidance weight."
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=10.0,
        help="Scheduler scale."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./test.mp4",
        help="Name of the resulting file"
    )

    parser.add_argument(
        "--offload",
        type=bool,
        default=False,
        help="Offload models to save memory or not"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0",
                    "text_embedder": "cuda:0"},
        conf_path=args.config,
        offload=args.offload,
    )

    if args.output_filename is None:
        args.output_filename = "./" + args.prompt.replace(" ", "_") + ".mp4"

    start_time = time.perf_counter()
    x = pipe(args.prompt,
             time_length=args.video_duration,
             width=args.width,
             height=args.height,
             num_steps=args.sample_steps,
             guidance_weight=args.guidance_weight,
             scheduler_scale=args.scheduler_scale,
             expand_prompts=args.expand_prompt,
             save_path=args.output_filename)
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
    print(f"Generated video is saved to {args.output_filename}")
