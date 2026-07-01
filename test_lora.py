import argparse
import time
import warnings
import logging

import torch

import os
from huggingface_hub import snapshot_download

from kandinsky.utils import set_hf_token
from kandinsky import get_video_pipeline, get_image_pipeline, get_distributed_pipeline


def validate_args(args):
    size = (args.width, args.height)
    if "i2i" in args.config:
        return
    elif "t2i" in args.config:
        supported_sizes = [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)]
    else:
        supported_sizes = [(512, 512), (512, 768), (768, 512), (1280, 768), (768, 1280), (1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)]
    if not size in supported_sizes:
        raise NotImplementedError(
            f"Provided size of video is not supported: {size}")


def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False
    )


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
        default="./configs/k5_lite_t2v_5s_sft_sd.yaml",
        help="The config file of the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The bear plays balalaika.",
        help="The prompt to generate video"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./assets/test_image.jpg",
        help="An image to generate image/video from."
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
        choices=[512, 640, 768, 896, 1152, 1024, 1280],
        help="Width of the video in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        choices=[512, 640, 768, 896, 1152, 1024, 1280],
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
        default=None,
        help="The sampling steps number."
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=None,
        help="Guidance weight."
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=5.0,
        help="Scheduler scale."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name of the resulting file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1137,
        help="Seed for the random number generator"
    )

    parser.add_argument(
        "--offload",
        action='store_true',
        default=False,
        help="Offload models to save memory or not"
    )
    parser.add_argument(
        "--magcache",
        action='store_true',
        default=False,
        help="Using MagCache (for 50 steps models only)"
    )
    parser.add_argument(
        "--qwen_quantization",
        action='store_true',
        default=False,
        help="Use quantized Qwen2.5-VL model (4-bit quantization)"
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        help="Name of the full attention algorithm to use for <=5 second generation",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"]
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="token to download restricted models like FLUX.1-dev VAE",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="GPUs num for tensor parallel",
    )
    parser.add_argument(
        "--lora_repo_id",
        type=str,
        default="kandinskylab/Kandinsky-5.0-I2V-Pro-LoRa-Microwave-right",
        help="lora repo id (on huggingface)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on, e.g. cuda:0, mps, cpu",
    )
    args = parser.parse_args()

    if args.hf_token:
        set_hf_token(args.hf_token)

    return args


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_generation_mode(config):
    mode = None 

    if "t2i" in config:
        mode = "t2i"
    elif "i2i" in config:
        mode = "i2i"
    elif "i2v" in config:
        mode = "i2v"
    elif "t2v" in config:
        mode = "t2v"

    assert mode is not None, "Generation mode is not found"
    return mode


if __name__ == "__main__":
    disable_warnings()
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    device_map = {"dit": args.device, "vae": args.device,
                  "text_embedder": args.device}
    mode = get_generation_mode(args.config)

    if mode == "t2i" or mode == "i2i":
        pipe = get_image_pipeline(
            device_map=device_map,
            resolution=1024,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
            mode=mode,
        )
    elif mode == "i2v" or mode == "t2v":
        pipe = get_video_pipeline(
            device_map=device_map,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
            mode=mode,
        )

    if args.output_filename is None:
        args.output_filename = "./" + args.prompt.replace(" ", "_")
        if len(args.output_filename) > 32:
            args.output_filename = args.output_filename[:32]
        if mode == "t2i" or mode == "i2i":
            args.output_filename = args.output_filename + ".png"
        else:
            args.output_filename = args.output_filename + ".mp4"

    # load lora adapter
    adapter_path = snapshot_download(
        repo_id=args.lora_repo_id,
        local_dir=os.path.join('./weights', args.lora_repo_id),
        token=None
    )
    pipe.load_adapter(
        adapter_config=os.path.join('./weights', args.lora_repo_id, "config_lora.json"),
        adapter_path= os.path.join('./weights', args.lora_repo_id, "lora.safetensors"),
    )

    if mode == "i2v" or mode == "t2v":
        pipe = get_distributed_pipeline(pipe, args.tp_size, mode=mode)

    start_time = time.perf_counter()
    if "t2i" in args.config:
        x = pipe(args.prompt,
                 width=args.width,
                 height=args.height,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    elif "i2i" in args.config:
        x = pipe(args.prompt,
                 image=args.image,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    elif "i2v" in args.config:
        x = pipe(args.prompt,
                 image=args.image,
                 time_length=args.video_duration,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    else:
        x = pipe(args.prompt,
                 time_length=args.video_duration,
                 width=args.width,
                 height=args.height,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
    print(f"Generated file is saved to {args.output_filename}")
