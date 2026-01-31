import os
from typing import Optional, Union

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .models.dit import get_dit, TransformerDecoderBlock
from .models.text_embedders import get_text_embedder
from .models.vae import build_vae
from .models.parallelize import parallelize_dit, parallelize_seq
from .i2v_pipeline import Kandinsky5I2VPipeline
from .t2v_pipeline import Kandinsky5T2VPipeline
from .t2i_pipeline import Kandinsky5T2IPipeline
from .i2i_pipeline import Kandinsky5I2IPipeline
from .magcache_utils import set_magcache_params

from PIL import Image
from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True


HF_TOKEN = None


def get_hf_token():
    return HF_TOKEN


def set_hf_token(hf_token):
    global HF_TOKEN
    HF_TOKEN = hf_token


def get_video_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
    mode: str = None,
    model_type: str = "base",  # "base" or "fp8"
    quantized_model_path: str = "/data/kandinsky-5/weights/K5_pro_5s_ao.pt",  # Путь к квантованным весам
):
    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    torch.cuda.set_device(local_rank)

    assert not (world_size > 1 and offload), "Offloading available only with not parallel inference"

    if world_size > 1:
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id=f"kandinskylab/Kandinsky-5.0-{mode.upper()}-Lite-sft-5s",
            allow_patterns="model/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        dit_path = os.path.join(cache_dir, f"model/kandinsky5lite_{mode}_sft_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
            token=get_hf_token()
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
            token=get_hf_token()
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)

    conf.model.dit_params.attention_engine = attention_engine
    conf.model.text_embedder.qwen.mode = mode
    text_embedder = get_text_embedder(conf.model.text_embedder, device='cpu',
        quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)
    
    if not offload: 
        text_embedder = text_embedder.to(device=device_map["text_embedder"]) 

    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"]) 

    dit = get_dit(conf.model.dit_params, text_token_padding=text_token_padding)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg)

    if model_type == "base":
        print(f"Loading BASE model weights from: {conf.model.checkpoint_path}")
        state_dict = load_file(conf.model.checkpoint_path, device='cpu')
        dit.load_state_dict(state_dict, assign=True)
    elif model_type == "fp8":
        print(f"Loading QUANTIZED FP8 Full model weights from: {quantized_model_path}")
        state_dict = torch.load(quantized_model_path)
        dit.load_state_dict(state_dict, assign=True)

    if not offload and world_size == 1:
        dit = dit.to(device_map["dit"])

    if mode == 't2v':
        return Kandinsky5T2VPipeline(
            device_map=device_map,
            dit=dit,
            text_embedder=text_embedder,
            vae=vae,
            local_dit_rank=local_rank,
            world_size=world_size,
            conf=conf,
            offload=offload,
        )
    
    elif mode == 'i2v':
        return Kandinsky5I2VPipeline(
            device_map=device_map,
            dit=dit,
            text_embedder=text_embedder,
            vae=vae,
            local_dit_rank=local_rank,
            world_size=world_size,
            conf=conf,
            offload=offload,
        )


def get_distributed_pipeline(
    pipeline,
    tp_size: int = None,
    mode: str = None,
):
    try:
        world_size = int(os.environ["WORLD_SIZE"])
    except:
        world_size = 1

    if world_size > 1:
        if tp_size is None:
            tp_size = world_size

        if tp_size > 1:
            tp_mesh = init_device_mesh(
                "cuda", (tp_size,), mesh_dim_names=("tensor_parallel",)
            )
        else:
            tp_mesh = None

        dp_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("fsdp",)
        )

    else:
        tp_mesh = None

    pipeline.device_mesh = tp_mesh

    if world_size > 1:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, 
            reduce_dtype=torch.bfloat16, 
            output_dtype=torch.bfloat16
        )

        pipeline.dit = pipeline.dit.to(torch.float32)
        for module in pipeline.dit.modules():
            if isinstance(module, TransformerDecoderBlock):
                fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(pipeline.dit, mesh=dp_mesh, mp_policy=mp_policy)

        pipeline.dit = parallelize_seq(pipeline.dit, tp_mesh, mode)

    return pipeline


def _get_TI2I_params(
    instruct_type: bool,
    model_name: str,
    weights_name: str,
    device_map: Union[str, torch.device, dict],
    resolution: int = 1024,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2IPipeline:
    assert resolution in [1024]

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    assert not (world_size > 1 and offload), "Offloading available only with not parallel inference"

    if world_size > 1:
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id=f"kandinskylab/{model_name}",
            allow_patterns="model/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        dit_path = os.path.join(cache_dir, f"model/{weights_name}")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            allow_patterns="vae/*",
            local_dir=os.path.join(cache_dir, "flux"),
            token=get_hf_token()
        )
        vae_path = os.path.join(cache_dir, "flux", "vae")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
            token=get_hf_token()
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
            token=get_hf_token()
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_ti2i_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path, instruct_type=instruct_type,
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "t2i"
    text_embedder = get_text_embedder(conf.model.text_embedder, device="cpu",
        quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)
    if not offload:
        text_embedder = text_embedder.to( device=device_map["text_embedder"])

    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"])

    dit = get_dit(conf.model.dit_params, text_token_padding=text_token_padding)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg)

    state_dict = load_file(conf.model.checkpoint_path, device='cpu')
    dit.load_state_dict(state_dict, assign=True)

    if not offload and world_size == 1:
        dit = dit.to(device_map["dit"])

    return dict(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
        offload=offload,
    )


def get_image_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 1024,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
    mode: str = None,
):
    if mode == 't2i':
        instruct_type = None
    elif mode == 'i2i':
        instruct_type = 'channel'

    kwargs = _get_TI2I_params(
        instruct_type=instruct_type,
        model_name=f'Kandinsky-5.0-{mode.upper()}-Lite',
        weights_name=f'kandinsky5lite_{mode}.safetensors',
        device_map=device_map,
        resolution=resolution,
        cache_dir=cache_dir,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        text_encoder2_path=text_encoder2_path,
        vae_path=vae_path,
        conf_path=conf_path,
        offload=offload,
        magcache=magcache,
        quantized_qwen=quantized_qwen,
        text_token_padding=text_token_padding,
        attention_engine=attention_engine,
    )

    if mode == 't2i':
        return Kandinsky5T2IPipeline(**kwargs)
    elif mode == 'i2i':
        return Kandinsky5I2IPipeline(**kwargs)


def get_default_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
) -> DictConfig:
    dit_params = {
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 1792,
        "ff_dim": 7168,
        "num_text_blocks": 2,
        "num_visual_blocks": 32,
        "axes_dims": [16, 24, 24],
        "visual_cond": True,
        "in_text_dim": 3584,
        "in_text_dim2": 768,
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "hunyuan",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 256,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 5.0,
        },
        "metrics": {"scale_factor": (1, 2, 2), "resolution": 512,},
    }

    return DictConfig(conf)


def get_default_ti2i_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
    instruct_type=None,
) -> DictConfig:
    dit_params = {
        "instruct_type": instruct_type,
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 2560,
        "ff_dim": 10240,
        "num_text_blocks": 2,
        "num_visual_blocks": 50,
        "axes_dims": [32,48, 48],
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "flux",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 512,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 3.5,
        },
        "metrics": {"scale_factor": (1, 1, 1)},
        "resolution": 512,
    }

    return DictConfig(conf)
