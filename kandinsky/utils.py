import os
from typing import Optional, Union
import numpy as np

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .models.dit import get_dit
from .models.text_embedders import get_text_embedder
from .models.vae import build_vae
from .models.parallelize import parallelize_dit
from .t2v_pipeline import Kandinsky5T2VPipeline
from .magcache_utils import magcache_forward, magcache_calibration, nearest_interp

from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True

def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 512,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
) -> Kandinsky5T2VPipeline:
    assert resolution in [512]

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id="ai-forever/kandinsky-5",
            allow_patterns="model/*",
            local_dir=cache_dir,
        )
        dit_path = os.path.join(cache_dir, "model/lite_sft_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)

    text_embedder = get_text_embedder(conf.model.text_embedder).to(
        device=device_map["text_embedder"]
    )
    vae = build_vae(conf.model.vae)
    vae = vae.eval().to(device=device_map["vae"])

    dit = get_dit(conf.model.dit_params)
    state_dict = load_file(conf.model.checkpoint_path)
    # UPD state dict
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if 'out_layer' in key:
            if 'cross_attention' in key:
                new_key = key.replace('cross_attention.out_layer', 'out_layer_cross')
            elif 'self_attention' in key:
                new_key = key.replace('self_attention.out_layer', 'out_layer_self')
        new_state_dict[new_key] = state_dict[key]
    del state_dict

    # MAGCACHE
    if magcache:
        print(f'using Magcache')
        dit.__class__.forward = magcache_forward
        dit.cnt = 0
        dit.num_steps = 50 * 2
        dit.magcache_thresh = 0.12
        dit.K = 2
        dit.accumulated_err = [0.0, 0.0]
        dit.accumulated_steps = [0, 0]
        dit.accumulated_ratio = [1.0, 1.0]
        dit.retention_ratio = 0.2
        dit.residual_cache = [None, None]
        dit.mag_ratios = np.array([1.0]*100)
        # Nearest interpolation when the num_steps is different from the length of mag_ratios
        if len(dit.mag_ratios) != 50 * 2:
            print(f'interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}')
            mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], 50)
            mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], 50)
            interpolated_mag_ratios = np.concatenate(
                [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
            dit.mag_ratios = interpolated_mag_ratios

    dit.load_state_dict(new_state_dict, assign=True)
    dit = dit.to(device_map["dit"])

    if world_size > 1:
        dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    return Kandinsky5T2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
    )


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
        },
        "metrics": {"scale_factor": (1, 2, 2)},
        "resolution": 512,
    }

    return DictConfig(conf)