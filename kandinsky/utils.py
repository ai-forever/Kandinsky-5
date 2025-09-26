import os
from typing import Optional, Union
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
        # the [1.0]*1 is the padding value of first magnitude ratio. 
        dit.mag_ratios = np.array([1.0]*2+[1.0]*98)
        #[0.88556, 0.87688, 0.94313, 0.94916, 1.06431, 1.06496, 1.09917, 1.10019, 1.0463, 1.04599, 1.03523, 1.03574, 1.03616, 1.0356, 1.03347, 1.03495, 1.03811, 1.03887, 1.02237, 1.02189, 1.02875, 1.02939, 1.02677, 1.02752, 1.0264, 1.0278, 1.01866, 1.01876, 1.02226, 1.0225, 1.02018, 1.02086, 1.01971, 1.02083, 1.0224, 1.02355, 1.01953, 1.02023, 1.01971, 1.02072, 1.01781, 1.0188, 1.01881, 1.01945, 1.03258, 1.03438, 1.00541, 1.00518, 1.01771, 1.01763, 1.01897, 1.021, 1.01818, 1.01894, 1.01445, 1.01523, 1.01738, 1.01837, 1.0171, 1.01803, 1.01687, 1.01784, 1.01581, 1.01683, 1.0163, 1.01705, 1.01798, 1.01876, 1.01604, 1.01689, 1.01521, 1.01591, 1.01667, 1.01782, 1.01515, 1.01649, 1.01592, 1.01679, 1.01186, 1.01267, 1.01001, 1.0113, 1.01199, 1.01244, 1.00547, 1.00727, 0.99823, 1.00002, 0.98935, 0.99117, 0.97106, 0.97364, 0.93154, 0.93352, 0.83683, 0.8406])
        # Nearest interpolation when the num_steps is different from the length of mag_ratios
        if len(dit.mag_ratios) != 50 * 2:
            print(f'interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}')
            mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], 50)
            mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], 50)
            interpolated_mag_ratios = np.concatenate(
                [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
            dit.mag_ratios = interpolated_mag_ratios

    # calibration, del before opensource
    # dit.__class__.forward = magcache_calibration
    # dit.cnt = 0
    # dit.num_steps = 50 * 2
    # dit.norm_ratio = [] # mean of magnitude ratio
    # dit.norm_std = [] # std of magnitude ratio
    # dit.cos_dis = [] # cosine distance of residual features
    # dit.residual_cache = [None, None]

    dit.load_state_dict(new_state_dict)
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