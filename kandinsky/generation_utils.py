import numpy as np
import torch
from tqdm import tqdm 

from .models.utils import create_doc_causal_mask, fast_doc_causal_sta


def get_sparse_params(conf, batch_embeds, cu_seqlens, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, C = batch_embeds["visual"].shape
    visual_cu_seqlens = cu_seqlens["visual_rope"].to(device=device)
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "flex":
        P = 8
        block_mask, _ = fast_doc_causal_sta(
            T,
            H,
            W,
            P,
            visual_cu_seqlens,
            conf.model.attention.causal,
            sta_local=conf.model.attention.local,
            sta_global=conf.model.attention.glob,
            return_mask=True,
            window_size=conf.model.attention.window,
        )
        block_mask = block_mask.to(device=device)
        torch_mask = None
        sparse_params = {
            "block_mask": block_mask,
            "torch_mask": torch_mask,
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
        }
    elif conf.model.attention.type == "torch":
        torch_mask = create_doc_causal_mask(
            T, H, W, seq=visual_cu_seqlens, causal=conf.model.attention.causal
        )
        torch_mask = torch_mask.to(dtype=torch.bool, device=device)
        block_mask = None
        sparse_params = {
            "block_mask": block_mask,
            "torch_mask": torch_mask,
            "attention_type": conf.model.attention.type,
            "to_fractal": False,
        }
    elif conf.model.attention.type == "nabla":
        sparse_params = {
            "block_mask": None,
            "torch_mask": None,
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "visual_seqlens": visual_cu_seqlens,
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params


@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_cu_seqlens,
    text_cu_seqlens,
    null_text_cu_seqlens,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    block_mask=None,
    torch_mask=None,
    sparse_params=None,
):
    pred_velocity = dit(
        x,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        t * 1000,
        visual_cu_seqlens,
        text_cu_seqlens,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=conf.metrics.scale_factor,
        sparse_params=sparse_params,
    )
    if abs(guidance_weight - 1.0) > 1e-6:
        uncond_pred_velocity = dit(
            x,
            null_text_embeds["text_embeds"],
            null_text_embeds["pooled_embed"],
            t * 1000,
            visual_cu_seqlens,
            null_text_cu_seqlens,
            visual_rope_pos,
            null_text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
        )
        pred_velocity = uncond_pred_velocity + guidance_weight * (
            pred_velocity - uncond_pred_velocity
        )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    shape,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_cu_seqlens,
    text_cu_seqlens,
    null_text_cu_seqlens,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    conf,
    progress=False,
    seed=6554,
):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(*shape, device=device, generator=g)

    sparse_params = get_sparse_params(
        conf, {"visual": img}, {"visual_rope": visual_cu_seqlens}, device
    )
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for timestep, timestep_diff in tqdm(list(zip(timesteps[:-1], torch.diff(timesteps)))):
        time = timestep.unsqueeze(0).repeat(visual_cu_seqlens.shape[0] - 1)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_cu_seqlens,
            text_cu_seqlens,
            null_text_cu_seqlens,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            block_mask=None,
            torch_mask=None,
            sparse_params=sparse_params,
        )
        img = img + timestep_diff * pred_velocity
    return img


def generate_sample(
    shape,
    captions,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
):
    bs, duration, height, width, dim = shape
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens = text_embedder.encode(
            captions, type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens = text_embedder.encode(
            [negative_caption] * len(captions), type_of_content=type_of_content
        )

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)

    visual_cu_seqlens = duration * torch.arange(
        bs + 1, dtype=torch.int32, device=device
    )
    visual_rope_pos = [
        torch.cat([torch.arange(end) for end in torch.diff(visual_cu_seqlens).cpu()]),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.cat(
        [torch.arange(end) for end in torch.diff(text_cu_seqlens).cpu()]
    )
    null_text_rope_pos = torch.cat(
        [torch.arange(end) for end in torch.diff(null_text_cu_seqlens).cpu()]
    )

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                (bs * duration, height, width, dim),
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_cu_seqlens,
                text_cu_seqlens,
                null_text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                conf,
                seed=seed,
                progress=progress,
            )
    torch.cuda.empty_cache()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    torch.cuda.empty_cache()

    return images
