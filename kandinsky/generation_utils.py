import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from torch.distributed import all_gather
from tqdm import tqdm 

from .models.utils import fast_sta_nabla
import torchvision.transforms.functional as F


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params



def build_time_conditions(duration, device, condition_fps=None, latent_time=None, temporal_compression_ratio=4):
    if condition_fps is not None:
        condition_fps = torch.tensor([float(condition_fps)], dtype=torch.float32, device=device)

    if latent_time is not None:
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.flatten().to(device=device, dtype=torch.float32)
        else:
            latent_time = torch.tensor(latent_time, dtype=torch.float32, device=device).flatten()
        if latent_time.numel() != duration:
            raise ValueError(
                f"latent_time must have {duration} values for this sample, got {latent_time.numel()}"
            )
        return condition_fps, latent_time

    if condition_fps is None:
        return None, None

    latent_time = torch.zeros(duration, dtype=torch.float32, device=device)
    if duration > 1:
        latent_time[1:] = (1.0 + temporal_compression_ratio * torch.arange(duration - 1, dtype=torch.float32, device=device)) / float(condition_fps.item())
    return condition_fps, latent_time

def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1,2,3),keepdim=True)
    source_std = source.std(dim=(1,2,3),keepdim=True)
    #magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean
    
    return normalized

def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy
    
    if samples.shape[0] <= 1:
        return (latents, "Only one frame, no normalization needed")
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr:nFr+min(reference_frames, samples.shape[0]-1)]
    
    # print("First frame stats - Mean:", first_frames.mean(dim=(1,2,3)), "Std: ", first_frames.std(dim=(1,2,3)))
    # print(f"Reference frames stats - Mean: {reference_frames_data.mean().item():.4f}, Std: {reference_frames_data.std().item():.4f}")
    
    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)
    
    samples[:nFr] = normalized_first
    
    return samples

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
    condition_fps=None,
    latent_time=None,
):
    with torch._dynamo.utils.disable_cache_limit():
        pred_velocity = dit(
            x,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            condition_fps=condition_fps,
            latent_time=latent_time,
        )
        if abs(guidance_weight - 1.0) > 1e-6:
            uncond_pred_velocity = dit(
                x,
                null_text_embeds["text_embeds"],
                null_text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                null_text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=null_attention_mask,
                condition_fps=condition_fps,
                latent_time=latent_time,
            )
            pred_velocity = uncond_pred_velocity + guidance_weight * (
                pred_velocity - uncond_pred_velocity
            )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    img,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    seed=6554,
    tp_mesh=None,
    attention_mask=None,
    null_attention_mask=None,
    condition_fps=None,
    latent_time=None,
):
    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    if tp_mesh and first_frames is None: # do not split on gpus for i2v
        tp_rank = tp_mesh["tensor_parallel"].get_local_rank()
        tp_world_size = tp_mesh["tensor_parallel"].size()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]

    for timestep, timestep_diff in tqdm(list(zip(timesteps[:-1], torch.diff(timesteps)))):
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            if first_frames is not None:
                first_frames = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                img[:1] = first_frames
                visual_cond_mask[:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
            condition_fps=condition_fps,
            latent_time=latent_time,
        )
        img[..., :pred_velocity.shape[-1]] += timestep_diff * pred_velocity
        # NOTE: remove extra channels that can be added in Image Editing (I2I)
    return img[..., :pred_velocity.shape[-1]]


def resize_video(video, visual_size):
    height, width = video.shape[-2:]
    nearest_height, nearest_width = visual_size

    scale_factor = min(height / nearest_height, width / nearest_width)
    video = F.resize(video, (int(height / scale_factor), int(width / scale_factor)))

    height, width = video.shape[-2:]
    video = F.crop(
        video,
        (height - nearest_height) // 2,
        (width - nearest_width) // 2,
        nearest_height,
        nearest_width,
    )
    return video


def encode_video(data, vae, image_vae): # batch, channels, time, h, w
    if image_vae:
        assert data.shape[2] == 1
        data = vae.encode(data[:, :, 0]).latent_dist.sample()[:, :, None]
    else:
        data = vae.encode(data)[0]
    data *= vae.config.scaling_factor
    return data.permute(0, 2, 3, 4, 1) # batch, time, h, w, channels


def generate_sample(
    shape,
    caption,
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
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
    condition_fps=None,
    latent_time=None,
):
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to('cpu')

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    condition_fps, latent_time = build_time_conditions(
        duration, device, condition_fps=condition_fps, latent_time=latent_time
    )

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)
        
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                condition_fps=condition_fps,
                latent_time=latent_time,
            )
            
    if tp_mesh:
        tensor_list = [
        torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tensor_parallel")
        )
        latent_visual = torch.cat(tensor_list, dim=1)

    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

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

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images

def generate_sample_ti2i(
    shape,
    caption,
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
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    image_vae=False,
    image=None
):
    bs, duration, height, width, dim = shape
    
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
    
    if duration == 1:
        if image is None:
            type_of_content = "image"
        else:
            type_of_content = 'image_edit'
    else:
        type_of_content = "video"

    
    if image is not None:
        image = [resize_video(image, (height * 8, width * 8))]

    if dit.instruct_type == 'channel':
        if image is not None:
            if offload:
                vae.to(vae_device)
            edit_latent = [(i.to(device=vae_device, dtype=torch.bfloat16) / 127.5 - 1.0) for i in image]
            edit_latent = torch.cat([encode_video(i[:,:,None], vae, image_vae).squeeze(0) for i in edit_latent], 0)
            edit_latent = torch.cat([edit_latent, torch.ones_like(img[...,:1])],-1)
            if offload:
                vae.to('cpu')
        else:
            edit_latent = torch.cat([torch.zeros_like(img), torch.zeros_like(img[...,:1])],-1)
        img = torch.cat([img, edit_latent],dim=-1)
    
    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, images=image
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content, images=image
        )

    if offload:
        text_embedder = text_embedder.to('cpu')

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device,dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device,dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    condition_fps, latent_time = build_time_conditions(duration, device, condition_fps=condition_fps, latent_time=latent_time)

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)
        
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )
            
    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

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
            if image_vae:
                images = images[:,:,0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images

def generate_sample_i2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    images,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
    condition_fps=None,
    latent_time=None,
):
    text_embedder.embedder.mode = "i2v"
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to('cpu')

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    condition_fps, latent_time = build_time_conditions(
        duration, device, condition_fps=condition_fps, latent_time=latent_time
    )

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    first_frames = images

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                first_frames,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                condition_fps=condition_fps,
                latent_time=latent_time,
            )

    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device)
            for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tensor_parallel")
        )
        latent_visual = torch.cat(tensor_list, dim=1)

    if images is not None:
        images = images.to(device=latent_visual.device, dtype=latent_visual.dtype)
        latent_visual[:1] = images
    latent_visual = normalize_first_frame(latent_visual)

    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

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

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images
