import torch
import os
from omegaconf.dictconfig import DictConfig
from ..kandinsky.models.vae import build_vae, AutoencoderKLHunyuanVideo
from ..kandinsky.models.text_embedders import Kandinsky5TextEmbedder
from ..kandinsky.models.dit import get_dit
from ..kandinsky.generation_utils import generate, resize_video
from ..kandinsky.i2v_pipeline import resize_image
from ..kandinsky.i2i_pipeline import find_nearest, Kandinsky5I2IPipeline
import folder_paths
from comfy.comfy_types import ComfyNodeABC
from comfy.utils import ProgressBar as pbar
from safetensors.torch import load_file
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms


class Kandinsky5LoadTextEmbedders:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "qwen2_5_vl_7b_instruct"}),
                "clip": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "clip_text"}),
                "qwen_quantized": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "gen_type": ("ENUM", {"choices": ["t2i", "i2i", "t2v", "i2v"], "default": "t2v"})
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_te"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return clip and qwen text embedders"

    def load_te(self, qwen, clip, qwen_quantized, gen_type=None):
        qwen_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],qwen)
        clip_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],clip)
        max_length = 512 if gen_type == "i2i" else 256
        conf = {'qwen': {'checkpoint_path': qwen_path, 'max_length': max_length},
            'clip': {'checkpoint_path': clip_path, 'max_length': 77},
        }
        return (Kandinsky5TextEmbedder(DictConfig(conf), device='cuda:0',quantized_qwen=qwen_quantized),)
class Kandinsky5LoadDiT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (folder_paths.get_filename_list("diffusion_models"), ),
            }
        }
    RETURN_TYPES = ("MODEL","CONFIG", "ENUM")
    RETURN_NAMES = ("model","conf", "gen_type")
    FUNCTION = "load_dit"
    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return kandy dit"
    def parse_cfg_name(self, model_name):
        name_parts = model_name.split(".")[0].split(sep="_")
        name_parts[0] = name_parts[0].replace("kandinsky5", "k5_")
        version, type = name_parts[:2]
        if type == 'i2i':
            return 'k5_lite_i2i_sft_hd.yaml'
        if type == 'i2v':
            sec = name_parts[-1]
            return f'{version}_{type}_{sec}_sft_sd.yaml'
        if type == 't2v':
            sec = name_parts[-1]
            return f'{version}_{type}_{sec}_sft_sd.yaml'
        if type == 't2i':
            return "k5_lite_t2i_sft_hd.yaml"

    def load_dit(self, dit):
        dit_path = folder_paths.get_full_path_or_raise("diffusion_models", dit)
        current_file = Path(__file__)
        parent_directory = current_file.parent.parent
        conf_name = self.parse_cfg_name(dit)
        conf = OmegaConf.load(os.path.join(parent_directory, "configs", conf_name))
        dit = get_dit(conf.model.dit_params)
        state_dict = load_file(dit_path)
        dit.load_state_dict(state_dict)
        gen_type = conf_name.split("_")[2]
        assert gen_type in ['i2v', 't2v', 'i2i', 't2i']
        return (dit,conf, gen_type)
class Kandinsky5TextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "gen_type": ("ENUM",),
            },
            "optional": {
                "image": ("IMAGE",),
                "extended_text": ("PROMPT",),
            },
        }
    RETURN_TYPES = ("CONDITION", "CONDITION")
    RETURN_NAMES = ("TEXT", "POOLED")
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, model, prompt, gen_type, image=None, extended_text=None):
        text = extended_text if extended_text is not None else prompt
        device='cuda:0'
        model = model.to(device)
        if gen_type == 'i2i':
            type_of_content = 'image_edit'
        elif gen_type == 't2i':
            type_of_content = 'image'
        else:
            type_of_content = "video"

        if image is not None:
            image = torch.clamp(image * 255.0, 0, 255).to(torch.uint8)
            images = [image.permute(0,3,1,2),]
        else:
            images = None

        text_embeds = model.embedder([text], type_of_content=type_of_content, images=images)
        pooled_embed = model.clip_embedder([text])
        model = model.to('cpu')
        return (text_embeds, pooled_embed)

class Kandinsky5LoadVAE(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (os.listdir(folder_paths.get_folder_paths("vae")[0]), {"default": "hunyuan_vae"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_vae"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return vae"

    def load_vae(self, vae):
        vae_path = os.path.join(folder_paths.get_folder_paths("vae")[0],vae)
        name = 'flux' if 'flux' in vae_path else 'hunyuan'
        vae = build_vae(DictConfig({'checkpoint_path':vae_path, 'name':name}))
        vae = vae.eval()

        return (vae,)
class expand_prompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "gen_type": ("ENUM", {"choices": ["t2i", "i2i", "t2v", "i2v"], "default": "t2v"})
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("PROMPT","STRING")
    RETURN_NAMES = ("exp_prompt","log")
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("expanded prompt",)
    FUNCTION = "expand_prompt"

    CATEGORY = "conditioning"
    DESCRIPTION = "extend prompt with."
    def expand_prompt(self, model, prompt, gen_type, image=None, device='cuda:0'):
        if image is not None:
            to_pil = transforms.ToPILImage()
            image = image.squeeze().permute(2,0,1)
            pil_image = to_pil(image).convert("RGB")

        if gen_type in ['i2v', 't2v']:
            if image is not None:
                print(image.shape)
                print("i2v expander")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": pil_image
                            },
                            {
                                "type": "text",
                                "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
Here are some example descriptions from the dataset that the model was trained:
1. "Create a video showing a nighttime urban driving scene from inside a car. The driver is focused on the road ahead, with the city lights visible through the windshield. The GPS device on the dashboard continues to display navigation information. The camera remains steady, capturing the interior of the car and the changing street view outside as the vehicle moves forward. The background shifts slightly to show different parts of the cityscape, including illuminated buildings and street signs."
2. "Create a video where the character, dressed in historical attire, is seen holding an umbrella with a logo. The character should move closer to the camera while maintaining a steady pace, keeping the umbrella raised. The background remains consistent with a foggy, outdoor setting, but the focus shifts more towards the character as they approach. The lighting should emphasize the details of the costume and the umbrella, enhancing the dramatic effect."
3. "Darken the scene while keeping the characters and setting unchanged, emphasizing a serious atmosphere."
IImportantly! These are just examples from a large training dataset of 20 mln videos.
Rewrite Prompt: "{prompt}" to get high-quality image to video generation from this image. Pay main attention to information about changes of objects.
Make prompt dynamic. Answer only with expanded prompt..""",
                            },
                        ],
                    }
                ]
            else:
                print("t2v expander")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
Here are some example descriptions from the dataset that the model was trained:
1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
IImportantly! These are just examples from a large training dataset of 200 million videos.
Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt.""",
                            },
                        ],
                    }
                ]
        elif gen_type in ['i2i', 't2i']:
            if image is not None:
                print("i2i expander")
                print(image.shape)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Rewrite and enhance the original editing instruction with richer detail, clearer structure, and improved descriptive quality. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Explain what needs to be changed and what needs to be left unchanged. Explain in details how to change  camera potision or tell that camera position shouldn't be changed.
example:
Original text: add text 911 and 'Police'
Result: Add the word "911" in large blue letters to the hood. Below that, add the word "POLICE." Keep the camera position unchanged, as do the background, car position, and lighting.
Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
                            },
                            {
                                "type": "image",
                                "image": pil_image,
                            }
                        ],
                    }
                ]
            else:
                print("t2i expander")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Rewrite and enhance the original prompt with richer detail, clearer structure, and improved descriptive quality. Expand the scene, atmosphere, and context while preserving the user’s intent. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Strengthen visual clarity, style, and specificity, but do not change the meaning. Output only the enhanced prompt, written in polished, vivid language suitable for high-quality image generation.
example:
Original text: white mini police car with blue stripes, with 911 and 'Police' text
Result: A miniature model car simulating the official transport of the relevant authorities. The body is white with blue stripes. The word "911" is written in large blue letters on the hood and side. Below it, "POLICE" is used in a font. The windows are transparent, and the interior has black seats. The headlights have plastic lenses, and the roof has blue and red beacons. The radiator grille has vertical slots. The wheels are black with white rims. The doors are closed, the windows have black frames. The background is uniform white.
Here 911 in double quotes because it is text on image, 'Police' -> "POLICE" because it should be in double quotes and capital letters.
Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
                            },
                        ],
                    }
                ]

        model = model.to(device)
        text = model.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.embedder.processor(
            text=[text],
            images=None if image is None else [pil_image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.embedder.model.device)
        generated_ids = model.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = model.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(output_text[0])
        model = model.to('cpu')
        return (output_text[0],str(output_text[0]))
class Kandinsky5Generate(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "config": ("CONFIG", {"tooltip": "Config of model and generation."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "width": ("INT", {"default": 768, "min": 512, "max": 1024, "tooltip": "width of video."}),
                "height": ("INT", {"default": 512, "min": 512, "max": 1024, "tooltip": "height of video."}),
                "length": ("INT", {"default": 121, "min": 1, "max": 241, "tooltip": "lenght of video."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "scheduler_scale":("FLOAT", {"default": 10.0, "min": 1.0, "max": 25.0, "step":0.1, "round": 0.01, "tooltip": "scheduler scale"}),
                "precision": (["float16", "bfloat16"], {"default": "bfloat16"}),
                "positive_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "positive_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "negative_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "gen_type": ("ENUM", {"choices": ["t2i", "i2i", "t2v", "i2v"], "default": "t2v"})
            },
            "optional": {
                "image_latent": ("LATENT",),
                "seed": ("INT",{"default":5})
            },
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, config, steps, width, height, length, cfg, precision, positive_emb, positive_clip, negative_emb, negative_clip, scheduler_scale, gen_type, image_latent=None, seed=5):
        bs = 1
        device = 'cuda:0'
        model = model.to(device)
        patch_size = (1, 2, 2)
        autocast_type = torch.bfloat16 if precision=='bfloat16' else torch.float16 
        dim = config.model.dit_params.in_visual_dim
        if gen_type in ['i2i', 't2i']:
            print("Forcing generation length to 1 (image generation)")
            length = 1

        if image_latent is not None:
            length, height, width = 1 + (length - 1)//4, image_latent.shape[1], image_latent.shape[2]
        else:
            length, height, width = 1 + (length - 1)//4, height // 8, width // 8
        bs_text_embed, text_cu_seqlens, attention_mask = positive_emb
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = negative_emb
        text_embed = {"text_embeds": bs_text_embed, "pooled_embed": positive_clip }
        null_embed = {"text_embeds": bs_null_text_embed, "pooled_embed": negative_clip }

        visual_rope_pos = [
            torch.arange(length // patch_size[0]),
            torch.arange(height // patch_size[1]),
            torch.arange(width // patch_size[2])
        ]
        text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(text_cu_seqlens).cpu()])
        null_text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(null_text_cu_seqlens).cpu()])

        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        img = torch.randn(bs * length, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
        if getattr(config.model.dit_params, 'instruct_type', None) == 'channel':
            if image_latent is not None:
                image_latent = torch.cat([image_latent, torch.ones_like(img[...,:1], device=device)],-1)
            else:
                image_latent = torch.cat([torch.zeros_like(img, device=device), torch.zeros_like(img[...,:1], device=device)],-1)
            img = torch.cat([img, image_latent],dim=-1)

        first_frames = image_latent if gen_type == 'i2v' else None
    
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=autocast_type):
                latent_visual = generate(
                    model, device, img, steps, 
                    text_embed, null_embed,
                    visual_rope_pos, text_rope_pos, null_text_rope_pos,
                    cfg, scheduler_scale, 
                    first_frames, 
                    config,
                    attention_mask=attention_mask,
                    null_attention_mask=null_attention_mask
                )
                if gen_type == 'i2v' and image_latent is not None:
                    image_latent = image_latent.to(device=latent_visual.device, dtype=latent_visual.dtype)
                    latent_visual[:1] = image_latent
        model = model.to('cpu')
        return (latent_visual,)

class Kandinsky5VAEDecode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "vae."}),
                "latent": ("LATENT", {"tooltip": "latent."}),}
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"
    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, model, latent):
        device = 'cuda:0'
        model = model.to(device)
        autocast_type = torch.float16 if isinstance(model, AutoencoderKLHunyuanVideo) else torch.bfloat16

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=autocast_type):
                bs = 1
                images = latent.reshape(bs, -1, latent.shape[-3], latent.shape[-2], latent.shape[-1])# bs, t, h, w, c
                # shape for decode: bs, c, t, h, w
                images = (images / 0.476986).permute(0, 4, 1, 2, 3)
                if not isinstance(model, AutoencoderKLHunyuanVideo):
                    images = images.squeeze(2)
                images = model.decode(images).sample
                if not isinstance(images, torch.Tensor):
                    images = images.sample
                if not isinstance(model, AutoencoderKLHunyuanVideo):
                    images = images.unsqueeze(2)
                images = ((images.clamp(-1., 1.) + 1.) * 0.5)#.to(torch.uint8)
        images = images[0].float().permute(1, 2, 3, 0)
        model = model.to('cpu')
        return (images,)

class Kandinsky5VAEImageEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "vae."}),
                "image": ("IMAGE", {"tooltip": "image."}), 
                "config": ("CONFIG", {"tooltip": "Config of model and generation"})},
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The encoded latent.",)
    FUNCTION = "encode"
    CATEGORY = "latent"
    DESCRIPTION = "Encodes image to latent."

    def encode(self, model, image, config):
        device = 'cuda:0'
        model = model.to(device)
        if getattr(config.model.dit_params, 'instruct_type', None) != 'channel':
            divisibility = 1024 if config.model.attention.type == "nabla" else 16
            image, k = resize_image(image.permute(0,3,1,2), max_area=512*768, divisibility=divisibility)
        else:
            image = image.permute(0,3,1,2)
        image = image * 2.0 - 1.

        with torch.no_grad():
            if isinstance(model, AutoencoderKLHunyuanVideo):
                image = image.to(device=device, dtype=torch.float16).unsqueeze(0).transpose(1,2)
                lat_image = model.encode(image, opt_tiling=False).latent_dist.sample()
            else:
                image = image.to(device=device, dtype=torch.bfloat16).unsqueeze(0).transpose(1,2)
                lat_image = model.encode(image[:, :, 0]).latent_dist.sample()[:, :, None]
            lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0)
            lat_image = lat_image * 0.476986

        model = model.to('cpu')
        return (lat_image,)
    

class Kandinsky5I2IResize(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "image."}), },
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The resized image.",)
    FUNCTION = "resize"
    CATEGORY = "image"
    DESCRIPTION = "Resizes image to size required by Image2Image model."

    def resize(self,image):
        height, width = find_nearest(Kandinsky5I2IPipeline.RESOLUTIONS[1024], image.shape[1:3])
        image = resize_video(image.permute(0,3,1,2), (height, width))
        image = image.permute(0,2,3,1)
        return (image,)

NODE_CLASS_MAPPINGS = {
    "Kandinsky5LoadTextEmbedders": Kandinsky5LoadTextEmbedders,
    "Kandinsky5TextEncode": Kandinsky5TextEncode,
    "Kandinsky5Generate": Kandinsky5Generate,
    "Kandinsky5LoadVAE": Kandinsky5LoadVAE,
    "Kandinsky5VAEDecode": Kandinsky5VAEDecode,
    "Kandinsky5VAEImageEncode": Kandinsky5VAEImageEncode,
    "Kandinsky5LoadDiT": Kandinsky5LoadDiT,
    "Kandinsky5I2IResize": Kandinsky5I2IResize,
    "expand_prompt": expand_prompt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky5LoadTextEmbedders": "Kandinsky5LoadTextEmbedders",
    "Kandinsky5TextEncode": "Kandinsky5TextEncode",
    "Kandinsky5Generate": "Kandinsky5Generate",
    "Kandinsky5LoadVAE": "Kandinsky5LoadVAE",
    "Kandinsky5VAEDecode": "Kandinsky5VAEDecode",
    "Kandinsky5VAEImageEncode": "Kandinsky5VAEImageEncode",
    "Kandinsky5LoadDiT": "Kandinsky5LoadDiT",
    "expand_prompt": "expand_prompt",
    "Kandinsky5I2IResize": "Kandinsky5I2IResize"
}
