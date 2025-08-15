import torch
import os
from omegaconf.dictconfig import DictConfig
from ..kandinsky.models.vae import build_vae
from ..kandinsky.models.text_embedders import Kandinsky5TextEmbedder
from ..kandinsky.models.dit import get_dit
from ..kandinsky.generation_utils import generate
import folder_paths
from comfy.comfy_types import ComfyNodeABC
from comfy.utils import ProgressBar as pbar
from safetensors.torch import load_file


class loadTextEmbeddersKandy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "qwen2_5_vl_7b_instruct"}),
                "clip": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "clip_text"})
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_te"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return clip and qwen text embedders"

    def load_te(self, qwen, clip, device="cuda:0"):
        qwen_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],qwen)
        clip_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],clip)
        conf = {'qwen': {'checkpoint_path': qwen_path, 'max_length': 256},
            'clip': {'checkpoint_path': clip_path, 'max_length': 77}
        }
        return (Kandinsky5TextEmbedder(DictConfig(conf), device=device),)
class loadDiTKandy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (folder_paths.get_filename_list("diffusion_models"), ),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_dit"
    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return kandy dit"

    def load_dit(self, dit, device="cuda:0"):
        
        dit_path = folder_paths.get_full_path_or_raise("diffusion_models", dit)
        dit_params = DictConfig({
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
        })
        dit = get_dit(dit_params)
        dit = dit.to(device=device)
        state_dict = load_file(dit_path)
        dit.load_state_dict(state_dict)
        return (dit,)
class KandyTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True})
            },
            "optional": {
                "extended_text": ("PROMPT",),
            },
        }
    RETURN_TYPES = ("CONDITION", "CONDITION")
    RETURN_NAMES = ("TEXT", "POOLED")
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, model, prompt, extended_text=None):
        text = extended_text if extended_text is not None else prompt
        text_embeds = model.embedder([text], type_of_content='video')
        pooled_embed = model.clip_embedder([text])
        return (text_embeds, pooled_embed)

class loadVAEKandy:
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

    def load_vae(self, vae, device="cuda:0"):
        vae_path = os.path.join(folder_paths.get_folder_paths("vae")[0],vae)
        vae = build_vae(DictConfig({'checkpoint_path':vae_path, 'name':'hunyuan'}))
        vae = vae.eval().to(device=device)

        return (vae,)
class expand_prompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True})
            }
        }
    RETURN_TYPES = ("PROMPT","STRING")
    RETURN_NAMES = ("exp_prompt","log")
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("expanded prompt",)
    FUNCTION = "expand_prompt"

    CATEGORY = "conditioning"
    DESCRIPTION = "extend prompt with."
    def expand_prompt(self, model, prompt):
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
        text = model.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.embedder.processor(
            text=[text],
            images=None,
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
        return (output_text[0],str(output_text[0]))
class KandyGenerate(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "width": ("INT", {"default": 768, "min": 512, "max": 768, "tooltip": "width of video."}),
                "height": ("INT", {"default": 512, "min": 512, "max": 768, "tooltip": "height of video."}),
                "lenght": ("INT", {"default": 31, "min": 5, "max": 121, "tooltip": "lenght of video."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "scheduler_scale":("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step":0.1, "round": 0.01, "tooltip": "scheduler scale"}),
                "attention_type":(["full", "sparse"],),
                "positive_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "positive_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "negative_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, steps, width, height, lenght, cfg, positive_emb, positive_clip, negative_emb, negative_clip, scheduler_scale, attention_type):
        bs = 1
        device = 'cuda:0'
        patch_size = (1, 2, 2)
        dim = 16
        height, width = height // 8, width // 8
        bs_text_embed, text_cu_seqlens = positive_emb
        bs_null_text_embed, null_text_cu_seqlens = negative_emb
        progress_bar = pbar(steps)
        text_embed = {"text_embeds": bs_text_embed, "pooled_embed": positive_clip }
        null_embed = {"text_embeds": bs_null_text_embed, "pooled_embed": negative_clip }

        visual_cu_seqlens = lenght * torch.arange(bs + 1, dtype=torch.int32, device=device)
        visual_rope_pos = [
            torch.cat([torch.arange(end) for end in torch.diff(visual_cu_seqlens).cpu()]),
            torch.arange(height // patch_size[1]), torch.arange(width // patch_size[2])
        ]
        text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(text_cu_seqlens).cpu()])
        null_text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(null_text_cu_seqlens).cpu()])
        attention = {
            "type": "flash" if attention_type=='full' else "nabla",
            "causal": False,
            "local": False,
            "glob": False,
            "window": 3,
            "P":0.9,
            "wT":11,
            "wW":3,
            "wH":3,
            "add_sta":True,
            "method": "topcdf"
            }
        dit_params = {
            "patch_size": [1, 2, 2],
            }
        conf = {
            "model": {
                "dit_params": dit_params,
                "attention": attention,
            },
            "metrics": {"scale_factor": (1, 2, 2)},
        }
        conf = DictConfig(conf)
        print(conf)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                latent_visual = generate(
                    model, device, (bs * lenght, height, width, dim), steps, 
                    text_embed, null_embed, 
                    visual_cu_seqlens, text_cu_seqlens, null_text_cu_seqlens,
                    visual_rope_pos, text_rope_pos, null_text_rope_pos,
                    cfg, scheduler_scale, conf 
                )
        return (latent_visual,)

class KandyVAEDecode(ComfyNodeABC):
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
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                bs = 1
                images = latent.reshape(bs, -1, latent.shape[-3], latent.shape[-2], latent.shape[-1])# bs, t, h, w, c
                # shape for decode: bs, c, t, h, w
                images = (images / 0.476986).permute(0, 4, 1, 2, 3)
                images = model.decode(images).sample
                if not isinstance(images, torch.Tensor):
                    images = images.sample
                images = ((images.clamp(-1., 1.) + 1.) * 0.5)#.to(torch.uint8)
        images = images[0].float().permute(1, 2, 3, 0)
        return (images,)

NODE_CLASS_MAPPINGS = {
    "loadTextEmbeddersKandy": loadTextEmbeddersKandy,
    "KandyTextEncode": KandyTextEncode,
    "KandyGenerate": KandyGenerate,
    "loadVAEKandy": loadVAEKandy,
    "KandyVAEDecode": KandyVAEDecode,
    "loadDiTKandy": loadDiTKandy,
    "expand_prompt": expand_prompt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "loadTextEmbeddersKandy": "loadTextEmbeddersKandy",
    "KandyTextEncode": "KandyTextEncode",
    "KandyGenerate": "KandyGenerate",
    "loadVAEKandy": "loadVAEKandy",
    "KandyVAEDecode": "KandyVAEDecode",
    "loadDiTKandy": "loadDiTKandy",
    "expand_prompt": "expand_prompt"

}
