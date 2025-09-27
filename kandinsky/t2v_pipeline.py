from typing import Union, List

import torch
import torchvision
from torchvision.transforms import ToPILImage

from .generation_utils import generate_sample


class Kandinsky5T2VPipeline:
    def __init__(
        self,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        resolution: int = 512,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf = None,
        offload: bool = False,
    ):
        if resolution not in [512]:
            raise ValueError("Resolution can be only 512")

        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = resolution

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload

        self.RESOLUTIONS = {
            512: [(512, 512), (512, 768), (768, 512)],
        }

    def expand_prompt(self, prompt):
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
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def __call__(
        self,
        text: str,
        time_length: int = 5,  # time in seconds 0 if you want generate image
        width: int = 768,
        height: int = 512,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**63 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        if self.resolution != 512:
            raise NotImplementedError("Only 512 resolution is available for now")

        if (height, width) not in self.RESOLUTIONS[self.resolution]:
            raise ValueError(
                f"Wrong height, width pair. Available (height, width) are: {self.RESOLUTIONS[self.resolution]}"
            )

        # PREPARATION
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        caption = text
        if expand_prompts:
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.expand_prompt(caption)
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        shape = (1, num_frames, height // 8, width // 8, 16)

        # GENERATION
        images = generate_sample(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            text_embedder_device=self.device_map["text_embedder"],
            progress=progress,
            offload=self.offload
        )
        torch.cuda.empty_cache()

        # RESULTS
        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for image in images.squeeze(2).cpu():
                    return_images.append(ToPILImage()(image))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(return_images):
                        for path, image in zip(save_path, return_images):
                            image.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(images):
                        for path, video in zip(save_path, images):
                            torchvision.io.write_video(
                                path,
                                video.float().permute(1, 2, 3, 0).cpu().numpy(),
                                fps=24,
                                options={"crf": "5"},
                            )
                return images
