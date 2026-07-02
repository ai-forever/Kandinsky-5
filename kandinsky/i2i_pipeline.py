import json
import logging
import struct
from typing import Union, Optional
import numpy as np

import transformers
import torch
from torchvision.transforms import ToPILImage
from .generation_utils import generate_sample_ti2i
from .models.utils import empty_device_cache
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from peft import PeftConfig, LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


def find_nearest(available_res, real_res):
    nearest_index = np.argmin(
        [
            *map(
                lambda x: abs((x[0] / x[1]) - (real_res[0] / real_res[1])),
                available_res,
            )
        ]
    )
    return available_res[nearest_index]


def read_safetensors_json(file_path):
    """Reads the metadata (JSON header) from a safetensors file."""
    with open(file_path, 'rb') as f:
        # Step 1: Read the first 8 bytes to get the size of the header (N)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('Q', header_size_bytes)[0]  # 'Q' is for unsigned 64-bit integer

        # Step 2: Read the next N bytes which contain the JSON header
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode('utf-8')

        # Step 3: Parse the JSON header
        header = json.loads(header_str)
        return header

class Kandinsky5I2IPipeline:
    RESOLUTIONS = {
        1024: [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)],
    }
    
    def __init__(
        self,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        resolution: int = 1024,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf = None,
        offload: bool = False,
    ):
        if resolution not in [1024]:
            raise ValueError("Resolution can be only 1024") 

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
        self._hf_peft_config_loaded = False
        self.peft_config = {}
        self.peft_triggers = {}
        self.peft_trigger = ""

    def expand_prompt(self, prompt, image):
        width, height = image.size
        image = image.resize((width//4, height//4))
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
                        "image": image,
                    }
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=image,
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
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: float = 3.0,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        image: Optional[Union[Image.Image, str]] = None
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        if self.resolution != 1024:
            raise NotImplementedError("Only 1024 resolution is available for now")
        if isinstance(image,str):
            image = Image.open(image)
        if height is None or width is None:
            assert image is not None, 'set (height, width) or image'
            assert height is None and width is None
            height, width = find_nearest(self.RESOLUTIONS[self.resolution], image.size[::-1])
        else:
            if (height, width) not in self.RESOLUTIONS[self.resolution]:
                raise ValueError(
                    f"Wrong height, width pair. Available (height, width) are: {self.RESOLUTIONS[self.resolution]}"
                )

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.expand_prompt(caption,image=image)
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        if image is not None:
            image = pil_to_tensor(image)[None]

        shape = (1, 1, height // 8, width // 8, 16)

        # GENERATION
        images = generate_sample_ti2i(
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
            offload=self.offload,
            image_vae=True,
            image=image
        )
        empty_device_cache()

        if self.offload:
            self.text_embedder = self.text_embedder.to(device=self.device_map["text_embedder"])

        # RESULTS
        if self.local_dit_rank == 0:
            return_images = []
            for image in images.cpu():
                return_images.append(ToPILImage()(image))
            if save_path is not None:
                if isinstance(save_path, str):
                    save_path = [save_path]
                if len(save_path) == len(return_images):
                    for path, image in zip(save_path, return_images):
                        image.save(path)
            return return_images

    def load_adapter(self, adapter_config: Union[PeftConfig, str], adapter_path: Optional[str] = None,
                     adapter_name: Optional[str] = None, trigger: Optional[str] = None) -> None:
        if adapter_name is None:
            adapter_name = "default"
        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            try:
                with open(adapter_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config = LoraConfig(**adapter_config)
            except:
                raise TypeError(f"adapter_config should be an instance of PeftConfig or a path to a json file.")
        self.peft_config[adapter_name] = adapter_config

        inject_adapter_in_model(adapter_config, self.dit, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        adapter_state_dict = load_file(adapter_path)
        adapter_metadata = read_safetensors_json(adapter_path)
        if trigger is not None:
            self.peft_trigger = trigger
        else:
            if "__metadata__" in adapter_metadata and "trigger" in adapter_metadata["__metadata__"]:
                self.peft_trigger = adapter_metadata["__metadata__"]["trigger"]
            else:
                self.peft_trigger = ""
        self.peft_triggers[adapter_name] = self.peft_trigger

        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = key
            for prefix in ["base_model.model.", "transformer."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break

            new_key = new_key.replace(".default", "")
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(
            self.dit, processed_adapter_state_dict, adapter_name
        )

        if incompatible_keys is not None:
            err_msg = ""
            origin_name = "state_dict"
            # Check for unexpected keys.
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                err_msg = (
                    f"Loading adapter weights from {origin_name} led to unexpected keys not found in the model: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )

            # Check for missing keys.
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                # Filter missing keys specific to the current adapter, as missing base model keys are expected.
                lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                if lora_missing_keys:
                    err_msg += (
                        f"Loading adapter weights from {origin_name} led to missing keys in the model: "
                        f"{', '.join(lora_missing_keys)}"
                    )

            if err_msg:
                logging.warning(err_msg)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[list[str], str]) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.dit.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatibility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
        self.peft_trigger = self.peft_triggers[adapter_name]
    
    def disable_adapters(self) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.dit.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True
        self.peft_trigger = ""
