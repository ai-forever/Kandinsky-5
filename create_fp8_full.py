import torch

from torchao.quantization import (
    quantize_,
    ModuleFqnToConfig,
    Float8DynamicActivationFloat8WeightConfig,
)

from kandinsky import get_video_pipeline


def build_dit_fp8_config() -> ModuleFqnToConfig:
    base_cfg = Float8DynamicActivationFloat8WeightConfig()

    # Всё остальное квантуем base_cfg
    module_cfg = {"_default": base_cfg}


    disabled_fqns = [
        "time_embeddings.in_layer",
        "time_embeddings.out_layer",
        "text_embeddings.in_layer",
        "visual_embeddings.in_layer",
        "out_layer.modulation.out_layer",
        "out_layer.out_layer",
    ]

    for i in range(4): 

        disabled_fqns.append(
            f"text_transformer_blocks.{i}.feed_forward.out_layer"
        )

    for i in range(60):
        disabled_fqns.append(
            f"visual_transformer_blocks.{i}.feed_forward.out_layer"
        )

    for fqn in set(disabled_fqns): 
        module_cfg[fqn] = None

    return ModuleFqnToConfig(module_cfg)


if __name__ == "__main__":

    pipe = get_video_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"},
        conf_path="/data/kandinsky-5/configs/k5_pro_t2v_5s_sft_sd.yaml",
        model_type="base",
        mode="t2v"
    ) 

    ao_cfg = build_dit_fp8_config()
    quantize_(pipe.dit,  ao_cfg)
    torch.save(pipe.dit.state_dict(), "/data/kandinsky-5/weights/K5_pro_5s_ao.pt")


