import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.compress import compress, is_real_quantized
from modelopt.torch.quantization.config import CompressConfig
import modelopt.torch.opt as mto

from kandinsky import get_T2V_pipeline
from kandinsky import get_I2V_pipeline


def quant_mpt(model,mode = str):
    _default_disabled_quantizer_cfg = {}

    config = {
        "quant_cfg": {
            **_default_disabled_quantizer_cfg,

            # Включаем только для всех nn.Linear (weights-only, NVFP4)
            "nn.Linear": {
                "*weight_quantizer": {
                    "num_bits": (2, 1),
                    "block_sizes": {
                        -1: 16,
                        "type": "dynamic",
                        "scale_bits": (4, 3),
                    },
                    "enable": True,
                    "pass_through_bwd": False
                },
                "*input_quantizer": {"enable": False},
            },
            "*visual_embeddings.in_layer*": {"enable": False}, # Отключение квантования для visual_embeddings
        },

        "algorithm": "max",
    }
    # PTQ
    model = mtq.quantize(model, config)

    mtq.print_quant_summary(model)

    ccfg = CompressConfig()
    ccfg.compress = {"default": True}
    compress(model, ccfg)      
    mto.save(model, "K5Pro_nvfp4.pth") # Сохраняем веса

    print("Real-quantized?", is_real_quantized(model))

    return model

if __name__ == "__main__":

    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"},
        conf_path="/kandinsky-5/configs/k5_pro_t2v_5s_sft_sd.yaml",
        model_type="base"
    ) 

    # pipe = get_I2V_pipeline(
    #     device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"},
    #     conf_path="/kandinsky-5/configs/k5_pro_i2v_5s_sft_sd.yaml",
    #     model_type="base"
    # ) 

    pipe.dit = quant_mpt(pipe.dit)





