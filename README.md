<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a href="https://habr.com/ru/companies/sberbank/articles/951800/">Habr</a> | <a href="https://kandinskylab.ai/">Project Page</a> | <a href="https://arxiv.org/abs/2511.14993">Technical Report</a> | 🤗 <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite> Video Lite </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-pro> Video Pro </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite> Image Lite </a> | <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky5"> 🤗 Diffusers </a>  | <a href="https://github.com/kandinskylab/kandinsky-5/blob/main/comfyui/README.md">ComfyUI</a>
</div>

<h1>Kandinsky 5.0: A family of diffusion models for Video & Image generation</h1>

In this repository, we provide a family of diffusion models to generate a video or an image given a textual prompt and/or image.



https://github.com/user-attachments/assets/b06f56de-1b05-4def-a611-1a3159ed71b0



## Project Updates
- 🔥 ```2026/08/07```: [Kandinsky Time Adapter](https://github.com/kandinskylab/kandinsky-5/blob/time_adapter/TIME_ADAPTER_README.md) and [Kandinsky WM 1.0](https://github.com/kandinskylab/kandinsky-wm) are open-sourced.
- 🔥 ```2025/12/12```: Kandinsky 5.0 Video Pro is ranked as Top-1 open-source Text-to-Video model at [LMArena](https://lmarena.ai/leaderboard/text-to-video).
- 🔥 ```2025/12/03```: our Kandinsky 5.0 Video Pro and Image Lite models are [accepted to diffusers](https://github.com/huggingface/diffusers/commit/d0c54e5563c3245b57d2b374e8e334da77305c05)
- 🔥 ```2025/11/26```: our simplified LoRa training is available on [kandinsky-5-lora-train](https://github.com/kandinskylab/kandinsky-5-lora-train)
- 🔥 ```2025/11/24```: LoRas for Camera control is open-sourced: [Lite LoRAs](https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite-loras) and [Pro LoRAs](https://huggingface.co/collections/kandinskylab/kandinsky-50-video-pro-loras). Inference code is avaibale in `examples/inference_examples_i2v_lora.ipynb` and `examples/inference_examples_t2v_lora.ipynb`
- 🔥 ```2025/11/20```: `Kandinsky 5.0 Video Pro` is open-sourced. T2V & I2V models are available.
- 🔥 ```2025/11/15```: `Kandinsky 5.0 Lite I2V` & `Kandinsky 5.0 Lite T2I` models are open-sourced.
- 🔥 ```2025/10/19```: Further VAE tiling optimization. NF4 version of Qwen2.5-VL from Bitsandbytes is supported. Flash Attention 2, Flash Attention 2, Sage Attention or SDPA can be selected for 5-seconds generation using option --attention_engine. Now generation should work on the GPUS with 12 GB of memory. Kandinsky 5 Video Lite is [accepted to diffusers](https://github.com/huggingface/diffusers/pull/12478).
- 🔥 ```2025/10/7```: The ComfyUI README file has been updated. SDPA support has been added, allowing you to run our code without Flash attention. Magcache support for nocfg checkpoints has been added, allowing Magcache support for sft and nocfg checkpoints. Memory consumption in the VAE has been reduced, with the entire pipeline now running at 24 GB with offloading.
- 🔥 ```2025/09/29```: We have open-sourced `Kandinsky 5.0 T2V Lite` a lite (2B parameters) version of `Kandinsky 5.0 Video` text-to-video generation model. Released checkpoints: `kandinsky5lite_t2v_pretrain_5s`, `kandinsky5lite_t2v_pretrain_10s`, `kandinsky5lite_t2v_sft_5s`, `kandinsky5lite_t2v_sft_10s`, `kandinsky5lite_t2v_nocfg_5s`, `kandinsky5lite_t2v_nocfg_10s`, `kandinsky5lite_t2v_distilled16steps_5s`, `kandinsky5lite_t2v_distilled16steps_10s` contains weight from pretrain, supervised finetuning, cfg distillation and diffusion distillation into 16 steps. 5s checkpoints are capable of generating videos up to 5 seconds long. 10s checkpoints is faster models checkpoints trained with [NABLA](https://huggingface.co/ai-forever/Wan2.1-T2V-14B-NABLA-0.7) algorithm and capable to generate videos up to 10 seconds long.

## Community Works

If your research or project builds upon Kandinsky 5, and you would like more people to see it, please inform us.

- [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner) now supports Kandinsky 5.0 LoRA and even full-rank training.
- [CacheDiT](https://github.com/vipshop/cache-dit/tree/main) offers Fully Cache Acceleration support for Kandinsky-5 with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_kandinsky5_t2v.py) for more details.

## Table of Contents
1. [Kandinsky 5.0 Video Pro](#kandinsky-50-video-pro)
2. [Kandinsky 5.0 Video Lite](#kandinsky-50-video-lite)
3. [Kandinsky 5.0 Image Lite](#kandinsky-50-image-lite)
4. [Kandinsky 5.0 Image Editing](#kandinsky-50-image-editing)
5. [Kandinsky Time Adapter](#kandinsky-time-adapter)
6. [Kandinsky WM 1.0](#kandinsky-wm-10)
7. [Quickstart & Run examples](#quickstart)


## Kandinsky 5.0 Video Pro

Kandinsky 5.0 Video Pro is a line-up of 19B models that generates high-quality HD videos from English and Russian prompts with controllable camera motion.

We provide several Text-to-Video model variants, each optimized for different use cases:

* SFT model — delivers the highest generation quality;

* Pretrain model — designed for fine-tuning by researchers and enthusiasts.

All models are available in two versions: for generating 5-second and 10-second videos.

Additionally, we provide Image-to-Video model capable to generate video given input image and text prompt.

### Pipeline

**Latent diffusion pipeline** with **Flow Matching**.

**Diffusion Transformer (DiT)** as the main generative backbone with **cross-attention to text embeddings**.

- **Qwen2.5-VL** and **CLIP** provides text embeddings.

- **HunyuanVideo 3D VAE** encodes/decodes video into a latent space.

- **DiT** is the main generative module using cross-attention to condition on text.

<img width="1600" height="477" alt="Picture1" src="https://github.com/user-attachments/assets/17fc2eb5-05e3-4591-9ec6-0f6e1ca397b3" />

<img width="800" height="406" alt="Picture2" src="https://github.com/user-attachments/assets/f3006742-e261-4c39-b7dc-e39330be9a09" />

### Model Zoo

| Model                               | config | video duration | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|----------------|-----|------------|----------------|
| Kandinsky 5.0 T2V Pro SFT 5s HD       | configs/k5_pro_t2v_5s_sft_hd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s) |      1241     |
| Kandinsky 5.0 T2V Pro SFT 10s HD     |configs/k5_pro_t2v_10s_sft_hd.yaml| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-10s) |      -     |
| Kandinsky 5.0 T2V Pro SFT 5s SD       | configs/k5_pro_t2v_5s_sft_sd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s) |      560     |
| Kandinsky 5.0 T2V Pro SFT 10s SD     |configs/k5_pro_t2v_10s_sft_sd.yaml| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-10s) |      1158     |
| Kandinsky 5.0 T2V Pro pretrain 5s HD     |-| 5s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-5s) |      1241     |
| Kandinsky 5.0 T2V Pro pretrain 10s HD     |-| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-10s) |      -     |
| Kandinsky 5.0 T2V Pro pretrain 5s SD     |-| 5s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-5s) |      560     |
| Kandinsky 5.0 T2V Pro pretrain 10s SD     |-| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-10s) |      1158     |
| Kandinsky 5.0 I2V Pro HD 5s       | configs/k5_pro_i2v_5s_sft_hd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s) |      -     |
| Kandinsky 5.0 I2V Pro SD 5s       | configs/k5_pro_i2v_5s_sft_sd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s) |      -     |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

<table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/918cd953-7777-4f6f-bc98-e3f42f045cb1" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5ed4eed7-5f4c-4b05-8886-a62131efea75" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/299f810b-d9b9-4bf9-8ec5-af30762879a4" width=100 controls autoplay loop></video>
      </td>
     
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/6946e0e8-3088-4584-a4df-162bb24c4548" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5aab3a8d-6447-43b5-b78b-862b1f0ce6f7" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/118eeeb8-c33c-4799-bc89-a5430417c771" width=100 controls autoplay loop></video>
      </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/fbfeeab1-2d79-468d-9fbd-4a944b1d541e" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9fb24941-ff42-467b-b4e0-601c6833acaa" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/540dafda-cb0b-4b17-ac00-3c3b4ae0794c" width=100 controls autoplay loop></video>
      </td>
  </tr>

</table>

### Results:

#### Side-by-Side evaluation

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/73e5ff00-2735-40fd-8f01-767de9181918" /></img>
      </td>
      <td>
         <img width="200" alt="image" src="https://github.com/user-attachments/assets/f449a9e7-74b7-481d-82da-02723e396acd" /></img>
      </td>

  <tr>
      <td>
          Comparison with Veo 3 
      </td>
      <td>
          Comparison with Veo 3 fast
      </td>
  <tr>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/a6902fb6-b5e8-4093-adad-aa4caab79c6d" /></img>
      </td>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/09986015-3d07-4de8-b942-c145039b9b2d" /></img>
      </td>
  <tr>
      <td>
          Comparison with Wan 2.2 A14B Text-to-Video mode
      </td>
      <td>
          Comparison with Wan 2.2 A14B Image-to-Video mode
      </td>

</table>

## Kandinsky 5.0 Video Lite

Kandinsky 5.0 T2V Lite is a lightweight video generation model (2B parameters) that ranks #1 among open-source models in its class. It outperforms larger Wan models (5B and 14B) and offers the best understanding of Russian concepts in the open-source ecosystem.

We provide 8 model variants, each optimized for different use cases:

* SFT model — delivers the highest generation quality;

* CFG-distilled — runs 2× faster;

* Diffusion-distilled — enables low-latency generation with minimal quality loss (6× faster);

* Pretrain model — designed for fine-tuning by researchers and enthusiasts.

All models are available in two versions: for generating 5-second and 10-second videos.

Additionally, we provide Image-to-Video model capable to generate video given input image and text prompt.


### Model Zoo

| Model                               | config | video duration | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|----------------|-----|------------|----------------|
| Kandinsky 5.0 T2V Lite SFT 5s       |configs/k5_lite_t2v_5s_sft_sd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s) |      139 s     |
| Kandinsky 5.0 T2V Lite SFT 10s      |configs/k5_lite_t2v_10s_sft_sd.yaml| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s) |      224 s     |
| Kandinsky 5.0 T2V Lite pretrain 5s  |configs/k5_lite_t2v_5s_pretrain_sd.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s) |      139 s      |
| Kandinsky 5.0 T2V Lite pretrain 10s |configs/k5_lite_t2v_10s_pretrain_sd.yaml | 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s) |     224 s      |
| Kandinsky 5.0 T2V Lite no-CFG 5s    |configs/k5_lite_t2v_5s_nocfg_sd.yaml| 5s             | 50  |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s) |       77 s     |
| Kandinsky 5.0 T2V Lite no-CFG 10s   |configs/k5_lite_t2v_10s_nocfg_sd.yaml| 10s            | 50  |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-10s) |     124 s      |
| Kandinsky 5.0 T2V Lite distill 5s   |configs/k5_lite_t2v_5s_distil_sd.yaml| 5s             | 16  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s)|       35 s     |
| Kandinsky 5.0 T2V Lite distill 10s  |configs/k5_lite_t2v_10s_distil_sd.yaml| 10s            | 16  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s)|      61 s      |
| Kandinsky 5.0 I2V Lite 5s  |configs/k5_lite_i2v_5s_sft_sd.yaml| 5s            | 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s)|      139 s      |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

#### Kandinsky 5.0 T2V Lite SFT

<table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bc38821b-f9f1-46db-885f-1f70464669eb" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9f64c940-4df8-4c51-bd81-a05de8e70fc3" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/77dd417f-e0bf-42bd-8d80-daffcd054add" width=100 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/385a0076-f01c-4663-aa46-6ce50352b9ed" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7c1bcb31-cc7d-4385-9a33-2b0cc28393dd" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/990a8a0b-2df1-4bbc-b2e3-2859b6f1eea6" width=100 controls autoplay loop></video>
      </td>
  </tr>

</table>


#### Kandinsky 5.0 T2V Lite Distill

<table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/861342f9-f576-4083-8a3b-94570a970d58" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/302e4e7d-781d-4a58-9b10-8c473d469c4b" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3e70175c-40e5-4aec-b506-38006fe91a76" width=100 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b7da85f7-8b62-4d46-9460-7f0e505de810" width=100 controls autoplay loop></video>
      </td>

</table>


### Results:

#### Side-by-Side evaluation

The evaluation is based on the expanded prompts from the [Movie Gen benchmark](https://github.com/facebookresearch/MovieGenBench), which are available in the expanded_prompt column of the benchmark/moviegen_bench.csv file.

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_sora.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_14B.jpg" width=400 ></img>
      </td>
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_5B.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_A14B.jpg" width=400 ></img>
      </td>
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_1.3B.jpg" width=400 ></img>
      </td>

</table>

#### Distill Side-by-Side evaluation

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_5s_vs_kandinsky_5_video_lite_distill_5s.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_10s_vs_kandinsky_5_video_lite_distill_10s.jpg" width=400 ></img>
      </td>

</table>


## Kandinsky 5.0 Image Lite

Kandinsky 5.0 Image Lite is a line-up of 6B image generation models with the following capabilities:

* 1K resulution (1280x768, 1024x1024 and others).

* High visual quality

* Strong text-writing

* Russian concepts understanding


### Model Zoo

| Model                               | config | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|-----|------------|----------------|
| Kandinsky 5.0 T2I Lite  |configs/k5_lite_t2i_sft_hd.yaml| 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite)|      13 s      |
| Kandinsky 5.0 T2I Lite pretrain  |-| 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain)|      13 s      |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <image src="https://github.com/user-attachments/assets/f46e6866-15ce-445d-bb81-9843a341e2a9" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/74f3af1f-b11e-4174-9f36-e956b871a6e6" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/7e469d09-8b96-4691-b929-dd809827adf9" width=200 ></image>
      </td>
  <tr>
</table>
<table border="0" style="width: 200; text-align: left; margin-top: 10px;">
      <td>
          <image src="https://github.com/user-attachments/assets/8054b25b-5d71-4547-8822-b07d71d137f4" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/f4825237-640b-4b2d-86e6-fd08fe95039f" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/73fbbc2a-3249-4b70-8931-2893ab0107a5" width=200 ></image>
      </td>

</table>
<table border="0" style="width: 200; text-align: left; margin-top: 10px;">
      <td>
          <image src="https://github.com/user-attachments/assets/c309650b-8d8b-4e44-bb63-48287e22ff44" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/d5c0fcca-69b7-4d77-9c36-cd2fb87f2615" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/7895c3e8-2e72-40b8-8bf7-dcac859a6b29" width=200 ></image>
      </td>

</table>

### Results


### Results:

#### Side-by-Side evaluation

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img width="200" src="https://github.com/user-attachments/assets/d5f984e6-f847-49bd-b961-b3f27c141c56" /></img>
      </td>
      <td>
          <img width="200" src="https://github.com/user-attachments/assets/c34dbf24-6a14-4b0f-9b59-c6300dc21c7c" /></img>
      </td>
  <tr>
      <td>
          Comparison with FLUX.1 dev
      </td>
      <td>
          Comparison with Qwen-Image
      </td>

</table>



## Kandinsky 5.0 Image Editing

Kandinsky 5.0 Image Editing is a line-up of 6B image editing models with the following capabilities:

- 1K resulution (1280x768, 1024x1024 and others).

- High visual quality

- Strong text-writing

- Russian concepts understanding

### Model Zoo

| Model                               | config | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|-----|------------|----------------|
| Kandinsky 5.0 T2I Editing  |configs/k5_lite_i2i_sft_hd.yaml| 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite) |  -  |
| Kandinsky 5.0 T2I Editing pretrain  |-| 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain) |  -  |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img width="400" alt="image" src="https://github.com/user-attachments/assets/027bdeaf-2bed-4a00-9d6a-77a706100ed8" /></image>
      </td>
      <td>
         <img width="400" alt="image" src="https://github.com/user-attachments/assets/6b8c059c-e65d-4560-88e7-4543c56d7a3f" /></image>
      </td>
      
  <tr>
      <td>
          Change this to a cowboy hat.
      </td>
      <td>
          Turn this into a neon sign hanging
on a brick wall in a cool modern office.
      </td>
  </tr>
  <tr>
      <td>
          <img width="400" alt="image" src="https://github.com/user-attachments/assets/b579d635-1710-453e-954c-12f76748dafc" /></image>
      </td>
      <td>
          <img width="400"  alt="image" src="https://github.com/user-attachments/assets/9074e1c7-28aa-405d-9eca-38dfa6f7e6c9" /></image>
      </td>
  <tr>
      <td>
         Swap your sweatshirt for a se-
quined evening dress, add some bright jewelry,
and brighten your lips and eyes. Keep the angle. 
      </td>
      <td>
         Turn this into a real photograph of
the same dog.
      </td> 
  </tr>
</table>



### Results:

#### Side-by-Side evaluation

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img width="200"  alt="image" src="https://github.com/user-attachments/assets/a8f30810-00c2-4dbf-97ae-3135ca81f961" /></img>
      </td>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/21534266-4511-40e2-a306-e30c12bbf26c" /></img>
      </td>
  <tr>
      <td>
          Comparison with FLUX.1 Kontext [dev]
      </td>
      <td>
          Comparison with Qwen-Image-Edit-2509
      </td>
</table>


## Kandinsky Time Adapter

[Kandinsky Time Adapter](https://github.com/kandinskylab/kandinsky-5/blob/time_adapter/TIME_ADAPTER_README.md) adds FPS-conditioned motion control to Kandinsky 5.0 I2V Lite. The implementation and instructions are available in the separate `time_adapter` branch.

## Kandinsky WM 1.0

[Kandinsky WM 1.0](https://github.com/kandinskylab/kandinsky-wm) is a family of image-to-video models for Physical AI, covering autonomous driving, robotics, and general physics. The models and instructions are available in a separate repository.

## Quickstart

#### Installation
Clone the repo:
```sh
git clone https://github.com/kandinskylab/kandinsky-5.git
cd kandinsky-5
```

Install dependencies:
```sh
pip install -r requirements.txt
```

To improve inference performance on NVidia Hopper GPUs, we recommend installing [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/?tab=readme-ov-file#flashattention-3-beta-release).

#### Model Download
```sh
python download_models.py
```
use `models` argument to download some specific models, otherwise all models will be downloaded

example to download only `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s` and `kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s`:
```sh
python download_models.py --models kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s,kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s
```

#### Run Kandinsky 5.0 T2V Lite SFT 5s

```sh
python test.py --prompt "A dog in red hat"
```

#### Run Kandinsky 5.0 T2V Lite SFT 10s 

```sh
python test.py --config ./configs/k5_lite_t2v_10s_sft_sd.yaml --prompt "A dog in red hat" --video_duration 10 
```


#### Run Kandinsky 5.0 I2V Lite 5s

```sh
python test.py --config ./configs/k5_lite_i2v_5s_sft_sd.yaml --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --video_duration 5
```

#### Run Kandinsky 5.0 T2I Lite

```sh
python test.py --config ./configs/k5_lite_t2i_sft_hd.yaml --prompt "A dog in a red hat" --width=1280 --height=768
```

### T2V Inference

```python
import torch
from kandinsky import get_T2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2V_pipeline(device_map, conf_path="configs/k5_lite_t2v_5s_sft_sd.yaml")

images = pipe(
    seed=42,
    time_length=5,
    width=768,
    height=512,
    save_path="./test.mp4",
    text="A cat in a red hat",
)
```

### I2V Inference

```python
import torch
from kandinsky import get_I2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_I2V_pipeline(device_map, conf_path="configs/k5_lite_i2v_5s_sft_sd.yaml")

images = pipe(
    seed=42,
    time_length=5,
    save_path='./test.mp4',
    text="The bear plays balalaika.",
    image = "assets/test_image.jpg",
)
```

### T2I Inference

```python
import torch
from kandinsky import get_T2I_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2I_pipeline(device_map, conf_path="configs/k5_lite_t2i_sft_hd.yaml")

images = pipe(
    seed=42,
    save_path='./test.png',
    text="A cat in a red hat with a label 'HELLO'"
)
```


### I2I Inference


```python
import torch
from kandinsky import get_I2I_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_I2I_pipeline(
    resolution=1024, offload=True,
    device_map=device_map,
)
out = pipe(
    "Replace the cat with a husky, leave the rest unchanged",
    image='./assets/cat_in_hat.png'
)

```


Please, refer to [examples](examples) folder for more examples in various notebooks.

### Distributed Inference

For a faster inference, we also provide the capability to perform inference in a distributed way:
```
NUMBER_OF_NODES=1
NUMBER_OF_DEVICES_PER_NODE=1 / 2 / 4
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py
```

### Optimized Inference

#### Offloading
For less memory consumption you can use **offloading** of the models.
```sh
python test.py --prompt "A dog in red hat" --offload
```

#### Magcache
Also we provide [Magcache](https://github.com/Zehong-Ma/MagCache) inference for faster generations (now available for sft 5s and sft 10s checkpoints).

```sh
python test.py --prompt "A dog in red hat" --magcache
```

#### Qwen encoder quantization
To reduce GPU memory needed for Qwen encoder we provide option to use NF4-quantized version from [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

```sh
python test.py --prompt "A dog in red hat" --qwen_quantization
```

#### Attention engine selection
Depending on your hardware you can use the follwing full attention algorithm implementation:
* PyTorch [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
* [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
* [Sage Attention](https://github.com/thu-ml/SageAttention)

The attention algorithm can be selected using an option "--attention_engine" of test.py script for 5 second (and less) video generation. For 10-second generation we use sparse attention algorithm [NABLA](https://arxiv.org/abs/2507.13546).

Note that currently (19 Oct. 2025) version build from source contains a bug and produces noisy output. A temporary workaround to fix it is decribed [here](https://github.com/thu-ml/SageAttention/issues/277).

```sh
python test.py --prompt "A dog in red hat" --attention_engine=flash_attention_3
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=flash_attention_2
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=sdpa
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=sage
```

By default we use option --attention_engine=auto which enables automatic selection of the most optimal algorithm installed in your system.

### ComfyUI

See the instruction [here](comfyui)


### Beta testing
You can apply to participate in the beta testing of the Kandinsky Video Lite via the [telegram bot](https://t.me/kandinsky_access_bot).

## 📑 Todo List

- [ ] Kandinsky 5.0 Video Pro
  - [ ] Checkpoints
      - [x] sft
      - [x] pretrain
      - [ ] rl
      - [ ] distil 16 steps
      - [x] I2V
  - [x] ComfyUI integration
  - [ ] Diffusers integration
  - [x] Caching acceleration support
  - [x] Multi-GPU Inference code of the models
- [ ] Kandinsky 5.0 Video Lite
  - [ ] Checkpoints
      - [x] sft
      - [x] pretrain
      - [ ] rl
      - [x] cfg distil 
      - [x] distil 16 steps
      - [ ] autoregressive generation
      - [x] I2V
  - [x] ComfyUI integration
  - [x] Diffusers integration
  - [x] Caching acceleration support
  - [x] Multi-GPU Inference code of the models
- [ ] Kandinsky 5.0 Image Lite
  - [x] Checkpoints
      - [x] rl
      - [x] pretrain
  - [x] ComfyUI integration
  - [ ] Diffusers integration
  - [x] Caching acceleration support
  - [x] Multi-GPU Inference code of the models
- [ ] Kandinsky 5.0 Image Editing
  - [x] Checkpoints
      - [x] sft
      - [x] pretrain
  - [ ] ComfyUI integration
  - [ ] Diffusers integration
  - [x] Multi-GPU Inference code of the models
- [ ] Technical report


# Authors


<B>Core Contributors</B>:
- <B>Video</B>: Alexey Letunovskiy, Maria Kovaleva, Lev Novitskiy, Denis Koposov, Dmitrii
Mikhailov, Anastasiia Kargapoltseva, Anna Dmitrienko, Anastasia Maltseva
- <B>Image & Editing</B>: Nikolai Vaulin, Nikita Kiselev, Alexander Varlamov
- <B>Pre-training Data</B>: Ivan Kirillov, Andrey Shutkin, Nikolai Vaulin, Ilya Vasiliev
- <B>Post-training Data</B>: Julia Agafonova, Anna Averchenkova, Olga Kim
- <B>Research Consolidation & Paper</B>: Viacheslav Vasilev, Vladimir Polovnikov
  
<B>Contributors</B>: Yury Kolabushin, Kirill Chernyshev, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Kormilitsyn Semen, Tatiana Nikulina, Olga Vdovchenko, Polina Mikhailova, Polina
Gavrilova, Nikita Osterov, Bulat Akhmatov

<B>Track Leaders</B>: Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis
Parkhomenko

<B>Project Supervisor</B>: Denis Dimitrov


# Citation

```
@misc{arkhipkin2025kandinsky50familyfoundation,
      title={Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation}, 
      author={Vladimir Arkhipkin and Vladimir Korviakov and Nikolai Gerasimenko and Denis Parkhomenko and Viacheslav Vasilev and Alexey Letunovskiy and Nikolai Vaulin and Maria Kovaleva and Ivan Kirillov and Lev Novitskiy and Denis Koposov and Nikita Kiselev and Alexander Varlamov and Dmitrii Mikhailov and Vladimir Polovnikov and Andrey Shutkin and Julia Agafonova and Ilya Vasiliev and Anastasiia Kargapoltseva and Anna Dmitrienko and Anastasia Maltseva and Anna Averchenkova and Olga Kim and Tatiana Nikulina and Denis Dimitrov},
      year={2025},
      eprint={2511.14993},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14993}, 
}

@misc{mikhailov2025nablanablaneighborhoodadaptiveblocklevel,
      title={$\nabla$NABLA: Neighborhood Adaptive Block-Level Attention}, 
      author={Dmitrii Mikhailov and Aleksey Letunovskiy and Maria Kovaleva and Vladimir Arkhipkin
              and Vladimir Korviakov and Vladimir Polovnikov and Viacheslav Vasilev
              and Evelina Sidorova and Denis Dimitrov},
      year={2025},
      eprint={2507.13546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13546}, 
}
```

# Acknowledgements

We gratefully acknowledge the open-source projects and research that made Kandinsky 5.0 possible:

- [PyTorch](https://pytorch.org/) — for model training and inference.  
- [FlashAttention 3](https://github.com/Dao-AILab/flash-attention) — for efficient attention and faster inference.  
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL) — for providing high-quality text embeddings.  
- [CLIP](https://github.com/openai/CLIP) — for robust text–image alignment.  
- [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) — for video latent encoding and decoding.  
- [MagCache](https://github.com/Zehong-Ma/MagCache) — for accelerated inference.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — for integration into node-based workflows.  

We deeply appreciate the contributions of these communities and researchers to the open-source ecosystem.





