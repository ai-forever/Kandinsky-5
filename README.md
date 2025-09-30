<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a href="">Habr</a> | <a href="https://ai-forever.github.io/Kandinsky-5/">Project Page</a> | Technical Report (soon) | <a href=https://huggingface.co/collections/ai-forever/kandisnky-50-t2v-lite-68d71892d2cc9b02177e5ae5> Models🤗 </a>
</div>

<h1>Kandinsky 5.0: A family of diffusion models for Video & Image generation</h1>

In this repository, we provide a family of diffusion models to generate a video or an image (<em>Coming Soon</em>) given a textual prompt and distilled model for faster generation.

https://github.com/user-attachments/assets/b9ff0417-02a4-4f6b-aacc-60c44e7fe6f1

## Project Updates

- 🔥 **Source**: ```2025/09/29```: We have open-sourced `Kandinsky 5.0 T2V Lite` a lite (2B parameters) version of `Kandinsky 5.0 Video` text-to-video generation model. Released checkpoints: `kandinsky5lite_t2v_pretrain_5s`, `kandinsky5lite_t2v_pretrain_10s`, `kandinsky5lite_t2v_sft_5s`, `kandinsky5lite_t2v_sft_10s`, `kandinsky5lite_t2v_nocfg_5s`, `kandinsky5lite_t2v_nocfg_10s`, `kandinsky5lite_t2v_distilled16steps_5s`, `kandinsky5lite_t2v_distilled16steps_10s` contains weight from pretrain, supervised finetuning, cfg distillation and diffusion distillation into 16 steps. 5s checkpoints are capable of generating videos up to 5 seconds long. 10s checkpoints is faster models checkpoints trained with [NABLA](https://huggingface.co/ai-forever/Wan2.1-T2V-14B-NABLA-0.7) algorithm and capable to generate videos up to 10 seconds long.

## Kandinsky 5.0 T2V Lite

Kandinsky 5.0 T2V Lite is a lightweight video generation model (2B parameters) that ranks #1 among open-source models in its class. It outperforms larger Wan models (5B and 14B) and offers the best understanding of Russian concepts in the open-source ecosystem.

We provide 8 model variants, each optimized for different use cases:

* SFT model — delivers the highest generation quality;

* CFG-distilled — runs 2× faster;

* Diffusion-distilled — enables low-latency generation with minimal quality loss (6× faster);

* Pretrain model — designed for fine-tuning by researchers and enthusiasts.

All models are available in two versions: for generating 5-second and 10-second videos.

## Pipeline

**Latent diffusion pipeline** with **Flow Matching**.

**Diffusion Transformer (DiT)** as the main generative backbone with **cross-attention to text embeddings**.

- **Qwen2.5-VL** and **CLIP** provides text embeddings.

- **HunyuanVideo 3D VAE** encodes/decodes video into a latent space.

- **DiT** is the main generative module using cross-attention to condition on text.

<img width="1600" height="477" alt="Picture1" src="https://github.com/user-attachments/assets/17fc2eb5-05e3-4591-9ec6-0f6e1ca397b3" />

<img width="800" height="406" alt="Picture2" src="https://github.com/user-attachments/assets/f3006742-e261-4c39-b7dc-e39330be9a09" />


## Model Zoo

| Model                               | config | video duration | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|----------------|-----|------------|----------------|
| Kandinsky 5.0 T2V Lite SFT 5s       |configs/config_5s_sft.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s) |      139 s     |
| Kandinsky 5.0 T2V Lite SFT 10s      |configs/config_10s_sft.yaml| 10s            | 100 |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s) |      224 s     |
| Kandinsky 5.0 T2V Lite pretrain 5s  |configs/config_5s_pretrain.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s) |      139 s      |
| Kandinsky 5.0 T2V Lite pretrain 10s |configs/config_10s_pretrain.yaml | 10s            | 100 |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s) |     224 s      |
| Kandinsky 5.0 T2V Lite no-CFG 5s    |configs/config_5s_nocfg.yaml| 5s             | 50  |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s) |       77 s     |
| Kandinsky 5.0 T2V Lite no-CFG 10s   |configs/config_10s_nocfg.yaml| 10s            | 50  |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s) |     124 s      |
| Kandinsky 5.0 T2V Lite distill 5s   |configs/config_5s_distil.yaml| 5s             | 16  | 🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s)|       35 s     |
| Kandinsky 5.0 T2V Lite distill 10s  |        | 10s            | 16  |            |      55 s      |              |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

#### Kandinsky 5.0 T2V Lite SFT

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bc38821b-f9f1-46db-885f-1f70464669eb" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9f64c940-4df8-4c51-bd81-a05de8e70fc3" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/77dd417f-e0bf-42bd-8d80-daffcd054add" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/385a0076-f01c-4663-aa46-6ce50352b9ed" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7c1bcb31-cc7d-4385-9a33-2b0cc28393dd" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/990a8a0b-2df1-4bbc-b2e3-2859b6f1eea6" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>


#### Kandinsky 5.0 T2V Lite Distill

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/861342f9-f576-4083-8a3b-94570a970d58" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/302e4e7d-781d-4a58-9b10-8c473d469c4b" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3e70175c-40e5-4aec-b506-38006fe91a76" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b7da85f7-8b62-4d46-9460-7f0e505de810" width=200 controls autoplay loop></video>
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

</table>

#### VBench results

<div align="center">
  <picture>
    <img src="assets/vbench.png">
  </picture>
</div>

## Quickstart

#### Installation
Clone the repo:
```sh
git clone https://github.com/ai-forever/Kandinsky-5.git
cd Kandinsky-5
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

#### Run Kandinsky 5.0 T2V Lite SFT 5s

```sh
python test.py --prompt "A dog in red hat"
```

#### Run Kandinsky 5.0 T2V Lite SFT 10s 

```sh
python test.py --config ./configs/config_10s_sft.yaml --prompt "A dog in red hat" --video_duration 10 
```

#### Run Kandinsky 5.0 T2V Lite pretrain 5s

```sh
python test.py --config ./configs/config_5s_pretrain.yaml --prompt "A dog in red hat"
```

#### Run Kandinsky 5.0 T2V Lite pretrain 10s

```sh
python test.py --config ./configs/config_10s_pretrain.yaml --prompt "A dog in red hat" --video_duration 10
```

#### Run Kandinsky 5.0 T2V Lite no-CFG 5s

```sh
python test.py --config ./configs/config_5s_nocfg.yaml --prompt "A dog in red hat" 
```

#### Run Kandinsky 5.0 T2V Lite no-CFG 10s

```sh
python test.py --config ./configs/config_10s_nocfg.yaml --prompt "A dog in red hat" --video_duration 10
```

#### Run Kandinsky 5.0 T2V Lite distill 5s

```sh
python test.py --config ./configs/config_5s_distil.yaml --prompt "A dog in red hat"          
```

#### Run Kandinsky 5.0 T2V Lite distill 10s

Coming soon

### Inference

```python
import torch
from IPython.display import Video
from kandinsky import get_T2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2V_pipeline(device_map, conf_path="configs/config_5s_sft.yaml")

images = pipe(
    seed=42,
    time_length=5,
    width=768,
    height=512,
    save_path="./test.mp4",
    text="A cat in a red hat",
)

Video("./test.mp4")
```

Please, refer to [inference_example.ipynb](inference_example.ipynb) notebook for more usage details.

### Distributed Inference

For a faster inference, we also provide the capability to perform inference in a distributed way:
```
NUMBER_OF_NODES=1
NUMBER_OF_DEVICES_PER_NODE=1 / 2 / 4
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py
```

### ComfyUI

See the instruction [here](comfyui)

## 📑 Todo List
- Kandinsky 5.0 Lite Text-to-Video
    - [x] Multi-GPU Inference code of the 2B models
    - [ ] Checkpoints 2B models
      - [x]  pretrain
      - [x] sft
      - [ ] rl
      - [x] cfg distil 
      - [x] distil 16 steps
      - [ ] autoregressive generation
    - [x] ComfyUI integration
    - [ ] Diffusers integration
    - [ ] Caching acceleration support
- Kandinsky 5.0 Lite Image-to-Video
    - [ ] Multi-GPU Inference code of the 2B model
    - [ ] Checkpoints of the 2B model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Text-to-Video
    - [ ] Multi-GPU Inference code of the models
    - [ ] Checkpoints of the model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Image-to-Video
    - [ ] Multi-GPU Inference code of the model
    - [ ] Checkpoints of the model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- [ ] Technical report

# Authors
<B>Project Leader:</B> Denis Dimitrov</br>

<B>Team Leads:</B> Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko</br>

<B>Core Contributors:</B> Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim, Anastasiia Kargapoltseva, Nikita Kiselev</br>

<B>Contributors:</B> Anna Dmitrienko,  Anastasia Maltseva, Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov, Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Tatiana Nikulina, Polina Gavrilova</br>

# Citation

```
@misc{kandinsky2025,
    author = {Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov,
              Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim,
              Anastasiia Kargapoltseva, Nikita Kiselev, Vladimir Arkhipkin, Vladimir Korviakov,
              Nikolai Gerasimenko, Denis Parkhomenko, Anna Dmitrienko, Anastasia Maltseva,
              Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov,
              Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina,
              Tatiana Nikulina, Polina Gavrilova, Denis Dimitrov},
    title = {Kandinsky 5.0: A family of diffusion models for Video & Image generation},
    howpublished = {\url{https://github.com/ai-forever/Kandinsky-5}},
    year = 2025
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
