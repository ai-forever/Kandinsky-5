<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a href="">Article</a> | <a href=>Project Page</a> |Technical Report (soon) | <a href=> Models🤗 (soon) </a>
</div>

<h1>Kandinsky 5.0 Video Lite: A family of diffusion models for Video generation</h1>

In this repository, we provide a family of diffusion models to generate a video given a textual prompt or an image (<em>Coming Soon</em>) and distilled model for a faster generation.

## Project Updates

- 🔥 **Source**: ```2025/09/29```: We have open-sourced `Kandinsky 5.0 T2V Lite` a lite (2B parameters) version of `Kandinsky 5.0 Video` text-to-video generation model. Released checkpoints: `lite_pretrain_5s`, `lite_pretrain_10s`, `lite_sft_5s`, `lite_sft_10s`, `lite_nocfg_5s`, `lite_nocfg_10s`, `lite_distil_5s`, `lite_distil_10s` contains weight from pretrain, supervised finetuning, cfg distillation and distillation in 16 steps. 5s checkpoints are capable of generating videos up to 5 seconds long. 10s checkpoints is faster models checkpoints trained with [NABLA](https://huggingface.co/ai-forever/Wan2.1-T2V-14B-NABLA-0.7) algorithm and capable to generate videos up to 10 seconds long.

## Table of contents
<ul>
  <li><a href="#kandinsky-50-t2v">Kandinsky 5.0 T2V Lite</a>: Lite text-to-video model </em></li>
  <li><a href="#kandinsky-50-t2v">Kandinsky 5.0 T2V Pro</a>: Pro text-to-video model - <em>Coming Soon</em></li>
  <li><a href="#kandinsky-50-i2v-image-to-video">Kandinsky 5.0 I2V Lite</a>: A lite image-to-video model - <em>Coming Soon</em> </li>
  <li><a href="#kandinsky-50-i2v-image-to-video">Kandinsky 5.0 I2V Pro</a>: A pro image-to-video model - <em>Coming Soon</em> </li>
</ul>


## Kandinsky 5.0 T2V Lite

Kandinsky 5.0 T2V Lite — top 1 video generation model among open source in its class (small and lightweight model with 2B parameters, better than the Wan 5B, 14B), top 1 in knowledge of Russian concepts in open source.

https://github.com/user-attachments/assets/b9ff0417-02a4-4f6b-aacc-60c44e7fe6f1


### Reesults: 
TODO: add SBS

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <!-- <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d5a0c11e-020b-4e56-9a17-5b3995890908" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/98ba32be-96c7-4d6c-8ffa-3cf77710581a" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/140b64ae-9c34-4763-98a6-4c7408be3a4e" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d3eab231-d7e8-4f0a-9829-2b066ad8301d" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f955f0e0-7141-4413-aa1e-11827c108f83" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4eb10e1d-60a0-4ff9-ad7e-9b5ab0a0fff8" width=200 controls autoplay loop></video>
      </td>
  </tr> -->

</table>

### Load Models

```python download_models.py```

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

pipe = get_T2V_pipeline(device_map)

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
NUMBER_OF_DEVICES_PER_NODE=2 or 4
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py
```

## 📑 Todo List
- Kandinsky 5.0 Lite Text-to-Video
    - [x] Multi-GPU Inference code of the 2B models
    - [ ] Checkpoints 2B models
      - [x]  pretrain
      - [x] sft
      - [ ] rl
      - [ ] cfg distil 
      - [x] distil 16 steps
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Lite Image-to-Video
    - [ ] Multi-GPU Inference code of the 2B model
    - [ ] Checkpoints of the 2B model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Text-to-Video
    - [ ] Multi-GPU Inference code of the 2B models
    - [ ] Checkpoints 2B models
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Image-to-Video
    - [ ] Multi-GPU Inference code of the 2B model
    - [ ] Checkpoints of the 2B model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration

 
## Quickstart

#### Installation
Clone the repo:
```sh
git clone TODO add actual repo
cd TODO add actual repo name
```

Install dependencies: # TODO add requirements.txt
```sh
pip install -r requirements.txt
```

#### Model Download
```sh
python download_models.py
```

#### Run Test Example
```sh
python test.py --width 768 --height 512 --prompt "A dog in red hat"
```

# Authors
<B>Project Leader:</B> Denis Dimitrov</br>

<B>Team Leads:</B> Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko</br>

<B>Core Contributors:</B> Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim, Anastasiia Kargapoltseva, Nikita Kiselev</br>

<B>Contributors:</B> Anna Dmitrienko,  Anastasia Maltseva, Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov, Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Tatiana Nikulina, Polina Gavrilova</br>
