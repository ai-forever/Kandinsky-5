<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a href="">Article</a> | <a href=>Project Page</a> | <a href=>Generate🤗</a> | Technical Report (soon) | <a href=> T2V Flash🤗</a>
</div>

<h1>Kandinsky 5.0: A family of diffusion models for Video generation</h1>

In this repository, we provide a family of diffusion models to generate a video given a textual prompt or an image (<em>Coming Soon</em>), a distilled model for a faster generation and a video to audio generation model.

## Project Updates

- 🔥 **Source**: ```2025/XX/XX```: We have open-sourced `Kandinsky 5.0 T2V` a pretrain-5s-lite, pretrain-10s-lite, sft-5s-lite, sft-10s-lite, distil-5s-lite versions of `Kandinsky 5.0 T2V` text-to-video generation model.

## Table of contents
<ul>
  <li><a href="#kandinsky-50-t2v">Kandinsky 5.0 T2V LITE</a>: Small text-to-video model </em></li>
  <li><a href="#kandinsky-50-t2v">Kandinsky 5.0 T2V PRO</a>: A text-to-video model - <em>Coming Soon</em></li>
  <li><a href="#kandinsky-50-i2v-image-to-video">Kandinsky 5.0 I2V</a>: An image-to-video model - <em>Coming Soon</em> </li>
</ul>


## Kandinsky 5.0 T2V LITE

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
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
  </tr>

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
NUMBER_OF_DEVICES_PER_NODE=8
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py
```

# Authors

<B>Project Leader:</B> Denis Dimitrov. </br>
<B>Scientific Advisors:</B> Andrey Kuznetsov, Sergey Markov.</br>
<B>Training Pipeline & Model Pretrain & Model Distillation:</B> Vladimir Arkhipkin, Lev Novitskiy, Maria Kovaleva. </br>
<B>Model Architecture:</B> Vladimir Arkhipkin, Maria Kovaleva, Arsen Kuzhamuratov, Nikolay Gerasimenko, Mikhail Zhirnov, Alexander Gambashidze, Konstantin Sobolev.</br>
<B>Data Pipeline:</B> Ivan Kirillov, Andrei Shutkin, Kirill Chernishev, Julia Agafonova, Elizaveta Dakhova, Denis Parkhomenko.</br>
<B>Quality Assessment:</B> Nikolay Gerasimenko, Anna Averchenkova, Victor Panshin, Vladislav Veselov, Pavel Perminov, Vladislav Rodionov, Sergey Skachkov, Stepan Ponomarev.</br>
<B>Other Contributors:</B> Viacheslav Vasilev, Andrei Filatov, Gregory Leleytner.</br>
