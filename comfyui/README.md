# Kandinsky 5 Video for ComfyUI

![Kandinsky 5 ComfyUI Workflow](../assets/comfyui_kandinsky5.png)


## Description

This project provides a workflow for generating videos using the Kandinsky 5 model within the ComfyUI environment.

## Installation and Setup

### 1. Install ComfyUI

If you don't have ComfyUI installed yet, follow these steps:

```bash
# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Launch ComfyUI
python main.py

```

### 2. Clone this repository into the ComfyUI custom_nodes folder:
```bash
# Navigate to ComfyUI custom_nodes folder
cd ComfyUI/custom_nodes

# Clone this repository and install requirements for model
git clone https://github.com/ai-forever/Kandinsky-5.git kandinsky
cd kandinsky
pip install -r requirements.txt
```

### 3. Load the Workflow
Launch ComfyUI (typically available at http://127.0.0.1:8188)

In the ComfyUI interface, click the "Load" button

Select the kandisnky5_lite_T2V.json file from this folder of this repository

The workflow will load into the ComfyUI interface

### 4. Download Checkpoints

Download the required models and place them in the appropriate folders. 

1. Run download_models.py It will download models and encoders to ./weights directory.
2. Rearrange them to comfyui paths(text_encoders/diffusion_models/vae).

```file-tree
ComfyUI/
├── models/
│   ├── text_encoders/          # For text_encoder and text_encoder2 models
│   ├── diffusion_models/       # For kandinsky5lite_t2v_*.safetensors models  
│   └── vae/                    # For vae model
```

### 5. Configure Parameters
After loading the workflow, configure the following parameters:

### Main Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| **Prompt** | Text description for video generation | Your descriptive text |
| **Negative Prompt** | What to exclude from generation | Unwanted elements description |
| **Width/Height/Length** | Output video size | 768x512x121 for 5s or 768x512x241 for 10s, Width and Height should be divisisble  by 128 for 10s model |
| **Steps** | Number of generation steps | 50, 16 for distilled version|
| **CFG Scale** |  | 1.0 for distilled16steps and nocfg, 5.0 for sft and pretrain models |
| **Scheduler Scale** | Noise scheduler scale | 5.0 for 5s, 10.0 for 10s |