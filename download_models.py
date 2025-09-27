from huggingface_hub import hf_hub_download, snapshot_download
import os

if __name__ == "__main__":

    cache_dir = "./weights"
    
    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    dit_path = snapshot_download(
        repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s",
        allow_patterns="model/*",
        local_dir=cache_dir,
    )

    # dit_path = snapshot_download(
    #     repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-10s",
    #     allow_patterns="model/*",
    #     local_dir=cache_dir,
    # )
    
    vae_path = snapshot_download(
        repo_id="hunyuanvideo-community/HunyuanVideo",
        allow_patterns="vae/*",
        local_dir=cache_dir,
    )

    text_encoder_path = snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=os.path.join(cache_dir, "text_encoder/"),
    )

    text_encoder2_path = snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=os.path.join(cache_dir, "text_encoder2/"),
    )

    