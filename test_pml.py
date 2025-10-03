import argparse
import time

from kandinsky import get_T2V_pipeline

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--path", type=str, default=None, help="Name of the resulting file")
    args = parser.parse_args()

    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0",
                    "text_embedder": "cuda:0"},
        conf_path="./configs/config_pml.yaml"
    )


    start_time = time.perf_counter()
    x = pipe("A cat in a red hat", time_length=5, width=768, height=512, num_steps=50, guidance_weight=5.0, scheduler_scale=10.0, save_path=args.path)
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
