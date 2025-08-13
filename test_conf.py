from kandinsky import get_T2V_pipeline

import time 

if __name__ == "__main__":

    # TEST config_flash_5s.yaml
    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"},
        conf_path="./configs/config_flash_5s.yaml"
    )
    start_time = time.perf_counter()
    x = pipe("a cat in a blue hat", time_length=5, width=768, height=512, save_path='./test2.mp4')
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")


    # TEST config_nabla_10s.yaml
    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0" },
        conf_path="./configs/config_nabla_10s.yaml"
    )
    start_time = time.perf_counter()
    x = pipe("a cat in a blue hat", time_length=10, width=768, height=512, save_path='./test3.mp4')
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")

    
    # TEST config_flash_5s.yaml distil version
    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0" },
        conf_path="./configs/config_flash_5s_distil.yaml"
    )
    start_time = time.perf_counter()
    x = pipe("a cat in a blue hat", time_length=5, width=768, height=512, guidance_weight=1.0, num_steps=16, scheduler_scale=5, save_path='./test4.mp4')
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")