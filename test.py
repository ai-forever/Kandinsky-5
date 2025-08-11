from kandinsky import get_T2V_pipeline

import time 

if __name__ == "__main__":
    
    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:1", "text_embedder": "cuda:2" },
        resolution = 512,
        dit_path="/home/jovyan/shares/SR008.fs2/maria/kandinsky/kandinsky5/saved_models/gathered_checkpoints/video_2B_T2V_I2V_512_768_1_step_45000.pt",
    )

    start_time = time.perf_counter()
    x = pipe("a cat in a blue hat", time_length=5, width=768, height=512, save_path='./test1.mp4')
    
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")