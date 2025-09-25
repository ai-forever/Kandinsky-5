from kandinsky import get_T2V_pipeline

import time 
import gc

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def print_mem_stats(text):
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated {text}: {peak_memory_bytes / (1024**3):.2f} GB")
    torch.cuda.reset_peak_memory_stats()


def main():
    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0" },
    )
    seed = 42

    # wu
    for i in range(2):
        start_time = time.perf_counter()
        x = pipe(
            "a cat in a blue hat", time_length=5, width=768, height=512, 
            save_path=f'./out/test/wu{i}.mp4', seed=seed+i)
        print(f"WU {i}, TIME ELAPSED: {time.perf_counter() - start_time}")

        print_mem_stats(f'WU {i}')
        del x
        torch.cuda.empty_cache()
        gc.collect()

    # test
    start_time = time.perf_counter()
    x = pipe(
        "a cat in a blue hat", time_length=5, width=768, height=512, 
        save_path=f'./out/test/test.mp4', seed=seed+2)
    print(f"TEST, TIME ELAPSED: {time.perf_counter() - start_time}")
    print_mem_stats(f'WU {i}')


if __name__ == "__main__":
    main()
