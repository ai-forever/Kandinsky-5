# Kandinsky 
For PML Conf 2025

### How to run:
1) Install dependencies: 
```sh
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```
(tested in env with python 3.11 and cuda 12.6)

2) Download models:
```sh
python download_models.py
```

3) Run example:
```sh
python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test_pml.py --path "./test1.mp4"
```
