NUMBER_OF_NODES=1 
NUMBER_OF_DEVICES_PER_NODE=2
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py