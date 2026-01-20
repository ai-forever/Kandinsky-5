import torch

def broadcast_string_as_tensor(data: str, rank: int):
    data_len = torch.tensor(len(data), dtype=torch.int32).to(rank)
    torch.distributed.broadcast(data_len, 0)
    torch.distributed.barrier()
    data_len = data_len.cpu().numpy()

    if rank == 0:
        data_tensor = torch.tensor(list(data.encode('utf-8'))).to(rank)
    else:
        data_tensor = torch.zeros(data_len, dtype=torch.int64).to(rank)
    torch.distributed.broadcast(data_tensor, 0)
    torch.distributed.barrier()
    data = bytes(data_tensor.cpu().tolist()).decode('utf-8')
    return data
