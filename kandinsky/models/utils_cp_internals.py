import torch.distributed as tdi

_CONTEXT_PARALLEL_GROUP = "Not_initialized"
_CONTEXT_PARALLEL_SIZE = None


def is_context_parallel_initialized():
    if _CONTEXT_PARALLEL_GROUP == "Not_initialized":
        return False
    else:
        return True


def set_context_parallel_group(size, group):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    _CONTEXT_PARALLEL_GROUP = group
    _CONTEXT_PARALLEL_SIZE = size


def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    assert _CONTEXT_PARALLEL_GROUP == "Not_initialized", "context parallel group is already initialized"
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    rank = tdi.get_rank()
    world_size = tdi.get_world_size()

    if world_size == context_parallel_size:
        _CONTEXT_PARALLEL_GROUP = None
    else:
        for i in range(0, world_size, context_parallel_size):
            ranks = range(i, i + context_parallel_size)
            group = tdi.new_group(ranks)
            if rank in ranks:
                _CONTEXT_PARALLEL_GROUP = group
                break


def get_context_parallel_group():
    assert _CONTEXT_PARALLEL_GROUP != "Not_initialized", "context parallel group is not initialized"

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE


def get_context_parallel_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    if _CONTEXT_PARALLEL_SIZE == 1:
        return 0

    rank = tdi.get_rank()

    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


def get_context_parallel_group_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    if _CONTEXT_PARALLEL_SIZE == 1:
        return 0

    rank = tdi.get_rank()

    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank