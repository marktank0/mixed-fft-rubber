# -*- coding: utf-8 -*-
"""Multi-GPU helpers: one process per GPU, launched by torchrun.

    torchrun --standalone --nproc_per_node=4 CNN/train.py

`setup()` reads the environment variables torchrun sets (RANK, WORLD_SIZE,
LOCAL_RANK); when they are absent every helper degrades to a no-op, so the
same train.py runs unchanged on one GPU or on a CPU.
"""

import os

import numpy as np
import torch
import torch.distributed as dist


class DistContext(object):
    """Rank bookkeeping for one process.

    `is_main` gates everything that must happen exactly once - printing,
    checkpointing, writing the history - while `world_size` scales the things
    that are split across ranks.
    """

    def __init__(self, rank=0, world_size=1, local_rank=0, enabled=False):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.enabled = enabled

    @property
    def is_main(self):
        return self.rank == 0

    def print(self, *args, **kwargs):
        """Print from rank 0 only, so the log is not multiplied by world_size."""
        if self.is_main:
            print(*args, **kwargs)

    def barrier(self):
        if self.enabled:
            dist.barrier()


def env_is_distributed():
    """True when torchrun (or SLURM via torchrun) set up the rendezvous."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup(backend=None):
    """Join the process group if we were launched under torchrun.

    Returns a DistContext either way. NCCL is the right backend for GPUs;
    gloo is the fallback for a CPU-only debug run.
    """
    if not env_is_distributed():
        return DistContext()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return DistContext(rank, world_size, local_rank, enabled=True)


def cleanup(ctx):
    if ctx.enabled:
        dist.barrier()
        dist.destroy_process_group()


def device_for(ctx, requested):
    """Pick this rank's device.

    Under torchrun each rank owns exactly one GPU, so the requested string is
    ignored in favour of the local rank's device.
    """
    if not torch.cuda.is_available():
        if str(requested).startswith("cuda"):
            print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if ctx.enabled:
        return torch.device("cuda", ctx.local_rank)
    return torch.device(requested)


def reduce_mean(value, ctx, device):
    """Average a plain Python float across ranks."""
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / ctx.world_size)


def gather_numpy(array, ctx, device):
    """Concatenate a per-rank 1-D float array across ranks.

    DistributedSampler pads the last batch so every rank walks the same number
    of samples, which is what lets this use a fixed-size all_gather.
    """
    if not ctx.enabled:
        return array
    tensor = torch.as_tensor(np.ascontiguousarray(array),
                             dtype=torch.float32, device=device)
    buckets = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
    dist.all_gather(buckets, tensor)
    return torch.cat(buckets).cpu().numpy()
