"""Run inference on a pretrained OmniLearn model.

Two entry points are exposed: :func:`batched_inference` for single-GPU
(or CPU) work, and :func:`batched_inference_distributed` for multi-GPU
runs launched under ``torchrun``. The latter expects the standard
``LOCAL_RANK`` / ``WORLD_SIZE`` environment variables to be set.

Both entry points return a CPU tensor in the original input order.
"""

from __future__ import annotations

import gc
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm.auto import tqdm

__all__ = [
    "batched_inference",
    "batched_inference_distributed",
    "cleanup_distributed",
    "is_main_process",
    "release_memory",
    "setup_distributed",
]


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed() -> int:
    """Initialise the NCCL process group and bind this process to its GPU.

    Returns
    -------
    int
        The local rank (= the GPU index this process owns).
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    """Tear down the process group."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """True if this process is global rank 0 (or non-distributed)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Inference -- single device
# ---------------------------------------------------------------------------


def batched_inference(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Run ``model`` on ``data`` in batches on a single device.

    Parameters
    ----------
    model : torch.nn.Module
    data : torch.Tensor
        Full input tensor.
    batch_size : int
    device : str, torch.device, or None
        If ``None``, ``"cuda"`` is used when available.
    show_progress : bool

    Returns
    -------
    torch.Tensor
        Concatenated outputs on CPU, in input order.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
    iterable = tqdm(loader, desc="Inference") if show_progress else loader

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch,) in iterable:
            outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Inference -- multi-GPU via torchrun
# ---------------------------------------------------------------------------


def _gather_in_input_order(
    local_results: torch.Tensor,
    n_total: int,
    world_size: int,
) -> torch.Tensor | None:
    """Gather per-rank tensors and reorder them into the original input order.

    NCCL only supports GPU tensors, so a separate Gloo group is used for
    the CPU-tensor communication. ``DistributedSampler`` with
    ``shuffle=False`` assigns rank ``r`` the indices
    ``r, r+world_size, r+2*world_size, ...``; this routine inverts that
    striping. Returns ``None`` on non-zero ranks.

    Note: a new Gloo group is created on every call. This is fine for
    single-shot CLI runs (the typical use case) but should be hoisted
    out by the caller if the function is invoked many times in one
    process.
    """
    cpu_group = dist.new_group(backend="gloo")
    global_rank = dist.get_rank()

    # Each rank may have processed a slightly different number of samples.
    local_size = torch.tensor([local_results.shape[0]], dtype=torch.long)
    all_sizes = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=cpu_group)
    sizes = [int(s.item()) for s in all_sizes]

    # Pad each rank's tensor to the maximum size so we can use dist.gather.
    max_size = max(sizes)
    output_shape = [max_size, *list(local_results.shape[1:])]
    padded = torch.zeros(output_shape, dtype=local_results.dtype)
    padded[: local_results.shape[0]] = local_results

    gathered = [torch.zeros_like(padded) for _ in range(world_size)] if global_rank == 0 else None
    dist.gather(padded, gathered, dst=0, group=cpu_group)

    if global_rank != 0:
        return None
    assert gathered is not None  # populated exactly on global rank 0

    # Reassemble in input order. trimmed[r] are the real samples held by rank r.
    trimmed = [gathered[r][: sizes[r]] for r in range(world_size)]
    in_order = torch.zeros(n_total, *local_results.shape[1:], dtype=local_results.dtype)
    for rank in range(world_size):
        rank_indices = list(range(rank, n_total, world_size))
        in_order[rank_indices] = trimmed[rank][: len(rank_indices)]
    return in_order


def batched_inference_distributed(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int = 256,
    num_workers: int = 2,
    show_progress: bool = True,
) -> torch.Tensor | None:
    """Run ``model`` on ``data`` distributed across multiple GPUs.

    Must be called inside a ``torchrun`` process group; see
    :func:`setup_distributed`.

    Parameters
    ----------
    model : torch.nn.Module
    data : torch.Tensor
        Full input tensor (replicated on every rank).
    batch_size : int
        Per-GPU batch size.
    num_workers : int
        DataLoader worker count.
    show_progress : bool
        Show a tqdm bar on rank 0 only.

    Returns
    -------
    torch.Tensor or None
        On global rank 0: outputs in original input order.
        On other ranks: ``None``.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    model.to(local_rank)
    model.eval()

    sampler: DistributedSampler[tuple[torch.Tensor, ...]] = DistributedSampler(
        TensorDataset(data),
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
    )
    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    iterable = tqdm(loader, desc="Inference") if show_progress and global_rank == 0 else loader

    local_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch,) in iterable:
            local_outputs.append(model(batch.to(local_rank)).cpu())

    local_results = torch.cat(local_outputs, dim=0)
    torch.cuda.empty_cache()

    return _gather_in_input_order(local_results, n_total=data.shape[0], world_size=world_size)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def release_memory() -> None:
    """Run garbage collection and release CUDA cached memory.

    Callers should drop their own references (``del``) first so the
    collected objects are actually reclaimable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
