"""Data-parallel GRPO on tandem: one process per device, each with its own relay engine
and its own share of the prompts, gradients averaged by tandem's DDP (its own ring
all-reduce, no torch.distributed, no NCCL). tandem lives in the submodule next to this
package."""

import os
import sys

_TANDEM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tandem")
if os.path.isdir(_TANDEM) and _TANDEM not in sys.path:
    sys.path.insert(0, _TANDEM)


def tandem():
    import tandem  # noqa: F401  (the submodule)
    from tandem import comm, ddp

    return comm, ddp


def launch(fn, size, *args, device="cpu"):
    """Runs fn(group, *args) on `size` tandem ranks; returns their results in rank order."""
    comm, _ = tandem()
    return comm.launch(fn, size, *args, device=device)


def wrap(policy, group):
    """tandem's DDP around the policy (rank 0's weights are broadcast to every rank)."""
    _, ddp = tandem()
    return ddp.DDP(policy, group)
