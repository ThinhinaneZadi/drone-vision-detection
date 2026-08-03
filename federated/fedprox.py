"""
fedprox.py — FedProx proximal-term support for local training.

FedProx (Li et al., 2020) modifies plain FedAvg by adding a proximal term
to each client's local loss: (mu/2) * ||theta_local - theta_global||^2.
This discourages a client's local weights from drifting too far from the
global model it started the round with, directly targeting the client-drift
problem already measured in this project's freeze=0 vs freeze=11 comparison.

Implemented here as a gradient modification (mathematically equivalent to
adding the term to the loss): before each optimizer step, we add
mu * (p - p_global) to each trainable parameter's gradient, then let the
underlying AdamW step proceed normally.

NOTE: Ultralytics uses gradient accumulation (accumulate ~= 64/batch_size),
so the optimizer's .step() is only called once every `accumulate` mini-
batches, not every mini-batch. This means the proximal term only has a
real, nonzero effect from the SECOND real optimizer step onward within a
local training run — verified directly via debug instrumentation during
development (see git history for federated/fedprox.py). At very small
batch sizes with few total mini-batches per epoch, a client may get only
one real optimizer step, in which case the proximal term will have
essentially no effect that round — batch=16 (this project's standard
setting for real experiments) reliably produces multiple real steps.
"""
import torch


class ProxAdamW(torch.optim.AdamW):
    """AdamW with an added FedProx proximal term toward a fixed reference
    (the global model's weights at the start of this round)."""

    def __init__(self, params, global_params, mu, **kwargs):
        super().__init__(params, **kwargs)
        self.mu = mu
        # global_params: list of tensors, same order/shapes as `params`,
        # detached and on the same device — never updated during training
        self._global_params = [g.detach().clone() for g in global_params]

    def step(self, closure=None):
        if self.mu > 0:
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.add_(p.data - self._global_params[idx], alpha=self.mu)
                    idx += 1
        return super().step(closure)
