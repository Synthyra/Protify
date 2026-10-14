"""Many independent probes trained at once: one stacked set of weights, a group axis, and closed-form gradients.

Group g is one probe: a linear map, or a one-hidden-layer MLP, from a w-wide input to C outputs. Every
parameter has the group axis first, so a batch of inputs (n, G, w) trains all G probes with two einsums, and a
streaming consumer can train the groups of one layer by slicing. Gradients are written by hand (the losses and
layers are simple), which avoids autograd graphs and full-size gradient buffers for each slice. The optimizer is
AdamW, as PyTorch defines it, with a learning rate and weight decay per group, so a small grid of settings trains
side by side as extra groups and each probe keeps the setting that validates best.

Symbols: n rows of a batch (proteins, or residues); G groups; w input width; m hidden width; C outputs.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import Tensor


LOSSES = ("multiclass", "multilabel", "regression")


@dataclass(frozen=True)
class ProbeSettings:
    """The architecture and optimizer of every probe in a bank."""

    kind: str = "linear"  # linear or mlp
    hidden: int = 512
    dropout: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.kind not in ("linear", "mlp"):
            raise ValueError(f"Probe kind must be 'linear' or 'mlp', not {self.kind!r}.")


def _uniform_(tensor: Tensor, fan_in: int, generator: torch.Generator) -> Tensor:
    """PyTorch's nn.Linear initialization, U(-1/sqrt(fan_in), 1/sqrt(fan_in)), for weights and biases alike."""
    bound = 1.0 / math.sqrt(fan_in)
    return tensor.uniform_(-bound, bound, generator=generator)


def _gelu_grad(pre: Tensor) -> Tensor:
    """Derivative of the exact GELU, Phi(x) + x phi(x)."""
    cdf = 0.5 * (1.0 + torch.erf(pre / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * pre.square()) / math.sqrt(2.0 * math.pi)
    return cdf + pre * pdf


class ProbeBank:
    """G probes of one input width and one output size, trained together."""

    def __init__(
        self,
        names: list[str],
        width: int,
        outputs: int,
        loss: str,
        settings: ProbeSettings,
        learning_rates: Tensor,
        weight_decays: Tensor,
        seed: int,
        device: torch.device | str,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"Loss must be one of {LOSSES}, not {loss!r}.")
        groups = len(names)
        assert learning_rates.shape == (groups,) and weight_decays.shape == (groups,)
        self.names = list(names)
        self.width = width
        self.outputs = outputs
        self.loss = loss
        self.settings = settings
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if settings.kind == "linear":
            shapes = {"weight": (groups, width, outputs), "bias": (groups, outputs)}
            fan_ins = {"weight": width, "bias": width}
        else:
            hidden = settings.hidden
            shapes = {
                "weight1": (groups, width, hidden), "bias1": (groups, hidden),
                "weight2": (groups, hidden, outputs), "bias2": (groups, outputs),
            }
            fan_ins = {"weight1": width, "bias1": width, "weight2": hidden, "bias2": hidden}
        self.params = {
            name: _uniform_(torch.empty(shape), fan_ins[name], generator).to(self.device)
            for name, shape in shapes.items()
        }
        self.initial = {name: value.clone() for name, value in self.params.items()}
        self.best = {name: value.clone() for name, value in self.params.items()}
        self.grads = {name: torch.zeros_like(value) for name, value in self.params.items()}
        self.first_moment = {name: torch.zeros_like(value) for name, value in self.params.items()}
        self.second_moment = {name: torch.zeros_like(value) for name, value in self.params.items()}
        self.learning_rates = learning_rates.to(self.device, torch.float32)  # (G,)
        self.weight_decays = weight_decays.to(self.device, torch.float32)  # (G,)
        self.steps = 0
        self.mean = torch.zeros(groups, width, device=self.device)  # (G, w)
        self.inverse_std = torch.ones(groups, width, device=self.device)  # (G, w)
        self.best_loss = torch.full((groups,), math.inf, device=self.device)  # (G,)
        self.best_step = torch.zeros(groups, dtype=torch.long, device=self.device)  # (G,)
        self.misses = torch.zeros(groups, dtype=torch.long, device=self.device)  # (G,)
        self.stopped = torch.zeros(groups, dtype=torch.bool, device=self.device)  # (G,)

    @property
    def groups(self) -> int:
        return len(self.names)

    # -- standardization -------------------------------------------------------------------------------------

    def set_standardization(self, mean: Tensor, variance: Tensor, floor: float = 1e-6) -> None:
        # mean, variance: (G, w) over the training rows
        self.mean = mean.to(self.device, torch.float32)
        self.inverse_std = variance.to(self.device, torch.float32).clamp_min(floor).rsqrt()

    def _prepare(self, X: Tensor, start: int, stop: int) -> Tensor:
        # X: (n, g, w) for groups start..stop
        return (X.to(torch.float32) - self.mean[start:stop]) * self.inverse_std[start:stop]  # (n, g, w)

    # -- forward and gradients --------------------------------------------------------------------------------

    def _weights(self, which: str) -> dict[str, Tensor]:
        return {"current": self.params, "best": self.best, "initial": self.initial}[which]

    def logits(self, X: Tensor, start: int = 0, stop: int | None = None, which: str = "current") -> Tensor:
        """Outputs of groups start..stop for inputs already gathered per group."""
        # X: (n, g, w)
        stop = self.groups if stop is None else stop
        weights = self._weights(which)
        inputs = self._prepare(X, start, stop)  # (n, g, w)
        if self.settings.kind == "linear":
            return torch.einsum("ngw,gwc->ngc", inputs, weights["weight"][start:stop]) + weights["bias"][start:stop]  # (n, g, C)
        pre = torch.einsum("ngw,gwm->ngm", inputs, weights["weight1"][start:stop]) + weights["bias1"][start:stop]  # (n, g, m)
        return torch.einsum("ngm,gmc->ngc", F.gelu(pre), weights["weight2"][start:stop]) + weights["bias2"][start:stop]  # (n, g, C)

    def _loss_terms(self, Z: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Summed loss per group and its gradient with respect to the logits, both unnormalized."""
        # Z: (n, g, C); targets: (n,) class indices, or (n, C) values
        if self.loss == "multiclass":
            log_probs = torch.log_softmax(Z, dim=-1)  # (n, g, C)
            picked = log_probs.gather(-1, targets.view(-1, 1, 1).expand(-1, Z.shape[1], 1)).squeeze(-1)  # (n, g)
            grad = log_probs.exp()  # (n, g, C)
            grad.scatter_add_(-1, targets.view(-1, 1, 1).expand(-1, Z.shape[1], 1), -torch.ones_like(picked).unsqueeze(-1))
            return -picked.sum(dim=0), grad  # (g,), (n, g, C)
        values = targets.to(Z.dtype).unsqueeze(1)  # (n, 1, C)
        if self.loss == "multilabel":
            loss = F.binary_cross_entropy_with_logits(Z, values.expand_as(Z), reduction="none")  # (n, g, C)
            return loss.sum(dim=(0, 2)), torch.sigmoid(Z) - values  # (g,), (n, g, C)
        difference = Z - values  # (n, g, C)
        return difference.square().sum(dim=(0, 2)), 2.0 * difference  # (g,), (n, g, C)

    def _denominator(self, rows: int) -> int:
        return rows if self.loss == "multiclass" else rows * self.outputs

    def accumulate(self, X: Tensor, targets: Tensor, start: int = 0, stop: int | None = None) -> Tensor:
        """Add the mean-loss gradient of groups start..stop over these rows to the gradient buffers.

        Returns the mean loss of each group. The rows are one batch's, so the step that follows is one optimizer
        step of a probe trained on that batch alone.
        """
        # X: (n, g, w); targets: (n,) or (n, C)
        stop = self.groups if stop is None else stop
        rows = X.shape[0]
        if rows == 0:
            return torch.zeros(stop - start, device=self.device)
        inputs = self._prepare(X, start, stop)  # (n, g, w)
        scale = 1.0 / self._denominator(rows)
        if self.settings.kind == "linear":
            weight, bias = self.params["weight"][start:stop], self.params["bias"][start:stop]
            Z = torch.einsum("ngw,gwc->ngc", inputs, weight) + bias  # (n, g, C)
            loss, dZ = self._loss_terms(Z, targets)  # (g,), (n, g, C)
            dZ = dZ * scale
            self.grads["weight"][start:stop] += torch.einsum("ngw,ngc->gwc", inputs, dZ)  # (g, w, C)
            self.grads["bias"][start:stop] += dZ.sum(dim=0)  # (g, C)
            return loss * scale
        weight1, bias1 = self.params["weight1"][start:stop], self.params["bias1"][start:stop]
        weight2, bias2 = self.params["weight2"][start:stop], self.params["bias2"][start:stop]
        pre = torch.einsum("ngw,gwm->ngm", inputs, weight1) + bias1  # (n, g, m)
        hidden = F.gelu(pre)  # (n, g, m)
        keep = 1.0 - self.settings.dropout
        dropped = (torch.rand_like(hidden) < keep).to(hidden.dtype) / keep if self.settings.dropout > 0 else None
        hidden_used = hidden * dropped if dropped is not None else hidden  # (n, g, m)
        Z = torch.einsum("ngm,gmc->ngc", hidden_used, weight2) + bias2  # (n, g, C)
        loss, dZ = self._loss_terms(Z, targets)  # (g,), (n, g, C)
        dZ = dZ * scale
        self.grads["weight2"][start:stop] += torch.einsum("ngm,ngc->gmc", hidden_used, dZ)  # (g, m, C)
        self.grads["bias2"][start:stop] += dZ.sum(dim=0)  # (g, C)
        d_hidden = torch.einsum("ngc,gmc->ngm", dZ, weight2)  # (n, g, m)
        if dropped is not None:
            d_hidden = d_hidden * dropped
        d_pre = d_hidden * _gelu_grad(pre)  # (n, g, m)
        self.grads["weight1"][start:stop] += torch.einsum("ngw,ngm->gwm", inputs, d_pre)  # (g, w, m)
        self.grads["bias1"][start:stop] += d_pre.sum(dim=0)  # (g, m)
        return loss * scale

    def loss_sums(self, X: Tensor, targets: Tensor, start: int = 0, stop: int | None = None, which: str = "current") -> Tensor:
        """Summed (not averaged) loss of groups start..stop, for validation totals across batches."""
        Z = self.logits(X, start, stop, which)  # (n, g, C)
        loss, _ = self._loss_terms(Z, targets)  # (g,)
        return loss

    def step(self) -> None:
        """One AdamW step of every group, with each group's own learning rate and weight decay, then clear grads."""
        self.steps += 1
        beta1, beta2 = self.settings.betas
        correction1 = 1.0 - beta1 ** self.steps
        correction2 = 1.0 - beta2 ** self.steps
        for name, param in self.params.items():
            grad = self.grads[name]
            shape = (-1,) + (1,) * (param.dim() - 1)
            rate = self.learning_rates.view(shape)  # (G, 1[, 1])
            decay = self.weight_decays.view(shape)  # (G, 1[, 1])
            param.mul_(1.0 - rate * decay)
            self.first_moment[name].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            self.second_moment[name].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denominator = (self.second_moment[name].sqrt() / math.sqrt(correction2)).add_(self.settings.eps)
            param.sub_(rate / correction1 * self.first_moment[name] / denominator)
            grad.zero_()

    # -- validation and early stopping -------------------------------------------------------------------------

    def observe(self, validation_loss: Tensor, patience: int, min_delta: float = 0.0) -> int:
        """Keep each improved group's weights as its best, count misses, stop groups that missed `patience` times.

        Returns how many groups are still training.
        """
        # validation_loss: (G,) mean loss of each group on the validation split
        improved = (validation_loss < self.best_loss - min_delta) & ~self.stopped  # (G,)
        self.best_loss = torch.where(improved, validation_loss, self.best_loss)
        self.best_step = torch.where(improved, torch.full_like(self.best_step, self.steps), self.best_step)
        for name, param in self.params.items():
            shape = (-1,) + (1,) * (param.dim() - 1)
            self.best[name] = torch.where(improved.view(shape), param, self.best[name])
        self.misses = torch.where(improved, torch.zeros_like(self.misses), self.misses + (~self.stopped).long())
        self.stopped |= self.misses >= patience
        return int((~self.stopped).sum().item())

    def select_settings(self, settings_count: int) -> tuple["ProbeBank", Tensor]:
        """Keep, for each probe, the setting of the grid with the lowest best validation loss.

        Groups are ordered setting-major (all probes of setting 0, then setting 1, ...). Returns the reduced bank,
        whose best and initial weights are the chosen setting's, and the chosen setting index of each probe.
        """
        probes = self.groups // settings_count
        assert probes * settings_count == self.groups
        losses = self.best_loss.view(settings_count, probes)  # (K, P)
        chosen = losses.argmin(dim=0)  # (P,)
        index = chosen * probes + torch.arange(probes, device=self.device)  # (P,)
        reduced = object.__new__(ProbeBank)
        reduced.__dict__.update(self.__dict__)
        reduced.names = [self.names[position] for position in index.tolist()]
        for attribute in ("params", "initial", "best", "grads", "first_moment", "second_moment"):
            setattr(reduced, attribute, {name: value[index] for name, value in getattr(self, attribute).items()})
        for attribute in ("learning_rates", "weight_decays", "mean", "inverse_std", "best_loss", "best_step", "misses", "stopped"):
            setattr(reduced, attribute, getattr(self, attribute)[index])
        return reduced, chosen

    def state(self, which: str = "best") -> dict[str, Tensor]:
        """Weights and standardization of every group, on the CPU, for saving."""
        saved = {f"{which}.{name}": value.detach().cpu() for name, value in self._weights(which).items()}
        saved["mean"] = self.mean.detach().cpu()
        saved["inverse_std"] = self.inverse_std.detach().cpu()
        saved["learning_rates"] = self.learning_rates.detach().cpu()
        saved["weight_decays"] = self.weight_decays.detach().cpu()
        saved["best_loss"] = self.best_loss.detach().cpu()
        saved["best_step"] = self.best_step.detach().cpu()
        return saved
