"""One probe head for every arm of the representation ablation.

Each branch of the input, one cached representation such as a max-pooled SAE codebook or a
mean and variance of hidden states, is clamped, layer-normalized, and projected to
`projection_size // k` for `k` branches, or to its own `branch_sizes` entry when the config
lists them. The projections are concatenated to `projection_size`. New training expands this
fused vector to 8192, reduces it to the smallest multiple of 256 covering the class count, and
applies LayerNorm and the output.

`branch_sizes` exists for a branch that carries far more than its neighbors, such as the
20,652-wide Atlas interaction profile: SAE and mean and variance keep 512 each, the profile gets
1024, and the fused vector is 2048. Their sum must be `projection_size`, so the trunk after the
projections is sized by one number whatever the split.
Saved residual trunks retain their blocks `h + Dropout(ReLU(Linear(h)))`. Only branch
projections differ between representation arms using the same saved topology.

The clamp exists for sparse features. After standardization a feature active in one protein
of ten thousand lands near a hundred standard deviations out, and unclamped it would dominate
the branch's LayerNorm.

`sngp` is the same head with a Gaussian process output and a spectral bound on every trunk
and projection Linear; see `sngp.py`.

Three variants serve the single-probe uncertainty experiment. `projected_linear` drops the
projections and the trunk: the clamped joined input is the feature the output Linear reads,
with no LayerNorm and no dropout. `projected_faithful` and `projected_beta_nll` keep the
projected trunk and add a variance output for regression. Faithful heteroscedastic regression
(Stirn et al. 2023) trains the mean by squared error alone and the variance by Gaussian
negative log likelihood on the detached trunk and the detached mean, so the mean learns what
a plain head learns. Beta-NLL (Seitzer et al. 2022) weights each row's Gaussian negative log
likelihood by its detached `variance ** beta`. `input_clip` None skips the clamp, for inputs
the standardizer has already clipped.

Shapes: `b` rows in a batch, `s` extra leading dimensions the caller's collator adds, `n`
flattened rows, `d_in` input width, `w` one slot's width of one branch, `m` slots, `k`
branches, `p` projection size, `f` the width the output layer reads, `c` outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .losses import compute_probe_loss, get_loss_fct, validate_label_mask
from .sngp import RandomFeatureGaussianProcess, bound_spectral_norm


# Probe type -> output layer. `sngp` also bounds every trunk and projection Linear, because a
# Gaussian process reads distance, and an unconstrained trunk is free to collapse distance.
PROJECTED_PROBE_OUTPUTS = {
    "projected": "linear",
    "sngp": "gaussian_process",
    "projected_linear": "linear",
    "projected_faithful": "faithful",
    "projected_beta_nll": "beta_nll",
}
# Probe type -> trunk. Every probe type not listed keeps the projected trunk.
PROJECTED_PROBE_TRUNKS = {"projected": "expanded_mlp", "projected_linear": "linear"}
OUTPUT_LAYERS = ("linear", "gaussian_process", "faithful", "beta_nll")
TRUNKS = ("projected", "expanded_mlp", "linear")
# Outputs that predict a Gaussian variance beside the mean: regression only.
VARIANCE_OUTPUTS = ("faithful", "beta_nll")
# The variance output is softplus(z) + this floor, so the negative log likelihood never divides
# by zero. Labels reach the head in their own units, so the floor is far below any real noise.
VARIANCE_FLOOR = 1e-6
# Task types the Gaussian process output supports: a Laplace posterior needs one softmax or one
# Gaussian likelihood per row.
GAUSSIAN_PROCESS_TASKS = ("singlelabel", "regression")
# Regression's noise variance is the training mean squared error, which is zero for a head
# that fits its training rows exactly, and the posterior precision divides by it. The floor
# is this fraction of Var(y_train), which caps the training signal-to-noise ratio at 1000
# (a training R^2 of 0.999), so it binds only on a near-exact fit, in any label units.
NOISE_VARIANCE_FLOOR_FRACTION = 1e-3
# The absolute minimum under that floor: the smallest normal float32, so the float32 buffer
# holding the noise variance stays positive for labels of any scale.
NOISE_VARIANCE_MINIMUM = float(torch.finfo(torch.float32).tiny)


@dataclass(frozen=True)
class BranchLayout:
    """How a probe input row is laid out: slots, each holding every branch in order.

    A single-sequence row is one slot. A pair row [A | B] is two, and a multi-column row one
    per column, because each protein's vector is itself the branches joined in spec order.
    A branch therefore reads its columns from every slot.
    """

    widths: tuple[int, ...]
    slots: int

    @classmethod
    def from_input(cls, input_size: int, widths: Sequence[int] | None) -> BranchLayout:
        """The layout of an `input_size`-wide row; no widths means one branch spanning it."""
        if widths is None:
            return cls((input_size,), 1)

        widths = tuple(int(width) for width in widths)
        if not widths or min(widths) < 1:
            raise ValueError(f"branch widths must be positive; got {widths}.")

        slot_width = sum(widths)
        if input_size % slot_width:
            raise ValueError(
                f"input_size {input_size} is not a whole number of slots of the branch widths "
                f"{widths}, which sum to {slot_width}."
            )

        return cls(widths, input_size // slot_width)

    def columns(self, branch: int) -> Tensor:
        """The row positions one branch reads, slot by slot."""
        slot_width = sum(self.widths)
        offset = sum(self.widths[:branch])
        starts = torch.arange(self.slots) * slot_width + offset  # (m,)
        within = torch.arange(self.widths[branch])  # (w,)
        return (starts.unsqueeze(-1) + within).reshape(-1)  # (m * w,)

    def gather(self, rows: Tensor, branch: int) -> Tensor:
        """One branch's columns of every row, slot by slot, in the order `columns` lists them.

        A view and a slice rather than an index tensor, so the probe keeps no buffer that a
        checkpoint would have to carry.
        """
        # rows: (n, m * sum(w))
        offset = sum(self.widths[:branch])
        width = self.widths[branch]  # w
        slots = rows.reshape(rows.shape[0], self.slots, sum(self.widths))  # (n, m, sum(w))
        return slots[:, :, offset:offset + width].reshape(rows.shape[0], self.slots * width)  # (n, m * w)


class ProjectedProbeConfig(PretrainedConfig):
    model_type = "projected_probe"

    def __init__(
        self,
        input_size: int = 768,
        projection_size: int = 1024,
        projected_hidden_size: int | None = None,
        n_layers: int = 1,
        dropout: float = 0.1,
        branch_widths: list[int] | None = None,
        branch_sizes: list[int] | None = None,
        input_clip: float | None = 10.0,
        output: str = "linear",
        spectral_bound: float = 0.95,
        gp_num_features: int = 1024,
        gp_lengthscale_squared: float = 1024.0,
        num_labels: int = 2,
        task_type: str = "singlelabel",
        trunk: str = "projected",
        beta: float = 0.5,
        **kwargs: object,
    ) -> None:
        # Transformers 5 saves the label count only as `id2label`, which `super().__init__`
        # restores from a saved config. The default here would otherwise overwrite it on load.
        if kwargs.get("id2label") and num_labels == 2:
            num_labels = len(kwargs["id2label"])
        super().__init__(**kwargs)
        self.input_size = input_size
        self.projection_size = projection_size
        self.projected_hidden_size = projected_hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.branch_widths = None if branch_widths is None else [int(width) for width in branch_widths]
        # None keeps the even split of an older saved head, which has no such field.
        self.branch_sizes = None if branch_sizes is None else [int(size) for size in branch_sizes]
        self.input_clip = input_clip
        self.output = output
        self.spectral_bound = spectral_bound
        self.gp_num_features = gp_num_features
        # The Gaussian process reads the trunk after its final LayerNorm, whose rows start with
        # squared norm projection_size, so two rows at correlation rho have kernel
        # exp(-projection_size (1 - rho) / gp_lengthscale_squared). The default 1024, at the
        # default width, gives uncorrelated rows e^-1 and the random features unit-variance
        # phases; sngp.py records how the scale moves in training and where training fails.
        self.gp_lengthscale_squared = gp_lengthscale_squared
        self.num_labels = num_labels
        self.task_type = task_type
        self.trunk = trunk
        # Beta-NLL's exponent on the detached variance: 0 is the Gaussian negative log
        # likelihood, 1 weights each row's gradient as squared error would.
        self.beta = beta


def gaussian_negative_log_likelihood(mean: Tensor, variance: Tensor, labels: Tensor) -> Tensor:
    """Each row's Gaussian negative log likelihood without its constant, `(log v + r^2 / v) / 2`."""
    # mean, variance, labels: (n,)
    return 0.5 * (variance.log() + (labels - mean) ** 2 / variance)  # (n,)


class ResidualBlock(nn.Module):
    """`h + Dropout(ReLU(Linear(h)))`."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        # h: (n, p)
        return h + self.dropout(torch.relu(self.linear(h)))  # (n, p)


class ExpandedProjectedBlock(nn.Module):
    """Expand fused features, then project to a class-aligned penultimate width."""

    def __init__(self, width: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.expand = nn.Linear(width, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reduce = nn.Linear(hidden_size, output_size)

    def forward(self, h: Tensor) -> Tensor:
        # h: (n, p); u is the expanded width and f the class-aligned width.
        expanded = self.dropout(torch.relu(self.expand(h)))  # (n, u)
        return self.dropout(torch.relu(self.reduce(expanded)))  # (n, f)


class ProjectedProbe(PreTrainedModel):
    """Per-branch projections into one trunk, with a linear or Gaussian process output."""

    config_class = ProjectedProbeConfig
    all_tied_weights_keys = {}

    @staticmethod
    def branch_projection_sizes(config: ProjectedProbeConfig, branch_count: int) -> list[int]:
        """Each branch's projected width: the config's `branch_sizes`, or `projection_size // k` for every branch."""
        if config.branch_sizes is None:
            if config.projection_size % branch_count:
                raise ValueError(
                    f"projection_size {config.projection_size} does not split evenly across "
                    f"{branch_count} branches."
                )

            return [config.projection_size // branch_count] * branch_count  # p / k each

        sizes = config.branch_sizes
        if len(sizes) != branch_count or min(sizes) < 1:
            raise ValueError(f"branch_sizes {sizes} must hold one positive width for each of the {branch_count} branches.")
        if sum(sizes) != config.projection_size:
            raise ValueError(
                f"branch_sizes {sizes} sum to {sum(sizes)}, but projection_size is {config.projection_size}; "
                f"the fused vector is the projections joined, so the two must agree."
            )

        return list(sizes)

    def __init__(self, config: ProjectedProbeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.num_labels = config.num_labels  # c
        self.input_clip = config.input_clip
        self.loss_fct = get_loss_fct(config.task_type)

        if config.output not in OUTPUT_LAYERS:
            raise ValueError(f"output must be one of {OUTPUT_LAYERS}; got {config.output!r}.")
        trunk = getattr(config, "trunk", "projected")
        if trunk not in TRUNKS:
            raise ValueError(f"trunk must be one of {TRUNKS}; got {trunk!r}.")
        gaussian_process = config.output == "gaussian_process"
        expanded_width = getattr(config, "projected_hidden_size", None)
        if expanded_width is not None:
            if type(expanded_width) is not int or expanded_width < 1:
                raise ValueError("projected_hidden_size must be a positive integer")
            if gaussian_process or trunk != "expanded_mlp":
                raise ValueError("expanded hidden layers require the projected trunk and a non-GP output")
        if gaussian_process and config.task_type not in GAUSSIAN_PROCESS_TASKS:
            raise ValueError(
                f"The Gaussian process output supports {GAUSSIAN_PROCESS_TASKS} tasks only, because "
                f"its Laplace posterior needs one softmax or one Gaussian likelihood per row; got "
                f"task_type {config.task_type!r}. Use probe_type 'projected' for this task."
            )
        if config.output in VARIANCE_OUTPUTS and (config.task_type != "regression" or config.num_labels != 1):
            raise ValueError(
                f"The {config.output!r} output predicts one Gaussian mean and variance, so it takes "
                f"regression with one label only; got task_type {config.task_type!r} with "
                f"{config.num_labels} labels."
            )
        if gaussian_process and trunk != "projected":
            raise ValueError("The Gaussian process output reads the projected trunk only.")

        self.trunk = trunk
        self.beta = float(getattr(config, "beta", 0.5))
        self.layout = BranchLayout.from_input(config.input_size, config.branch_widths)
        if trunk == "linear":
            # The output reads the clamped joined input: no projection, LayerNorm, or dropout.
            if config.branch_sizes is not None:
                raise ValueError("branch_sizes sizes the branch projections, which the linear trunk does not have.")
            self.branches = nn.ModuleList()
            self.blocks = nn.ModuleList()
            self.norm = None
            feature_size = config.input_size  # f = d_in
        else:
            branch_sizes = self.branch_projection_sizes(config, len(self.layout.widths))  # k entries
            self.branches = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(width * self.layout.slots), nn.Linear(width * self.layout.slots, branch_size))
                for width, branch_size in zip(self.layout.widths, branch_sizes, strict=True)
            )
            if trunk == "expanded_mlp":
                if config.n_layers != 1:
                    raise ValueError("The expanded projected head has exactly one two-layer MLP")
                feature_size = ((config.num_labels + 255) // 256) * 256  # f
                self.blocks = nn.ModuleList([ExpandedProjectedBlock(
                    config.projection_size, expanded_width or 8192, feature_size, config.dropout,
                )])
            else:
                self.blocks = nn.ModuleList(
                    ResidualBlock(config.projection_size, config.dropout) for _ in range(config.n_layers)
                )
                feature_size = config.projection_size  # f = p
            self.norm = nn.LayerNorm(feature_size)

        self.variance_output = None
        if config.output in VARIANCE_OUTPUTS:
            self.output = nn.Linear(feature_size, 1)
            self.variance_output = nn.Linear(feature_size, 1)
        elif gaussian_process:
            self.output = RandomFeatureGaussianProcess(
                config.projection_size,
                config.num_labels,
                num_features=config.gp_num_features,
                lengthscale_squared=config.gp_lengthscale_squared,
            )
            for branch in self.branches:
                bound_spectral_norm(branch[1], config.spectral_bound)
            for block in self.blocks:
                bound_spectral_norm(block.linear, config.spectral_bound)
        else:
            self.output = nn.Linear(feature_size, config.num_labels)

    @property
    def gaussian_process(self) -> RandomFeatureGaussianProcess | None:
        return self.output if isinstance(self.output, RandomFeatureGaussianProcess) else None

    @property
    def predicts_variance(self) -> bool:
        """Whether `predict_with_variance` returns a variance: the Gaussian process or a variance output."""
        return self.gaussian_process is not None or self.variance_output is not None

    def _clamp(self, rows: Tensor) -> Tensor:
        # rows: (n, ...)
        if self.input_clip is None:
            return rows  # (n, ...)
        return rows.clamp(-self.input_clip, self.input_clip)  # (n, ...)

    def trunk_features(self, embeddings: Tensor) -> Tensor:
        """The features the output layer reads: the layer-normalized trunk, or the clamped input."""
        # embeddings: (n, d_in)
        if self.trunk == "linear":
            return self._clamp(embeddings.to(self.output.weight.dtype))  # (n, f = d_in)

        embeddings = embeddings.to(self.norm.weight.dtype)  # (n, d_in)
        projections = []
        for branch_index, branch in enumerate(self.branches):
            branch_input = self.layout.gather(embeddings, branch_index)  # (n, m * w)
            branch_input = self._clamp(branch_input)  # (n, m * w)
            projections.append(branch(branch_input))  # (n, p / k)

        h = torch.cat(projections, dim=-1)  # (n, p)
        for block in self.blocks:
            h = block(h)  # (n, f); f=p for the legacy residual trunk.
        return self.norm(h)  # (n, f)

    def _variance(self, h: Tensor) -> Tensor:
        """The variance output's prediction, `softplus(z) + VARIANCE_FLOOR`.

        Faithful reads the detached features, so the variance's gradient never reaches the trunk.
        """
        # h: (n, f)
        if self.config.output == "faithful":
            h = h.detach()  # (n, f)
        return F.softplus(self.variance_output(h)).reshape(-1) + VARIANCE_FLOOR  # (n,)

    def _variance_loss(self, mean: Tensor, variance: Tensor, labels: Tensor) -> Tensor:
        """The training loss of a head with a variance output, averaged over rows."""
        # mean, variance: (n,); labels: (n,)
        if self.config.output == "faithful":
            squared_error = (mean - labels) ** 2  # (n,)
            likelihood = gaussian_negative_log_likelihood(mean.detach(), variance, labels)  # (n,)
            return squared_error.mean() + likelihood.mean()  # ()

        likelihood = gaussian_negative_log_likelihood(mean, variance, labels)  # (n,)
        weight = variance.detach() ** self.beta  # (n,)
        return (weight * likelihood).mean()  # ()

    def _output_logits(self, h: Tensor) -> Tensor:
        # h: (n, f)
        gaussian_process = self.gaussian_process
        if gaussian_process is None:
            return self.output(h)  # (n, c)

        phi = gaussian_process.features(h)  # (n, D)
        logits = gaussian_process.beta(phi)  # (n, c), MAP
        # MAP logits until fit_posterior runs, so early stopping reads the network it trains.
        # Regression keeps the MAP mean after it too; only its variance comes from the posterior.
        if not bool(gaussian_process.posterior_ready) or self.task_type == "regression":
            return logits  # (n, c)

        return gaussian_process.mean_field_logits(logits, gaussian_process.variance(phi))  # (n, c)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> SequenceClassifierOutput:
        # embeddings: (b, *s, d_in); attention_mask is unused.
        # labels, label_mask: (b,), (b, 1), or (b, c), depending on the task.
        # attention_mask: (b, l), unused
        if label_mask is not None:
            if labels is None:
                raise ValueError("label_mask requires labels")
            validate_label_mask(labels, label_mask)
        leading_shape = embeddings.shape[:-1]  # (b, *s)
        h = self.trunk_features(embeddings.reshape(-1, embeddings.shape[-1]))  # (n, f)
        logits = self._output_logits(h).reshape(*leading_shape, self.num_labels)  # (b, *s, c)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()  # (b, *s, c)

        loss = None
        if labels is not None and self.variance_output is not None:
            mean = logits.reshape(-1)  # (n,), c = 1
            variance = self._variance(h)  # (n,)
            truth = labels.reshape(-1).to(mean.dtype)  # (n,)
            if label_mask is not None:
                observed = label_mask.reshape(-1)  # (n,)
                mean, variance, truth = mean[observed], variance[observed], truth[observed]  # each (m,)
            loss = self._variance_loss(mean, variance, truth) if mean.numel() else mean.sum() + variance.sum()  # ()
        elif labels is not None:
            loss = compute_probe_loss(self.loss_fct, logits, labels, self.task_type, self.num_labels, label_mask)  # ()

        # logits: (b, *s, c); loss: () or None.
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)

    @torch.no_grad()
    def predict_with_variance(self, embeddings: Tensor) -> tuple[Tensor, Tensor | None]:
        """Predictions and, for the Gaussian process output, their variance.

        Leading dimensions are flattened to one row each. The linear output returns what
        `forward` does and no variance. The Gaussian process returns mean-field logits and the
        latent variance of the logit for classification, and the mean with the predictive
        variance `sigma_hat^2 + phi^T Sigma phi` for regression. A variance output returns the
        mean and its predicted variance. Call it in eval mode.
        """
        # embeddings: (b, *s, d_in)
        h = self.trunk_features(embeddings.reshape(-1, embeddings.shape[-1]))  # (n, f)
        if self.variance_output is not None:
            return self.output(h), self._variance(h)  # (n, 1), (n,)

        gaussian_process = self.gaussian_process
        if gaussian_process is None:
            predictions = self.output(h)  # (n, c)
            if self.task_type == "sigmoid_regression":
                predictions = predictions.sigmoid()  # (n, c)
            return predictions, None  # (n, c), None

        if not bool(gaussian_process.posterior_ready):
            raise RuntimeError("fit_posterior has not run, so the Gaussian process has no variance yet.")

        phi = gaussian_process.features(h)  # (n, D)
        logits = gaussian_process.beta(phi)  # (n, c)
        variance = gaussian_process.variance(phi)  # (n,)
        if self.task_type == "regression":
            return logits, variance + gaussian_process.noise_variance  # (n, 1), (n,)

        return gaussian_process.mean_field_logits(logits, variance), variance  # (n, c), (n,)

    @torch.no_grad()
    def fit_posterior(self, batches: Iterable[Mapping[str, Tensor]]) -> None:
        """Fit the Laplace posterior of the Gaussian process output in one pass.

        Run after training, with the weights early stopping restored. Classification weights
        each row by `p_hat (1 - p_hat)` of its top class under the MAP softmax. Regression
        uses `Sigma = (I / tau^2 + Phi^T Phi / sigma_hat^2)^-1` with `tau^2 = Var(y_train)`
        and `sigma_hat^2` the training mean squared error, floored at
        `NOISE_VARIANCE_FLOOR_FRACTION * tau^2` and `NOISE_VARIANCE_MINIMUM`. A linear output
        has no posterior, so this returns without reading a batch.
        """
        # batches: (b, *s, d_in) embeddings and labels per batch
        gaussian_process = self.gaussian_process
        if gaussian_process is None:
            return

        was_training = self.training
        self.eval()
        device = self.norm.weight.device
        gaussian_process.reset_precision()
        regression = self.task_type == "regression"
        rows = 0
        squared_error = 0.0
        label_sum = 0.0
        label_square_sum = 0.0
        for batch in batches:
            embeddings = batch["embeddings"].to(device)  # (b, *s, d_in)
            labels = batch["labels"].to(device).reshape(-1) if regression or "label_mask" in batch else None  # (n,) or None
            if "label_mask" in batch:
                label_mask = batch["label_mask"].to(device)  # labels' original shape
                validate_label_mask(batch["labels"].to(device), label_mask)
                observed = label_mask.reshape(-1)  # (n,)
                embeddings = embeddings.reshape(-1, embeddings.shape[-1])[observed]  # (m, d_in)
                labels = labels[observed]  # (m,)
                if not labels.numel():
                    continue
            phi = gaussian_process.features(
                self.trunk_features(embeddings.reshape(-1, embeddings.shape[-1]))
            )  # (n, D)
            logits = gaussian_process.beta(phi)  # (n, c), MAP
            if regression:
                labels = labels.to(torch.float64)  # (n,)
                residuals = logits.reshape(-1).to(torch.float64) - labels  # (n,)
                gaussian_process.accumulate_precision(phi, torch.ones_like(labels))
                squared_error += float((residuals ** 2).sum())
                label_sum += float(labels.sum())
                label_square_sum += float((labels ** 2).sum())
            else:
                top_probability = logits.softmax(dim=-1).amax(dim=-1)  # (n,)
                gaussian_process.accumulate_precision(phi, top_probability * (1.0 - top_probability))
            rows += phi.shape[0]

        if rows == 0:
            raise ValueError("fit_posterior received no training rows.")

        if regression:
            label_mean = label_sum / rows
            label_variance = label_square_sum / rows - label_mean ** 2
            if label_variance <= 0.0:
                raise ValueError("The training labels are constant, so the prior variance Var(y) is zero.")
            noise_variance = max(
                squared_error / rows, NOISE_VARIANCE_FLOOR_FRACTION * label_variance, NOISE_VARIANCE_MINIMUM
            )
            gaussian_process.finalize_posterior(prior_precision=1.0 / label_variance, noise_variance=noise_variance)
        else:
            gaussian_process.finalize_posterior()
        self.train(was_training)
