"""Load a packaged uncertainty head and predict with it.

A package is what `python -m protify.uncertainty package` writes from one finished cell: the
shipped heads, the embedding standardizer, and the calibration and conformal quantiles the
cell measured. `load_package` reads one back, and `PackagedHead.predict` takes the joined pooled
features a cell's collator builds, before standardization, and returns the method's prediction.

Prediction runs three phases, each with the cell's own arithmetic:
1. `seed_predictions`: each shipped head reads the standardized rows through
   `collect_predictions`, as it did when the cell saved its predictions. SNGP reports
   mean-field logits and its predictive variance, and the plain head its MC dropout mean.
2. `calibrated_predictive`: the per-seed predictions become the method's, through the
   functions `build_methods` itself uses, with the temperature or variance map the cell fitted
   on V_fit applied rather than fitted again.
3. `conformal_prediction`: the quantiles fitted on V_cal become sets or intervals.

A classification package also gives each row its answer, the confidence that the answer is
right, and that answer's conformal p-value, and `PackagedHead.select` chooses which rows to
answer by conformal selection (`answering.py`). A package with a `correctness/` directory also
gives each answer its reliability, the correctness model's probability that it is right, read
from the head's latent space (`correctness.py`); its answer rule then ranks by reliability.

`PackagedHead.decide` is the package's whole policy for a request (`applicability.py`): it
abstains on rows far from the calibration population or in a stratum V_cal cannot support,
selects among the rest, and leaves every row unassessed when the package cannot vouch for an
error target. `select` alone knows nothing of novelty.

MC dropout draws its passes from torch's generator seeded with the shipped seed, so a call
repeats itself exactly, but it does not repeat the passes the cell drew for its report.

Shapes: `n` rows, `W` one protein's joined width, `d_in` the head's input width, `c` classes
or labels, `b` rows in a batch.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from torch.utils.data import TensorDataset

from foundry.eval.conformal import Interval
from .answering import answer_confidence, conformal_p_values, decode, select_answers
from .applicability import (
    NOVELTY_SIGNAL,
    Decisions,
    applicability_status,
    decide,
    unassessed,
    unavailable_applicability,
    validate_applicability,
    validate_policy,
)
from .conformal_sets import class_scores, regression_interval
from .correctness import ShippedCorrectness
# The run builders are the report's own, so a package's arithmetic is exactly the report's.
from .methods import (
    Predictive,
    calibrated_probabilities,
    calibrated_variance,
    ensemble_run,
    head_run,
    mc_dropout_run,
    classification_predictive,
    probabilities,
    regression_predictive,
)
from .metrics import CLASSIFICATION_KINDS
from .observation_contract import partial_answer_unavailable, validate_observation_contract
from .predictions import SplitPredictions, collect_predictions
from .report import read_json
from ..data.dataset_classes import EmbeddingStandardizer
from ..data.declared_dataset import checked_digest, validate_declared_manifest
from ..data.pair_features import pair_vector_halves
from ..probes.projected_probe import ProjectedProbe


PACKAGE_SCHEME = "validation_carve"
CALIBRATION_RECORD = "calibration.json"
FEATURES_RECORD = "features.json"
METRICS_RECORD = "metrics.json"
ORIGIN_RECORD = "origin.json"
MODEL_CARD = "README.md"
STANDARDIZER_FILE = "standardizer.npz"
HEADS_DIRECTORY = "heads"
CORRECTNESS_DIRECTORY = "correctness"
DEFAULT_BATCH_SIZE = 256
# Method name -> (the prediction it starts from, whether it is calibrated on V_fit). The
# score-only methods are absent: they rank rows by another score but predict as their head.
METHOD_RECIPES = {
    "plain": ("head", False),
    "sngp": ("head", False),
    "temperature": ("head", True),
    "sngp_temperature": ("head", True),
    "mc_dropout": ("mc_dropout", False),
    "mc_dropout_temperature": ("mc_dropout", True),
    "ensemble": ("ensemble", False),
    "ensemble_temperature": ("ensemble", True),
    "sngp_ensemble": ("ensemble", False),
    "sngp_ensemble_temperature": ("ensemble", True),
}
# The one pseudo-split a prediction runs on, named for the methods.py functions that read splits.
ROWS = "rows"


def calibration_kind(predictive: str, calibrated: bool, kind: str) -> str | None:
    """How a calibrated method maps its prediction, as `build_methods` fits it.

    The head divides its own logits by a temperature. MC dropout and the ensembles keep only
    mean probabilities, so theirs divides the log of those, or their log odds for multilabel.
    Regression maps the variance to `sigma^2 = a v + b` whatever the prediction.
    """
    if not calibrated:
        return None
    if kind == "regression":
        return "variance_map"
    return "logit_temperature" if predictive == "head" else "probability_temperature"


@dataclass(frozen=True)
class Prediction:
    """What a packaged method says about each row.

    Binary, multiclass, and multilabel tasks fill `probs`, after the calibration; regression
    fills `mean` and, for a method with a spread, `sigma`. `uncertainty` is the method's score,
    higher for less certain rows, and None for a regression method without a spread. `sets`
    maps each set score the package ships (`lac`) and a level to each row's class membership, for
    binary and multiclass; `intervals` maps a level to each row's bounds, for regression.

    A classification package also fills `answers`, each row's top class or label set;
    `confidence`, the calibrated probability that the answer is right; and `p_values`, each
    answer's conformal p-value against being wrong, which `PackagedHead.select` reads. They are
    None for regression and for a package written before the answer rule.

    A package that ships a correctness model also fills `reliability`, its probability that each
    answer is right, and `signals`, the latent signals it read; the p-values are then reliability's.
    With partial classification calibration truth, answers remain point predictions while
    confidence, reliability, and p-values are unavailable. `observation_contract` states this
    limitation and the measured support. Selection abstains for all these rows.
    """

    probs: np.ndarray | None = None  # (n, c)
    mean: np.ndarray | None = None  # (n,)
    sigma: np.ndarray | None = None  # (n,)
    uncertainty: np.ndarray | None = None  # (n,)
    sets: Mapping[str, Mapping[float, np.ndarray]] = field(default_factory=dict)  # score -> level -> (n, c) bool
    intervals: Mapping[float, Interval] = field(default_factory=dict)  # level -> (n,) bounds
    answers: np.ndarray | None = None  # (n,) class, or (n, c) bool label set
    confidence: np.ndarray | None = None  # (n,)
    p_values: np.ndarray | None = None  # (n,)
    reliability: np.ndarray | None = None  # (n,)
    signals: Mapping[str, np.ndarray] | None = None  # name -> (n,)
    observation_contract: Mapping[str, Any] | None = None  # calibration support and unavailable claims


def calibrated_predictive(
    seed_predictions: Mapping[int, SplitPredictions],
    predictive: str,
    calibration: str | None,
    fitted: Mapping[str, float],
    kind: str,
) -> Predictive:
    """The method's prediction from its shipped seeds' head outputs, with its fitted calibration.

    Every step is the one `build_methods` takes, fed stored parameters instead of fitting them,
    and an ensemble averages its seeds in ascending order, as `build_methods` does.
    """
    seeds = sorted(seed_predictions)
    head_runs = tuple(head_run(seed, {ROWS: seed_predictions[seed]}, kind) for seed in seeds)
    if predictive == "ensemble":
        base = ensemble_run(head_runs, kind).splits[ROWS]
    elif len(seeds) != 1:
        raise ValueError(f"a {predictive!r} method ships one seed; got {seeds}.")
    elif predictive == "mc_dropout":
        base = mc_dropout_run(seeds[0], {ROWS: seed_predictions[seeds[0]]}, kind).splits[ROWS]
    else:
        base = head_runs[0].splits[ROWS]

    if calibration is None:
        return base
    if calibration == "variance_map":
        variance = calibrated_variance(base.variance, fitted["a"], fitted["b"], base.mean.shape[0])  # (n,)
        return regression_predictive(base.mean, variance)
    if calibration == "logit_temperature":
        outputs = seed_predictions[seeds[0]].outputs  # (n, c)
        return classification_predictive(probabilities(outputs, kind, temperature=fitted["temperature"]), kind)
    if calibration == "probability_temperature":
        return classification_predictive(calibrated_probabilities(base.probs, kind, fitted["temperature"]), kind)
    raise ValueError(f"unknown calibration {calibration!r}.")


def conformal_prediction(
    predictive: Predictive, conformal: Mapping[str, Any] | None, kind: str
) -> tuple[dict[str, dict[float, np.ndarray]], dict[float, Interval]]:
    """Conformal sets for binary and multiclass, or intervals for regression, from V_cal's quantiles.

    A class joins a set when its score is at most the level's quantile, as `prediction_sets`
    decides; an interval is `regression_interval`'s. Multilabel has neither.
    """
    if conformal is None:
        return {}, {}  # (sets, intervals): two empty dicts when there is no conformal record

    if kind in CLASSIFICATION_KINDS:
        sets: dict[str, dict[float, np.ndarray]] = {}
        for score in conformal["set_scores"]:
            scores = class_scores(predictive.probs, score)  # (n, c)
            sets[score] = {entry["level"]: scores <= entry[score] for entry in conformal["quantiles"]}  # level -> (n, c)
        return sets, {}  # (sets, intervals): (n, c) bool per score and level, then an empty dict

    has_sigma = predictive.variance is not None
    if has_sigma != (conformal["spread"] == "sigma"):
        raise ValueError(f"the quantiles were fitted with spread {conformal['spread']!r}, which this prediction lacks or adds.")
    sigma = np.sqrt(predictive.variance) if has_sigma else None  # (n,) or None
    intervals = {
        entry["level"]: regression_interval(predictive.mean, sigma, entry["quantile"], conformal["spread_floor"], entry["level"])
        for entry in conformal["quantiles"]
    }
    return {}, intervals  # (sets, intervals): an empty dict, then each level's interval with (n,) bounds


def _load_standardizer(path: Path) -> EmbeddingStandardizer:
    with np.load(path, allow_pickle=False) as archive:
        return EmbeddingStandardizer(archive["mean"], archive["scale"])  # (d_in,) each


def _collate_rows(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # collect_predictions reads labels with every batch; these rows carry a placeholder it
    # returns untouched and nothing reads.
    # batch: (d_in,) and () per row, b rows
    return {
        "embeddings": torch.stack([row for row, _ in batch]),  # (b, d_in)
        "labels": torch.stack([label for _, label in batch]),  # (b,)
    }


@contextmanager
def _seeded_generator(seed: int, device: torch.device) -> Iterator[None]:
    """Torch's generator seeded for MC dropout passes, and restored for the caller afterwards."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        yield


@dataclass(frozen=True)
class PackagedHead:
    """A package's shipped heads, standardizer, and calibration, ready to predict."""

    path: str
    calibration: Mapping[str, Any]
    features: Mapping[str, Any]
    heads: Mapping[int, ProjectedProbe]
    standardizer: EmbeddingStandardizer | None
    correctness: ShippedCorrectness | None = None

    @property
    def method(self) -> str:
        return self.calibration["method"]

    @property
    def kind(self) -> str:
        return self.calibration["task_type"]

    @property
    def target_contract(self) -> Mapping[str, Any] | None:
        """Frozen output labels, units, and inverse transform; absent in legacy packages."""
        return self.calibration.get("target_contract")

    @property
    def answer_rule(self) -> Mapping[str, Any] | None:
        """The conformal selection record, or None for regression and older packages."""
        return self.calibration.get("answer_rule")

    @property
    def observation_contract(self) -> Mapping[str, Any] | None:
        """Observed-target calibration scope; partial truth cannot identify P(right)."""
        return self.calibration.get("observation_contract")

    @property
    def applicability(self) -> Mapping[str, Any]:
        """The calibrated novelty cutoff and support table, or the reason the package has none."""
        record = self.calibration.get("applicability")
        return unavailable_applicability("package_predates_applicability") if record is None else record

    @property
    def decision_policy(self) -> Mapping[str, Any] | None:
        """The `answer_or_abstain_v1` record `decide` follows, or None for regression and older packages."""
        return self.calibration.get("decision_policy")

    @property
    def predictive(self) -> str:
        """The prediction the method starts from: `head`, `mc_dropout`, or `ensemble`."""
        return self.calibration["predictive"]

    @property
    def fixed_width_interval(self) -> bool:
        """Whether a regression package's conformal interval has one width for every row.

        A plain head has no spread, and a temperature-scaled one takes its variance map's constant as every row's sigma, so
        `mean +- quantile * (sigma + floor)` is the same width everywhere: ordinary split conformal on the absolute residual.
        SNGP, MC dropout, and ensembles carry a spread per row, so their intervals follow it. Classification has no interval.
        """
        return self.kind == "regression" and self.predictive == "head" and not self.method.startswith("sngp")

    def model_input(self, features: np.ndarray, partner_features: np.ndarray | None = None) -> np.ndarray:
        """The standardized rows the heads read, `(n, d_in)`.

        `features` is each protein's joined pooled vector in the branch order `features.json`
        lists, as `JoinedBranchCollator` densifies it. A pair task also takes the second
        protein's, and joins the two by the cell's pair rule before standardizing, as the
        collator does.
        """
        # features: (n, W), partner_features: (n, W) or None
        width = self.features["width"]
        rows = np.asarray(features, dtype=np.float32)  # (n, W)
        if rows.ndim != 2 or rows.shape[1] != width:
            raise ValueError(f"features must be (rows, {width}), one protein's joined branches; got {rows.shape}.")

        if self.features["pair"]:
            if partner_features is None:
                raise ValueError("this package scores protein pairs; pass the second protein's features too.")
            partner = np.asarray(partner_features, dtype=np.float32)  # (n, W)
            if partner.shape != rows.shape:
                raise ValueError(f"partner features must match the features' shape {rows.shape}; got {partner.shape}.")
            half_a, half_b = pair_vector_halves(rows, partner, self.features["pair_feature_mode"])  # two of (n, W)
            rows = np.concatenate([half_a, half_b], axis=1)  # (n, 2W)
        elif partner_features is not None:
            raise ValueError("this package scores single proteins; it takes no partner features.")

        if self.standardizer is not None:
            rows = self.standardizer.transform_numpy(rows)  # (n, d_in)
        return np.ascontiguousarray(rows)  # (n, d_in)

    def seed_predictions(
        self,
        features: np.ndarray,
        partner_features: np.ndarray | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | torch.device = "cpu",
    ) -> dict[int, SplitPredictions]:
        """Each shipped head's outputs on the rows, as the cell saved them for its splits."""
        # features: (n, W), partner_features: (n, W) or None
        device = torch.device(device)
        inputs = torch.from_numpy(self.model_input(features, partner_features))  # (n, d_in)
        rows = TensorDataset(inputs, torch.zeros(inputs.shape[0]))
        mc_samples = self.calibration["mc_dropout_samples"] if self.predictive == "mc_dropout" else 0

        predictions = {}
        for seed, head in self.heads.items():
            with _seeded_generator(seed, device):
                predictions[seed], _ = collect_predictions(
                    head, rows, _collate_rows, task_kind=self.kind, batch_size=batch_size, device=device, mc_samples=mc_samples
                )
        return predictions

    def predict(
        self,
        features: np.ndarray,
        partner_features: np.ndarray | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | torch.device = "cpu",
    ) -> Prediction:
        """The method's calibrated prediction, uncertainty, and conformal sets or intervals per row."""
        # features: (n, W), partner_features: (n, W) or None
        per_seed = self.seed_predictions(features, partner_features, batch_size=batch_size, device=device)
        prediction = calibrated_predictive(
            per_seed, self.predictive, self.calibration["calibration"], self.calibration["fitted"], self.kind
        )
        sets, intervals = conformal_prediction(prediction, self.calibration["conformal"], self.kind)
        answers = confidence = p_values = reliability = signals = None
        rule = self.answer_rule
        if partial_answer_unavailable(self.observation_contract):
            answers = decode(prediction.probs, self.kind, False)  # (n,) or (n, c), point predictions only
        if rule is not None:
            at_least_one_label = bool(rule["at_least_one_label"])
            answers = decode(prediction.probs, self.kind, at_least_one_label)  # (n,) or (n, c)
            confidence = answer_confidence(prediction.probs, answers, self.kind, at_least_one_label)  # (n,)
            if self.correctness is not None:
                # A correctness package ships one head, the one its reference was read through.
                (head,) = self.heads.values()
                signals = self.correctness.signals(head, self.model_input(features, partner_features), answers, batch_size)
                reliability = self.correctness.probability(confidence, signals)  # (n,)
            p_values = conformal_p_values(reliability if rule.get("score") == "correctness" else confidence, rule)  # (n,)
        return Prediction(
            probs=prediction.probs,
            mean=prediction.mean,
            sigma=None if prediction.variance is None else np.sqrt(prediction.variance),
            uncertainty=prediction.uncertainty,
            sets=sets,
            intervals=intervals,
            answers=answers,
            confidence=confidence,
            p_values=p_values,
            reliability=reliability,
            signals=signals,
            observation_contract=self.observation_contract,
        )

    def select(self, prediction: Prediction, target: float | None = None) -> np.ndarray:
        """Which of the prediction's rows to answer, by conformal selection at `target`, `(n,)` bool.

        `target` is the expected share of wrong answers allowed among the rows answered; by
        default half of V_cal's error rate. The rows are selected together, so pass every row
        that is to be answered or not at once.
        """
        rule = self.answer_rule
        if partial_answer_unavailable(self.observation_contract):
            if prediction.answers is None:
                raise ValueError("partial-label selection requires point predictions")
            # Reuse the public target validation even though no row can be selected.
            if target is not None:
                select_answers(np.empty(0), target)
            return np.zeros(len(prediction.answers), dtype=bool)  # (n,)
        if rule is None or prediction.p_values is None:
            raise ValueError(f"{self.path} has no answer rule: regression, or packaged before the rule existed.")
        return select_answers(prediction.p_values, rule["default_target"] if target is None else target)  # (n,)

    def novelty(self, prediction: Prediction) -> np.ndarray | None:
        """Each row's `SUPPORTED` or `NOVEL` against the package's calibrated cutoff, `(n,)`, or None when it has none."""
        record = self.applicability
        if record["status"] != "calibrated":
            return None
        if prediction.signals is None:
            raise ValueError(f"{self.path} has a novelty cutoff, so the prediction must carry the latent signals it reads.")
        return applicability_status(record, prediction.signals[NOVELTY_SIGNAL])  # (n,)

    def decide(self, prediction: Prediction) -> Decisions:
        """Answer, abstain from, or leave unassessed each of the prediction's rows, under the package's decision policy.

        The rows are decided together, as `select` chooses them, so pass every row of one request at once. The target is the
        policy's and cannot be changed here, because the policy's digest names it. A package that cannot vouch for an error
        target, or was written before its policy, leaves every row unassessed and never turns that into an answer.
        """
        count = int((prediction.probs if prediction.probs is not None else prediction.mean).shape[0])
        if self.kind == "regression":
            return unassessed(count, "regression_has_no_answer_rule")
        if partial_answer_unavailable(self.observation_contract):
            return unassessed(count, "partial_calibration_truth")
        rule, policy = self.answer_rule, self.decision_policy
        if rule is None:
            return unassessed(count, "no_answer_rule")
        if policy is None:
            return unassessed(count, "no_decision_policy")
        score = prediction.reliability if rule["score"] == "correctness" else prediction.confidence  # (n,)
        return decide(policy, self.applicability, status=self.novelty(prediction), score=score, p_values=prediction.p_values)


def check_decision_records(calibration: Mapping[str, Any], correctness: ShippedCorrectness | None) -> None:
    """Refuse a novelty record or decision policy that disagrees with the package's answer rule, reference, or each other.

    A package written before either record existed carries neither and passes; one of the two alone does not.
    """
    rule = calibration.get("answer_rule")
    record, policy = calibration.get("applicability"), calibration.get("decision_policy")
    if record is None:
        if policy is not None:
            raise ValueError("a decision policy needs the applicability record it pins")
        return

    validate_applicability(record, calibration["rows"]["v_cal"], None if rule is None else rule["score"])
    if record["status"] == "calibrated":
        if rule is None or correctness is None or record["reference_sha256"] != correctness.reference.content_sha256():
            raise ValueError("packaged novelty cutoff does not belong to the package's latent reference")
    else:
        reason = record["reason_code"]
        if reason == "package_predates_applicability":
            raise ValueError("a written applicability record cannot say the package predates it")
        if (reason == "no_latent_reference" and correctness is not None) or (reason == "insufficient_calibration_rows" and correctness is None):
            raise ValueError("packaged applicability contradicts whether the package ships a latent reference")
    if (rule is None) != (policy is None):
        raise ValueError("a package ships a decision policy exactly when it ships an answer rule")
    if policy is not None:
        validate_policy(policy, rule, record)


def load_package(path: str | os.PathLike[str]) -> PackagedHead:
    """Read a package directory: its records, its standardizer, and every shipped head."""
    root = Path(path)
    calibration = read_json(str(root / CALIBRATION_RECORD))
    features = read_json(str(root / FEATURES_RECORD))
    if calibration.get("scheme") != PACKAGE_SCHEME:
        raise ValueError(f"{root} is not an uncertainty package of scheme {PACKAGE_SCHEME!r}.")
    target = calibration.get("target_contract")
    origin_path = root / ORIGIN_RECORD
    origin = read_json(str(origin_path)) if origin_path.exists() else {}
    release = origin.get("dataset", {}).get("release")
    observations = calibration.get("observation_contract")
    if observations != origin.get("observation_contract"):
        raise ValueError("packaged observation contract differs from its origin record")
    if observations is not None:
        validate_observation_contract(observations, calibration["task_type"], calibration["num_labels"])
        if any(observations["support"][role]["rows"] != calibration["rows"][role] for role in ("v_fit", "v_cal")):
            raise ValueError("packaged observation counts differ from calibration rows")
        if partial_answer_unavailable(observations) and (calibration.get("answer_rule") is not None
                or calibration.get("correctness") is not None or (root / CORRECTNESS_DIRECTORY).exists()):
            raise ValueError("partial calibration truth cannot provide whole-answer confidence or selection")
    if (target is None) != (release is None):
        raise ValueError("packaged target contract and dataset release must both be present")
    if target is not None:
        validate_declared_manifest({key: value for key, value in release.items() if key != "manifest_sha256"})
        checked_digest(release["manifest_sha256"])
        expected_target = {name: release[name] for name in ("manifest_sha256", "endpoint", "task_type", "units", "target_transform", "label_vocabulary")}
        if target != expected_target or target["task_type"] != calibration["task_type"]:
            raise ValueError("packaged target contract differs from its frozen dataset release")
        kind = target["task_type"]
        width = 2 if kind == "binary" else 1 if kind == "regression" else len(target["label_vocabulary"])
        if width != calibration["num_labels"]:
            raise ValueError("packaged output width differs from the frozen vocabulary")
    recipe = METHOD_RECIPES.get(calibration["method"])
    expected = None if recipe is None else (recipe[0], calibration_kind(*recipe, calibration["task_type"]))
    if expected != (calibration["predictive"], calibration["calibration"]):
        raise ValueError(
            f"{root} ships method {calibration['method']!r} as {calibration['predictive']!r} with "
            f"{calibration['calibration']!r}, which this Protify does not predict that method with."
        )

    heads = {}
    for seed in calibration["seeds"]:
        head = ProjectedProbe.from_pretrained(root / HEADS_DIRECTORY / f"seed_{seed}").eval()
        if head.config.input_size != features["input_size"] or head.num_labels != calibration["num_labels"]:
            raise ValueError(
                f"{root}'s head for seed {seed} reads {head.config.input_size} columns into {head.num_labels} "
                f"outputs, but the package describes {features['input_size']} and {calibration['num_labels']}."
            )
        heads[seed] = head

    standardizer = None if features["standardizer"] is None else _load_standardizer(root / features["standardizer"])
    correctness = None
    if (root / CORRECTNESS_DIRECTORY).is_dir():
        if len(heads) != 1:
            raise ValueError(f"{root} ships a correctness model beside {len(heads)} heads; it reads one head's latent space.")
        correctness = ShippedCorrectness.load(root / CORRECTNESS_DIRECTORY)
    rule = calibration.get("answer_rule")
    if rule is not None and (rule.get("score") == "correctness") != (correctness is not None):
        raise ValueError(f"{root}'s answer rule ranks by {rule.get('score')!r}, but the package {'lacks' if correctness is None else 'ships'} a correctness model.")
    check_decision_records(calibration, correctness)
    return PackagedHead(str(root), calibration, features, heads, standardizer, correctness)
