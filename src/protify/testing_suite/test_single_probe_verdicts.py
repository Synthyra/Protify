"""The single-probe decision rules on constructed scheme T series.

Every series is built so the rule's answer is known: which candidates improve and why the rest
fail, the pnorm exception to the NLL rule, the cheapest tier winning over a better AURC in a
dearer one, the fallback's head following the head verdict, the ensemble's gain kept, and the
quantile clip replacing the clamp only when it is better and worse on nothing.
"""

import json
import numpy as np
import pytest


try:
    from src.protify.uncertainty.single_probe import (
        OUTPUT_JSON,
        OUTPUT_MARKDOWN,
        SingleProbeRules,
        single_probe_verdicts,
        write_single_probe,
    )
    from src.protify.uncertainty.verdicts import RepeatSeries
except ImportError:
    from protify.uncertainty.single_probe import (
        OUTPUT_JSON,
        OUTPUT_MARKDOWN,
        SingleProbeRules,
        single_probe_verdicts,
        write_single_probe,
    )
    from protify.uncertainty.verdicts import RepeatSeries


REPEATS = 60
ARM = "both"
DATASETS = {"bin": ("binary", "mcc"), "multi": ("multiclass", "mcc"), "reg": ("regression", "spearman")}
REFERENCE = {"primary": 0.5, "nll": 0.4, "aurc": 0.2, "shift_auroc": 0.6}


class Builder:
    """Scheme T series with a little noise, every (head, method) set relative to the reference."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.values: dict[tuple[str, str, str, str, str], np.ndarray] = {}

    def add(self, head: str, method: str, *, primary: float = 0.0, nll: float | None = 0.0, aurc: float = 0.0, datasets=DATASETS) -> None:
        """A method whose primary is the reference's plus `primary`, and NLL and AURC the reference's times one plus the given."""
        for data_name in datasets:
            kind, primary_metric = DATASETS[data_name]
            noise = lambda: 1e-4 * self.rng.normal(size=REPEATS)  # noqa: E731
            key = (data_name, ARM, head, method)
            self.values[(*key, primary_metric)] = REFERENCE["primary"] + primary + noise()
            if nll is not None:
                self.values[(*key, "gaussian_nll" if kind == "regression" else "nll")] = REFERENCE["nll"] * (1.0 + nll) + noise()
            self.values[(*key, "aurc")] = REFERENCE["aurc"] * (1.0 + aurc) + noise()
            self.values[(*key, "shift_auroc")] = REFERENCE["shift_auroc"] + noise()

    def series(self) -> RepeatSeries:
        return RepeatSeries(
            self.values, {name: kind for name, (kind, _) in DATASETS.items()}, {name: metric for name, (_, metric) in DATASETS.items()}
        )


def _base(builder: Builder, linear_primary: float = 0.0, qclip: bool = True) -> None:
    """The reference, the plain heads, the ceiling, and, unless told not to, the clamp's heads."""
    builder.add("projected", "plain", nll=0.05, aurc=0.05)
    builder.add("projected", "temperature")
    builder.add("projected", "ensemble_temperature", primary=0.02, nll=-0.2, aurc=-0.2)
    builder.add("linear", "plain", primary=linear_primary, nll=0.05, aurc=0.05)
    if qclip:
        builder.add("projected_qclip", "plain")
        builder.add("projected_qclip", "temperature")


def _candidate(verdicts: dict, family: str, head: str, method: str) -> dict:
    record = verdicts["per_arm"][ARM][family]
    return next(candidate for candidate in record["candidates"] if (candidate["head"], candidate["method"]) == (head, method))


def test_an_improving_candidate_in_the_cheapest_tier_is_adopted_over_a_better_one_in_a_dearer_tier():
    builder = Builder()
    _base(builder)
    builder.add("linear", "temperature", primary=0.0, nll=-0.05, aurc=-0.05)
    builder.add("linear", "loss_head", primary=0.0, nll=-0.05, aurc=-0.3)
    builder.add("projected", "laplace_temperature", primary=-0.05, nll=-0.1, aurc=-0.1)  # loses accuracy
    builder.add("faithful", "faithful_temperature", nll=0.02, aurc=-0.1, datasets=("reg",))  # worse NLL
    verdicts = single_probe_verdicts(builder.series(), [ARM])

    classification = verdicts["per_arm"][ARM]["classification"]
    assert classification["datasets"] == ["bin", "multi"]
    assert _candidate(verdicts, "classification", "linear", "temperature")["improves"]
    assert _candidate(verdicts, "classification", "linear", "loss_head")["improves"]
    laplace = _candidate(verdicts, "classification", "projected", "laplace_temperature")
    assert not laplace["improves"] and laplace["failed"] == ["primary lower bound above -0.01"]
    assert classification["adopted"] == {"head": "linear", "method": "temperature", "tier": 1, "fallback": False}

    regression = verdicts["per_arm"][ARM]["regression"]
    faithful = _candidate(verdicts, "regression", "faithful", "faithful_temperature")
    assert faithful["datasets"] == ["reg"] and faithful["failed"] == ["relative NLL interval below 0"]
    assert _candidate(verdicts, "classification", "faithful", "faithful_temperature")["datasets"] == []
    assert regression["adopted"]["method"] == "temperature" and regression["adopted"]["tier"] == 1


def test_pnorm_improves_on_rules_one_and_three_alone_since_its_nll_is_temperatures():
    builder = Builder()
    _base(builder)
    # Score-only: no primary or NLL of its own, which come from temperature on its head.
    builder.add("projected", "temperature_pnorm", nll=None, aurc=-0.1, datasets=("bin", "multi"))
    for data_name in ("bin", "multi"):
        del builder.values[(data_name, ARM, "projected", "temperature_pnorm", "mcc")]
    verdicts = single_probe_verdicts(builder.series(), [ARM])

    pnorm = _candidate(verdicts, "classification", "projected", "temperature_pnorm")
    assert pnorm["nll_rule_replaced"] and pnorm["improves"]
    assert "relative NLL interval below 0" not in pnorm["conditions"]
    assert pnorm["pooled"]["relative_nll"]["mean"] == pytest.approx(0.0, abs=1e-3)
    assert verdicts["per_arm"][ARM]["classification"]["adopted"]["method"] == "temperature_pnorm"
    # Its share of the ensemble's AURC gain: -0.1 of the ensemble's -0.2.
    assert pnorm["ensemble_share"]["relative_aurc"]["share"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(("linear_primary", "fallback_head", "verdict"), [
    (0.05, "linear", "linear better"),
    (0.0, "linear", "equivalent"),
    (-0.05, "projected", "projected better"),
])
def test_with_no_improving_candidate_the_fallback_follows_the_head_verdict(linear_primary, fallback_head, verdict):
    builder = Builder()
    _base(builder, linear_primary)
    builder.add("linear", "temperature", primary=linear_primary, nll=0.01, aurc=0.01)
    verdicts = single_probe_verdicts(builder.series(), [ARM])
    for family in ("classification", "regression"):
        record = verdicts["per_arm"][ARM][family]
        assert record["head"]["verdict"] == verdict
        assert record["adopted"] == {"head": fallback_head, "method": "temperature", "tier": None, "fallback": True, "with": "split conformal on V_cal"}


@pytest.mark.parametrize(("qclip_primary", "qclip_nll", "replaces"), [(0.05, -0.01, True), (0.05, 0.05, False), (0.0, -0.05, False)])
def test_the_quantile_clip_replaces_the_clamp_only_when_better_and_worse_on_nothing(qclip_primary, qclip_nll, replaces):
    builder = Builder()
    _base(builder, qclip=False)
    builder.add("projected_qclip", "plain", primary=qclip_primary)
    builder.add("projected_qclip", "temperature", primary=qclip_primary, nll=qclip_nll)
    verdicts = single_probe_verdicts(builder.series(), [ARM])
    assert verdicts["per_arm"][ARM]["classification"]["clamp"]["replaces_clamp"] is replaces


def test_the_verdicts_are_written_as_json_and_markdown(tmp_path):
    builder = Builder()
    _base(builder)
    builder.add("linear", "temperature", nll=-0.05, aurc=-0.05)
    verdicts = single_probe_verdicts(builder.series(), [ARM], SingleProbeRules())
    write_single_probe(tmp_path, verdicts)
    written = json.loads((tmp_path / OUTPUT_JSON).read_text(encoding="utf-8"))
    assert written["per_arm"][ARM]["classification"]["adopted"]["method"] == "temperature"
    assert {row["head"] for row in written["shift"]} >= {"projected", "linear"}
    text = (tmp_path / OUTPUT_MARKDOWN).read_text(encoding="utf-8")
    assert "Adopted: temperature on linear (tier 1)." in text and chr(0x2014) not in text, "the Markdown uses no em dash"
