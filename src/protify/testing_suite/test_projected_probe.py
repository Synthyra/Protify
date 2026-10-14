"""The projected probe: one head whose trunk is the same for every representation.

The ablation compares representations through this head, so two properties carry the result.
Each branch must read only its own columns, in every slot of a pair row, or one arm would leak
into another's projection. And the trunk after the projections must be identical across arms,
or a difference between arms could be the head's rather than the representation's.
"""

import sys
import pytest
import torch


try:
    from src.protify.main import parse_arguments
    from src.protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from src.protify.probes.projected_probe import BranchLayout, ProjectedProbe, ProjectedProbeConfig
except ImportError:
    from protify.main import parse_arguments
    from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from protify.probes.projected_probe import BranchLayout, ProjectedProbe, ProjectedProbeConfig


def _probe(**overrides) -> ProjectedProbe:
    settings = dict(input_size=12, projection_size=8, dropout=0.0, num_labels=3, task_type="singlelabel")
    settings.update(overrides)
    return ProjectedProbe(ProjectedProbeConfig(**settings)).eval()


@pytest.mark.parametrize(
    "input_size, widths, slots",
    [(12, None, 1), (12, [5, 7], 1), (24, [5, 7], 2)],
)
def test_logits_have_one_row_per_input_row(input_size, widths, slots):
    torch.manual_seed(0)
    probe = _probe(input_size=input_size, branch_widths=widths)
    assert probe.layout.slots == slots

    logits = probe(torch.randn(4, input_size)).logits  # (4, 3)
    assert logits.shape == (4, 3)


def test_regression_returns_one_output_and_a_loss():
    torch.manual_seed(0)
    probe = _probe(num_labels=1, task_type="regression", branch_widths=[5, 7])
    output = probe(torch.randn(4, 12), labels=torch.randn(4))
    assert output.logits.shape == (4, 1)
    assert torch.isfinite(output.loss)


def test_pair_rows_gather_each_branch_from_both_slots():
    layout = BranchLayout.from_input(24, [5, 7])
    assert layout.columns(0).tolist() == [0, 1, 2, 3, 4, 12, 13, 14, 15, 16]
    assert layout.columns(1).tolist() == [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23]

    rows = torch.arange(48.0).reshape(2, 24)  # (2, 24)
    for branch in range(2):
        assert torch.equal(layout.gather(rows, branch), rows[:, layout.columns(branch)])


@pytest.mark.parametrize("input_size", [12, 24])
def test_perturbing_one_branch_leaves_the_other_projection_unchanged(input_size):
    torch.manual_seed(0)
    probe = _probe(input_size=input_size, branch_widths=[5, 7])
    captured = []
    for branch in probe.branches:
        branch.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))

    rows = torch.randn(3, input_size)  # (3, input_size)
    perturbed = rows.clone()
    columns = probe.layout.columns(1)  # (m * w,)
    # Random rather than a constant shift, which the branch LayerNorm would remove.
    perturbed[:, columns] = torch.randn(3, columns.shape[0])
    probe(rows)
    probe(perturbed)

    first_branch, second_branch, first_branch_perturbed, second_branch_perturbed = captured
    assert torch.equal(first_branch, first_branch_perturbed)
    assert not torch.allclose(second_branch, second_branch_perturbed)


@pytest.mark.parametrize("output", ["linear", "gaussian_process"])
def test_trunk_and_output_parameters_are_identical_across_arms(output):
    """Only the branch projections may differ between a one-branch and a two-branch arm."""
    arms = [
        _probe(input_size=40, branch_widths=None, output=output, projection_size=16),
        _probe(input_size=6, branch_widths=None, output=output, projection_size=16),
        _probe(input_size=46, branch_widths=[40, 6], output=output, projection_size=16),
    ]

    def shared_parameters(probe):
        modules = [*probe.blocks, probe.norm, probe.output]
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    assert len({shared_parameters(probe) for probe in arms}) == 1
    assert all(sum(branch[1].out_features for branch in probe.branches) == 16 for probe in arms)


def test_layout_and_split_errors_are_refused():
    with pytest.raises(ValueError, match="whole number of slots"):
        _probe(input_size=13, branch_widths=[5, 7])
    with pytest.raises(ValueError, match="does not split evenly"):
        _probe(input_size=15, branch_widths=[5, 5, 5], projection_size=8)
    with pytest.raises(ValueError, match="Use probe_type 'projected'"):
        _probe(output="gaussian_process", task_type="multilabel")


def test_branch_sizes_give_each_branch_its_own_projected_width():
    """SAE and mean and variance keep half of 1024 each, the profile gets 1024, and the trunk reads 2048."""
    torch.manual_seed(0)
    probe = _probe(input_size=20, branch_widths=[5, 7, 8], projection_size=16, branch_sizes=[4, 4, 8], trunk="expanded_mlp", projected_hidden_size=32)

    assert [branch[1].out_features for branch in probe.branches] == [4, 4, 8]
    assert [branch[1].in_features for branch in probe.branches] == [5, 7, 8]
    assert probe.blocks[0].expand.in_features == 16, "the trunk reads the joined projections"
    assert probe(torch.randn(4, 20)).logits.shape == (4, 3)


def test_no_branch_sizes_is_the_even_split_it_always_was():
    even = _probe(input_size=12, branch_widths=[5, 7], projection_size=16)
    explicit = _probe(input_size=12, branch_widths=[5, 7], projection_size=16, branch_sizes=[8, 8])

    assert even.config.branch_sizes is None
    assert [branch[1].out_features for branch in even.branches] == [branch[1].out_features for branch in explicit.branches] == [8, 8]
    assert [name for name, _ in even.named_parameters()] == [name for name, _ in explicit.named_parameters()]


def test_a_wide_branch_cannot_leak_into_a_narrow_one_through_its_projection():
    """The profile's 1024 columns reach only its own projection: perturbing it leaves the other two projections bit for bit."""
    torch.manual_seed(0)
    probe = _probe(input_size=20, branch_widths=[5, 7, 8], projection_size=16, branch_sizes=[4, 4, 8])
    captured = []
    for branch in probe.branches:
        branch.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))

    rows = torch.randn(3, 20)  # (3, 20)
    perturbed = rows.clone()
    perturbed[:, 12:] = torch.randn(3, 8)
    probe(rows)
    probe(perturbed)

    first, second, third, first_again, second_again, third_again = captured
    assert torch.equal(first, first_again) and torch.equal(second, second_again)
    assert not torch.allclose(third, third_again)


@pytest.mark.parametrize("sizes, message", [
    ([4, 12], "must hold one positive width for each of the 3 branches"),
    ([4, 4, 0, 8], "must hold one positive width"),
    ([4, 4, 7], "sum to 15, but projection_size is 16"),
])
def test_branch_sizes_that_do_not_fit_the_branches_or_the_trunk_are_refused(sizes, message):
    with pytest.raises(ValueError, match=message):
        _probe(input_size=20, branch_widths=[5, 7, 8], projection_size=16, branch_sizes=sizes)


def test_branch_sizes_are_refused_where_there_are_no_projections():
    with pytest.raises(ValueError, match="linear trunk does not have"):
        _probe(input_size=12, branch_widths=[5, 7], trunk="linear", branch_sizes=[4, 4])


def test_a_saved_probe_keeps_its_branch_sizes_and_an_older_one_loads_evenly(tmp_path):
    torch.manual_seed(0)
    probe = _probe(input_size=20, branch_widths=[5, 7, 8], projection_size=16, branch_sizes=[4, 4, 8])
    probe.save_pretrained(tmp_path / "uneven")
    loaded = ProjectedProbe.from_pretrained(tmp_path / "uneven").eval()
    rows = torch.randn(5, 20)

    assert loaded.config.branch_sizes == [4, 4, 8]
    assert torch.allclose(probe(rows).logits, loaded(rows).logits)

    # A config written before the field existed has no `branch_sizes` key and must still build the even split.
    older = _probe(input_size=12, branch_widths=[5, 7], projection_size=16)
    saved = older.config.to_dict()
    saved.pop("branch_sizes")
    rebuilt = rebuild_probe_from_saved_config("projected", False, saved)
    assert rebuilt.config.branch_sizes is None
    assert [branch[1].out_features for branch in rebuilt.branches] == [8, 8]


def test_the_probe_arguments_hand_branch_sizes_to_the_projected_head_only():
    arguments = ProbeArguments(probe_type="projected", input_size=20, branch_widths=[5, 7, 8], projection_size=16, branch_sizes=[4, 4, 8], num_labels=2)
    probe = get_probe(arguments)

    assert probe.config.branch_sizes == [4, 4, 8]
    assert "branch_sizes" not in vars(ProbeArguments(probe_type="mlp", input_size=16))


@pytest.mark.parametrize("probe_type", ["projected", "sngp"])
def test_saved_config_rebuilds_the_same_probe(probe_type):
    torch.manual_seed(0)
    arguments = ProbeArguments(
        probe_type=probe_type,
        input_size=24,
        branch_widths=[5, 7],
        projection_size=8,
        num_labels=2,
        task_type="singlelabel",
        dropout=0.0,
    )
    probe = get_probe(arguments).eval()
    assert type(probe).__name__ == "ProjectedProbe"
    if probe_type == "sngp":
        probe.fit_posterior([{"embeddings": torch.randn(16, 24), "labels": torch.randint(0, 2, (16,))}])

    rebuilt = rebuild_probe_from_saved_config(probe_type, False, probe.config.to_dict())
    rebuilt.load_state_dict(probe.state_dict())
    rebuilt.eval()
    rows = torch.randn(5, 24)
    assert torch.allclose(probe(rows).logits, rebuilt(rows).logits)
    assert rebuilt.config.output == ("linear" if probe_type == "projected" else "gaussian_process")


def test_a_saved_sngp_probe_loads_with_its_posterior(tmp_path):
    torch.manual_seed(0)
    probe = _probe(input_size=24, branch_widths=[5, 7], output="gaussian_process", num_labels=2)
    probe.fit_posterior([{"embeddings": torch.randn(16, 24), "labels": torch.randint(0, 2, (16,))}])
    probe.save_pretrained(tmp_path)

    loaded = ProjectedProbe.from_pretrained(tmp_path).eval()
    rows = torch.randn(5, 24)
    expected_logits, expected_variance = probe.predict_with_variance(rows)
    logits, variance = loaded.predict_with_variance(rows)
    assert bool(loaded.gaussian_process.posterior_ready)
    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert torch.allclose(variance, expected_variance, atol=1e-6)


def test_other_probe_types_serialize_no_projected_fields():
    """A saved MLP config must carry exactly the fields it carried before this head existed."""
    arguments = ProbeArguments(probe_type="mlp", input_size=16)
    assert not {"projection_size", "spectral_bound", "input_clip", "branch_widths"} & set(vars(arguments))


@pytest.mark.parametrize("probe_type", ["projected", "sngp"])
def test_the_command_line_takes_both_heads_without_matrix_embeddings(probe_type, monkeypatch):
    monkeypatch.setattr(sys.modules[parse_arguments.__module__], "load_secrets_env", lambda: [])
    argv = ["main.py", "--data_names", "DeepLoc-2", "--probe_type", probe_type, "--projection_size", "512", "--spectral_bound", "0.9"]
    monkeypatch.setattr(sys, "argv", argv)

    arguments = parse_arguments()
    probe_arguments = ProbeArguments(**vars(arguments))
    assert (probe_arguments.projection_size, probe_arguments.spectral_bound) == (512, 0.9)
    assert not arguments.matrix_embed
