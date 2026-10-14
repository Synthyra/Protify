"""Tests for seed_utils.py reproducibility utilities."""

import random
import numpy as np
import pytest
import torch

from protify.seed_utils import set_global_seed, get_global_seed, seed_worker, dataloader_generator


def test_set_global_seed_returns_seed() -> None:
    returned_seed = set_global_seed(42)
    assert returned_seed == 42


def test_get_global_seed_after_set() -> None:
    set_global_seed(123)
    assert get_global_seed() == 123


def test_set_global_seed_none_generates_seed() -> None:
    returned_seed = set_global_seed(None)
    assert isinstance(returned_seed, int)
    assert returned_seed >= 0


def test_reproducibility_torch() -> None:
    set_global_seed(99)
    first_draw = torch.randn(5)  # (n=5,)
    set_global_seed(99)
    second_draw = torch.randn(5)  # (n=5,)
    assert torch.equal(first_draw, second_draw)


def test_reproducibility_numpy() -> None:
    set_global_seed(99)
    first_draw = np.random.rand(5)  # (n=5,)
    set_global_seed(99)
    second_draw = np.random.rand(5)  # (n=5,)
    assert np.array_equal(first_draw, second_draw)


def test_reproducibility_random() -> None:
    set_global_seed(99)
    first_draw = [random.random() for _ in range(5)]
    set_global_seed(99)
    second_draw = [random.random() for _ in range(5)]
    assert first_draw == second_draw


def test_seed_worker_deterministic() -> None:
    torch.manual_seed(42)
    seed_worker(0)
    first_worker_draw = np.random.rand(3)  # (n=3,)
    first_random_value = random.random()

    torch.manual_seed(42)
    seed_worker(0)
    second_worker_draw = np.random.rand(3)  # (n=3,)
    second_random_value = random.random()

    assert np.array_equal(first_worker_draw, second_worker_draw)
    assert first_random_value == second_random_value


def test_dataloader_generator_returns_generator() -> None:
    generator = dataloader_generator(42)
    assert isinstance(generator, torch.Generator)


def _replayed_arguments(tmp_path, lines: list[str]):
    from protify.logger import LogReplayer

    log = tmp_path / "run.txt"
    log.write_text("\n".join(["header", *lines, ""]), encoding="utf-8")
    return LogReplayer(log).parse_log()


def test_a_replay_draws_from_the_logged_seed_not_the_command_line(tmp_path, monkeypatch) -> None:
    """The reseed once passed set_global_seed an argument it does not take, and a suppressed
    TypeError left the replay drawing from the replaying command's seed."""
    from protify import main

    determinism = []
    monkeypatch.setattr(main, "set_determinism", lambda: determinism.append(True))
    replay_args = _replayed_arguments(tmp_path, ["seed:\t7", "deterministic:\tTrue"])

    set_global_seed(42)  # what main() seeds from the replaying command
    main.seed_replayed_run(replay_args)
    replayed_draw = torch.randn(3)  # (n=3,)
    set_global_seed(7)

    assert get_global_seed() == 7
    assert torch.equal(replayed_draw, torch.randn(3))
    assert determinism == [True]


def test_a_replayed_log_without_a_seed_records_the_one_it_drew(tmp_path, monkeypatch) -> None:
    from protify import main

    monkeypatch.setattr(main, "set_determinism", lambda: pytest.fail("the log asked for no deterministic kernels"))
    replay_args = _replayed_arguments(tmp_path, ["probe_type:\tmlp"])

    main.seed_replayed_run(replay_args)

    assert isinstance(replay_args.seed, int)
    assert get_global_seed() == replay_args.seed
