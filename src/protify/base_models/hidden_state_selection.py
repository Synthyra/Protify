import torch


def select_hidden_state(
    last_hidden_state: torch.Tensor,
    hidden_states: tuple[torch.Tensor, ...] | None,
    hidden_state_index: int,
) -> torch.Tensor:
    # last_hidden_state: (b, l, d), hidden_states: (b, l, d) for each of n_layers + 1 states
    assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
    if hidden_state_index == -1:
        return last_hidden_state  # (b, l, d)
    assert hidden_states is not None, "hidden_state_index selection requires output_hidden_states=True."
    assert len(hidden_states) > 0, "Model returned no hidden states."
    return hidden_states[hidden_state_index]  # (b, l, d)
