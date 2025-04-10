import pytest
import torch

from finetuning.model import _init_new_rows_by_avg_sample


def test_init_new_rows_by_avg_sample_constant_first_rows():
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((5, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=2)
    assert torch.equal(weights, new_weights[0:3, :])


def test_init_new_rows_by_avg_sample_changed_last_rows():
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((5, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=2)
    assert not torch.equal(new_weights[3:4, :], torch.zeros((2, 3)))


def test_init_new_rows_by_avg_sample_close_to_mean():
    # NOTE: this test could technically fail as the new rows are sampled. Statistically this should happen like once in
    # a million years
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((4, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=1)
    assert torch.allclose(new_weights[3, :], torch.ones((1, 3)), atol=10.0)
