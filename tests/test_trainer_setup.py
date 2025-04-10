from unittest.mock import patch

import pytest

from finetuning.utils.checkpoints import handle_checkpoint_resume


@patch("finetuning.utils.checkpoints.get_last_checkpoint", return_value=None)
def test_auto_resume_no_checkpoints(get_last_checkpoint_mock):
    resume = handle_checkpoint_resume("auto", "direct")
    assert isinstance(resume, bool)
    assert not resume


@patch("finetuning.utils.checkpoints.get_last_checkpoint", return_value="/path/to/checkpoint")
def test_auto_resume_ckpt_exits(get_last_checkpoint_mock):
    resume = handle_checkpoint_resume("auto", "/path/to/")
    assert resume == "/path/to/checkpoint"
