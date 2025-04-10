import os

import datasets
import pytest

# Don't let tests use GPUs, which does not work in all environments + we don't want spurious GPU usage.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def datasets_cache_in_tmpdir(tmpdir):
    datasets.config.hf_datasets_cache = tmpdir
