from __future__ import annotations

import pytest

from tests.helpers import build_test_config


@pytest.fixture()
def small_config(tmp_path):
    return build_test_config(tmp_path)
