import torch_tilt_series


def test_imports_with_version():
    assert isinstance(torch_tilt_series.__version__, str)
