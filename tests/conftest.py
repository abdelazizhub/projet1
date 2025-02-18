import warnings
import pytest

@pytest.fixture(autouse=True)
def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)