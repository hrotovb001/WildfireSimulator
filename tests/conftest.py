import pytest
from wildfire_simulator.datasets import WildfireDataLoader

@pytest.fixture(scope="session")
def dataloader():
    return WildfireDataLoader()
