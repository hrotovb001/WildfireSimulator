import pytest
from wildfire_simulator.dataloader import WildfireDataLoader, TrialFileLoader, TrialCollection

@pytest.fixture(scope="session")
def dataloader():
    trials = TrialCollection(TrialFileLoader())
    return WildfireDataLoader(trials)
