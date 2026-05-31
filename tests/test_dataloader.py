import numpy as np

from wildfire_simulator.dataloader import TrialCollection

def test_landscape_layers(dataloader):
    elevation = dataloader.elevation 
    assert isinstance(elevation, np.ndarray)
    assert len(elevation.shape) == 2
    assert elevation.max() != elevation.min()
    assert not np.isnan(elevation).any()
    assert not (elevation == -9999).any()

    slope = dataloader.slope
    assert isinstance(slope, np.ndarray)
    assert len(slope.shape) == 2
    assert slope.max() != slope.min()
    assert not np.isnan(slope).any()
    assert not (slope == -9999).any()

    aspect = dataloader.aspect
    assert isinstance(aspect, np.ndarray)
    assert len(aspect.shape) == 2
    assert aspect.max() != aspect.min()
    assert not np.isnan(aspect).any()
    assert not (aspect == -9999).any()

    fuel = dataloader.fuel
    assert isinstance(fuel, np.ndarray)
    assert len(fuel.shape) == 2
    assert fuel.max() != fuel.min()
    assert not np.isnan(fuel).any()
    assert not (fuel == -9999).any()

    canopy_cover = dataloader.canopy_cover
    assert isinstance(canopy_cover, np.ndarray)
    assert len(canopy_cover.shape) == 2
    assert canopy_cover.max() != canopy_cover.min()
    assert not np.isnan(canopy_cover).any()
    assert not (canopy_cover == -9999).any()

    stand_height = dataloader.stand_height
    assert isinstance(stand_height, np.ndarray)
    assert len(stand_height.shape) == 2
    assert stand_height.max() != stand_height.min()
    assert not np.isnan(stand_height).any()
    assert not (stand_height == -9999).any()

    canopy_base_height = dataloader.canopy_base_height
    assert isinstance(canopy_base_height, np.ndarray)
    assert len(canopy_base_height.shape) == 2
    assert canopy_base_height.max() != canopy_base_height.min()
    assert not np.isnan(canopy_base_height).any()
    assert not (canopy_base_height == -9999).any()

    canopy_bulk_density = dataloader.canopy_bulk_density
    assert isinstance(canopy_bulk_density, np.ndarray)
    assert len(canopy_bulk_density.shape) == 2
    assert canopy_bulk_density.max() != canopy_bulk_density.min()
    assert not np.isnan(canopy_bulk_density).any()
    assert not (canopy_bulk_density == -9999).any()


def test_ignitions(dataloader):
    # ignitions are a dict where key is the ignition number (int)
    # and the value is the ignition
    # the ignition number comes from the file name "ignition_%d.shp"
    ignitions = dataloader.ignitions
    assert len(ignitions) > 0
    assert isinstance(ignitions, dict)

    # verify that every key is an int and every value is a valid pixel
    for k, v in ignitions.items():
        assert isinstance(k, int)
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert all(isinstance(x, int) for x in v)

    # pick the first ignition and confirm it lies inside the landscape
    first_key = next(iter(ignitions))
    ignition = ignitions[first_key]
    y, x = ignition
    elevation = dataloader.elevation
    assert y >= 0 and y < elevation.shape[0]
    assert x >= 0 and x < elevation.shape[1]


def test_trials(dataloader):
    trials = dataloader.trials
    assert len(dataloader.trials) > 0

    # each trial has fire, ignition number, windspeed,
    # winddir, foliar_moisture
    trial = dataloader.trials[0]
    assert isinstance(trial, dict)

    # each fire has a mask and a fire arrival time channel
    fire = trial["fire"]
    assert len(fire.shape) == 3
    assert fire.shape[0] == 2

    # the mask indicates where the fire has been (0 or 1)
    mask = fire[0]
    assert ((mask == 0) | (mask == 1)).all()

    # the arrival time indicates the time at which the fire reached a pixel (default to value of 0 for masked pixels)
    arrival = fire[1]
    assert not np.isnan(fire).any()
    assert not (fire == -9999).any()

    # all other properties are int
    assert isinstance(trial["ignition"], int)
    assert isinstance(trial["windspeed"], int)
    assert isinstance(trial["winddir"], int)
    assert isinstance(trial["foliar_moisture"], int)

    # file path where trial comes from
    assert isinstance(trial["file_path"], str)


def test_trial_collection_laziness():
    class FakeTrialFileLoader:
        def load(self, file_path):
            return self.data

    loader = FakeTrialFileLoader()
    trials = TrialCollection(loader)

    loader.data = "example data"
    assert trials[0] == "example data"

    loader.data = "some other example data"
    assert trials[0] == "some other example data"

