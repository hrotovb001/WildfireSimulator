import numpy as np
from torch.utils.data import Dataset

from wildfire_simulator.datasets import WildfireDataset

dataset = WildfireDataset()

def test_landscape_layers():
    elevation = dataset.elevation 
    assert isinstance(elevation, np.ndarray)
    assert len(elevation.shape) == 2
    assert elevation.max() != elevation.min()
    assert not np.isnan(elevation).any()
    assert not (elevation == -9999).any()

    slope = dataset.slope
    assert isinstance(slope, np.ndarray)
    assert len(slope.shape) == 2
    assert slope.max() != slope.min()
    assert not np.isnan(slope).any()
    assert not (slope == -9999).any()

    aspect = dataset.aspect
    assert isinstance(aspect, np.ndarray)
    assert len(aspect.shape) == 2
    assert aspect.max() != aspect.min()
    assert not np.isnan(aspect).any()
    assert not (aspect == -9999).any()

    fuel = dataset.fuel
    assert isinstance(fuel, np.ndarray)
    assert len(fuel.shape) == 2
    assert fuel.max() != fuel.min()
    assert not np.isnan(fuel).any()
    assert not (fuel == -9999).any()

    canopy_cover = dataset.canopy_cover
    assert isinstance(canopy_cover, np.ndarray)
    assert len(canopy_cover.shape) == 2
    assert canopy_cover.max() != canopy_cover.min()
    assert not np.isnan(canopy_cover).any()
    assert not (canopy_cover == -9999).any()

    stand_height = dataset.stand_height
    assert isinstance(stand_height, np.ndarray)
    assert len(stand_height.shape) == 2
    assert stand_height.max() != stand_height.min()
    assert not np.isnan(stand_height).any()
    assert not (stand_height == -9999).any()

    canopy_base_height = dataset.canopy_base_height
    assert isinstance(canopy_base_height, np.ndarray)
    assert len(canopy_base_height.shape) == 2
    assert canopy_base_height.max() != canopy_base_height.min()
    assert not np.isnan(canopy_base_height).any()
    assert not (canopy_base_height == -9999).any()

    canopy_bulk_density = dataset.canopy_bulk_density
    assert isinstance(canopy_bulk_density, np.ndarray)
    assert len(canopy_bulk_density.shape) == 2
    assert canopy_bulk_density.max() != canopy_bulk_density.min()
    assert not np.isnan(canopy_bulk_density).any()
    assert not (canopy_bulk_density == -9999).any()

def test_ignitions():
    ignitions = dataset.ignitions
    assert len(ignitions) > 0

    # the ignition is the pixel coordinate relative to landscape georeference
    # a given pixel coordinate respresents where the point in the .shp file
    # aligns with the numpy array for landscape
    ignition = ignitions[0]
    assert isinstance(ignition, tuple)
    assert len(ignition) == 2
    assert all(isinstance(x, int) for x in ignition)

    y, x = ignition
    elevation = dataset.elevation
    assert y >= 0 and y < elevation.shape[0]
    assert x >= 0 and x < elevation.shape[1]

def test_trails():
    trials = dataset.trials
    assert len(dataset.trials) > 0

    # each trial has fire, ignition number, windspeed,
    # winddir, foliar_moisture
    trial = dataset.trials[0]
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

def test_trial_array():
    assert len(dataset) > 0
    
    arr = dataset[0]

    # 8 landscape, 2 fire, windspeed, winddir and foliar_moisture
    # frame is centered at the ingition coordinate
    assert arr.shape == (13, 500, 500)
    assert not np.isnan(arr).any()
    assert not (arr == -9999).any()

def test_pytorch_dataset():
    assert isinstance(dataset, Dataset)

def test_min_max_norm():
    min_val = dataset.min_val
    assert min_val.shape == (13,)
    assert not np.isnan(min_val).any()
    assert not (min_val == -9999).any()

    max_val = dataset.max_val
    assert max_val.shape == (13,)
    assert not np.isnan(max_val).any()
    assert not (max_val == -9999).any()

    assert (min_val < max_val).all()

