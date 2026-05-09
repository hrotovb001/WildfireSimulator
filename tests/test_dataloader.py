import numpy as np

from wildfire_simulator.dataloader import DataLoader

def test_landscape_layers():
   loader = DataLoader()

   elevation = loader.elevation 
   assert isinstance(elevation, np.ndarray)
   assert len(elevation.shape) == 2
   assert elevation.max() != elevation.min()
   assert not np.isnan(elevation).any()

   slope = loader.slope
   assert isinstance(slope, np.ndarray)
   assert len(slope.shape) == 2
   assert slope.max() != slope.min()
   assert not np.isnan(slope).any()

   aspect = loader.aspect
   assert isinstance(aspect, np.ndarray)
   assert len(aspect.shape) == 2
   assert aspect.max() != aspect.min()
   assert not np.isnan(aspect).any()

   fuel = loader.fuel
   assert isinstance(fuel, np.ndarray)
   assert len(fuel.shape) == 2
   assert fuel.max() != fuel.min()
   assert not np.isnan(fuel).any()

   canopy_cover = loader.canopy_cover
   assert isinstance(canopy_cover, np.ndarray)
   assert len(canopy_cover.shape) == 2
   assert canopy_cover.max() != canopy_cover.min()
   assert not np.isnan(canopy_cover).any()

   stand_height = loader.stand_height
   assert isinstance(stand_height, np.ndarray)
   assert len(stand_height.shape) == 2
   assert stand_height.max() != stand_height.min()
   assert not np.isnan(stand_height).any()

   canopy_base_height = loader.canopy_base_height
   assert isinstance(canopy_base_height, np.ndarray)
   assert len(canopy_base_height.shape) == 2
   assert canopy_base_height.max() != canopy_base_height.min()
   assert not np.isnan(canopy_base_height).any()

   canopy_bulk_density = loader.canopy_bulk_density
   assert isinstance(canopy_bulk_density, np.ndarray)
   assert len(canopy_bulk_density.shape) == 2
   assert canopy_bulk_density.max() != canopy_bulk_density.min()
   assert not np.isnan(canopy_bulk_density).any()

def test_trails_layer():
    loader = DataLoader()

    trials = loader.trials
    assert len(loader.trials) > 0

    # each trial has a mask and a fire arrival time channel
    trial = loader.trails[0]
    assert isinstance(trial, np.ndarray)
    assert len(trial.shape) == 3
    assert trial.shape[0] == 2

    # the mask indicates where the fire has been (0 or 1)
    mask = trial[0]
    assert ((mask == 0) | (mask == 1)).all()

    # the arrival time indicates the time at which the fire reached a pixel (default to value of 0 for masked pixels)
    arrival = trial[1]
    assert not np.isnan(arrival).any()
