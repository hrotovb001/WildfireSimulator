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

   aspect = aspect.slope
   assert isinstance(aspect, np.ndarray)
   assert len(aspect.shape) == 2
   assert ascpect.max() != aspect.min()
   assert not np.isnan(slope).any()
