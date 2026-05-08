import sys
from pathlib import Path

# Make sure the project root is on Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dataloader import DataLoader

def test_landscape_layers():
   loader = DataLoader()

   elevation = loader.elevation 
   assert isinstance(elevation, np.ndarray)
   assert len(elevation.shape) == 2
   assert elevation.max() != elevation.min()
   assert not np.isnan(elevation).any()

