import numpy as np
import torch
from torch.utils.data import Dataset

from .dataloader import DataLoader


class WildfireDataset(Dataset):
    """Thin PyTorch Dataset wrapper around DataLoader."""

    def __init__(self):
        # Create the underlying DataLoader instance and expose its attributes
        self._loader = DataLoader()
        self.elevation = self._loader.elevation
        self.slope = self._loader.slope
        self.aspect = self._loader.aspect
        self.fuel = self._loader.fuel
        self.canopy_cover = self._loader.canopy_cover
        self.stand_height = self._loader.stand_height
        self.canopy_base_height = self._loader.canopy_base_height
        self.canopy_bulk_density = self._loader.canopy_bulk_density
        self.ignitions = self._loader.ignitions
        self.trials = self._loader.trials

        # Compute channel-wise min and max across all trials/frames for normalization
        n_channels = 13
        min_val = np.full(n_channels,  np.inf)
        max_val = np.full(n_channels, -np.inf)
        for i in range(self.__len__()):
            arr = self._loader[i]
            frame_min = np.min(arr, axis=(1, 2))
            frame_max = np.max(arr, axis=(1, 2))
            min_val = np.minimum(min_val, frame_min)
            max_val = np.maximum(max_val, frame_max)
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        # Delegate to the DataLoader which already returns the expected
        # (13, 500, 500) numpy array.
        return self._loader[idx]
