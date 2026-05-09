import os
import numpy as np
import rioxarray
from dotenv import load_dotenv

class DataLoader:
    """Loads landscape GeoTIFF (path from LANDSCAPE env variable) and makes
    elevation, slope, and aspect available as 2‑D numpy arrays."""

    def __init__(self):
        # Load environment from .env file
        load_dotenv()

        tif_path = os.getenv("LANDSCAPE")
        if not tif_path:
            raise EnvironmentError(
                "Environment variable 'LANDSCAPE' is not set. "
                "Make sure your .env file contains LANDSCAPE=<path>"
            )

        # Open the GeoTIFF with rioxarray – returns an xarray.DataArray
        da = rioxarray.open_rasterio(tif_path)

        # Store the full DataArray for potential later use
        self._landscape = da

        # The first band (index 0) is elevation
        self.elevation = da.isel(band=0).values

        # The second band (index 1) is slope
        self.slope = da.isel(band=1).values

        # The third band (index 2) is aspect
        self.aspect = da.isel(band=2).values

        # Band 3: fuel model (FBFM40)
        self.fuel = da.isel(band=3).values

        # Band 4: canopy cover (CC)
        self.canopy_cover = da.isel(band=4).values

        # Band 5: stand height (CH)
        self.stand_height = da.isel(band=5).values

        # Band 6: canopy base height (CBH)
        self.canopy_base_height = da.isel(band=6).values

        # Band 7: canopy bulk density (CBD)
        self.canopy_bulk_density = da.isel(band=7).values

        # Load fire trial arrival times from TRIALS directory
        trials_dir = os.getenv("TRIALS")
        self.trials = []
        if trials_dir and os.path.isdir(trials_dir):
            for fname in sorted(os.listdir(trials_dir)):
                if fname.lower().endswith('.tif') or fname.lower().endswith('.tiff'):
                    fpath = os.path.join(trials_dir, fname)
                    trial_ds = rioxarray.open_rasterio(fpath)
                    # each trial GeoTIFF has one band: fire arrival time,
                    # with NaN where fire never arrives
                    arr = trial_ds.isel(band=0).values
                    # mask: 1 where fire arrived (non‑NaN), 0 elsewhere
                    mask = (~np.isnan(arr)).astype(np.uint8)
                    # replace NaN with 0 so the array can be used numerically
                    arrival = np.where(np.isnan(arr), 0, arr)
                    stacked = np.stack([mask, arrival], axis=0)
                    self.trials.append(stacked)
