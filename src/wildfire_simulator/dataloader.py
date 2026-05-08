import os
import numpy as np
import rioxarray
from dotenv import load_dotenv

class DataLoader:
    """Loads landscape GeoTIFF (path from LANDSCAPE env variable) and makes
    elevation available as a 2‑D numpy array."""

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
