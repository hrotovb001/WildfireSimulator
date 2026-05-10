import os
import numpy as np
import rioxarray
import geopandas as gpd
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
                    # parse metadata from filename (example: "trail_I0_WS12_WD312_M74.tif")
                    base = os.path.splitext(os.path.basename(fname))[0]
                    parts = base.split('_')
                    # find the ignition number (starts with 'I')
                    ign_str = next(p for p in parts if p.startswith('I'))
                    ignition = int(ign_str[1:])
                    ws_str = next(p for p in parts if p.startswith('WS'))
                    windspeed = int(ws_str[2:])
                    wd_str = next(p for p in parts if p.startswith('WD'))
                    winddir = int(wd_str[2:])
                    m_str = next(p for p in parts if p.startswith('M'))
                    foliar_moisture = int(m_str[1:])
                    trial = {
                        "fire": stacked,
                        "ignition": ignition,
                        "windspeed": windspeed,
                        "winddir": winddir,
                        "foliar_moisture": foliar_moisture,
                    }
                    self.trials.append(trial)

        # Load ignition points from IGNITIONS directory
        ignitions_dir = os.getenv("IGNITIONS")
        self.ignitions = []
        if ignitions_dir and os.path.isdir(ignitions_dir):
            from rasterio.transform import rowcol
            # Get transform and CRS from the already opened landscape dataset
            transform = self._landscape.rio.transform()
            landscape_crs = self._landscape.rio.crs
            for fname in sorted(os.listdir(ignitions_dir)):
                if not (fname.lower().endswith('.shp') and fname.startswith('ignition_')):
                    continue
                fpath = os.path.join(ignitions_dir, fname)
                gdf = gpd.read_file(fpath)
                if len(gdf) == 0:
                    continue
                # Use the first geometry (expects a single Point per file)
                geom = gdf.geometry.iloc[0]
                # Reproject to landscape CRS when necessary
                if gdf.crs is not None and landscape_crs is not None and gdf.crs != landscape_crs:
                    gdf = gdf.to_crs(landscape_crs)
                    geom = gdf.geometry.iloc[0]
                # Convert to pixel coordinates (row, col) using the affine transform
                rows, cols = rowcol(transform, [geom.x], [geom.y])
                pixel = (int(rows[0]), int(cols[0]))
                self.ignitions.append(pixel)
