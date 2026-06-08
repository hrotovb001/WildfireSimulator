import numpy as np
from pathlib import Path
import shutil

from wildfire_simulator.utils import save_frame, save_comparison

def test_save_frame():
    rng = np.random.default_rng()
    data = rng.random((13, 500, 500))
    
    shutil.rmtree('./tmp', ignore_errors=True)

    save_frame(data, './tmp/random_frame.png')

    file_path = Path('./tmp/random_frame.png')
    assert file_path.is_file()

    size_bytes = file_path.stat().st_size
    assert size_bytes > 1000


def test_save_comparison():
    rng = np.random.default_rng()

    # 10 different fires where each image is an arrival map
    trues = rng.random((10, 500, 500))
    preds = rng.random((10, 500, 500))

    shutil.rmtree('./tmp', ignore_errors=True)

    save_comparison(preds, trues, './tmp/random_comparison.png')

    file_path = Path('./tmp/random_comparison.png')
    assert file_path.is_file()

    size_bytes = file_path.stat().st_size
    assert size_bytes > 1000
    
