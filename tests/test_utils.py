import numpy as np
from pathlib import Path
import shutil
from PIL import Image

from wildfire_simulator.utils import save_frame, save_comparison

def test_save_frame():
    rng = np.random.default_rng(seed=42)
    data = rng.random((13, 500, 500))
    
    shutil.rmtree('./tmp', ignore_errors=True)

    save_frame(data, './tmp/random_frame.png')

    file_path = Path('./tmp/random_frame.png')
    assert file_path.is_file()

    img1 = Image.open('./tmp/random_frame.png')
    img2 = Image.open('./tests/baseline/random_frame.png')

    assert img1.mode == img2.mode

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)

    assert arr1.shape == arr2.shape
    assert np.array_equal(arr1, arr2)


def test_save_comparison():
    rng = np.random.default_rng(seed=42)

    # 10 different fires where each image is an arrival map
    trues = rng.random((10, 500, 500))
    preds = rng.random((10, 500, 500))

    shutil.rmtree('./tmp', ignore_errors=True)

    save_comparison(preds, trues, './tmp/random_comparison.png')

    file_path = Path('./tmp/random_comparison.png')
    assert file_path.is_file()

    img1 = Image.open('./tmp/random_comparison.png')
    img2 = Image.open('./tests/baseline/random_comparison.png')

    assert img1.mode == img2.mode

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)

    assert arr1.shape == arr2.shape
    assert np.array_equal(arr1, arr2)

