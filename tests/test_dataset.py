import numpy as np
import torch
from torch.utils.data import Dataset

from wildfire_simulator.datasets import WildfireDataset

def test_dataset(dataloader):
    dataset = WildfireDataset(dataloader)

    assert len(dataset) > 0
    
    tensor = dataset[0]
    assert isinstance(tensor, torch.Tensor)

    # 2 fire, 8 landscape, windspeed, winddir and foliar_moisture
    # frame is centered at the ingition coordinate
    assert tensor.shape == (13, 500, 500)
    assert not torch.isnan(tensor).any()
    assert not (tensor == -9999).any()

def test_pytorch_dataset_type():
    assert issubclass(WildfireDataset, Dataset)

def test_min_max_norm(dataloader):
    dataset = WildfireDataset(dataloader)

    min_val = dataset.min_val
    assert min_val.shape == (13,)
    assert not np.isnan(min_val).any()
    assert not (min_val == -9999).any()

    max_val = dataset.max_val
    assert max_val.shape == (13,)
    assert not np.isnan(max_val).any()
    assert not (max_val == -9999).any()

    assert (min_val < max_val).all()

