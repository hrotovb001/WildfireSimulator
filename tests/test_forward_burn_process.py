import numpy as np

from wildfire_simulator.datasets import WildfireDataset
from wildfire_simulator.forward_burn_process import ForwardBurnProcess

def test_forward_burn_process(dataloader):
    dataset = WildfireDataset(dataloader)
    burner = ForwardBurnProcess()

    tensor = dataset[0]

    # burner masks fire channels 0 (mask) and 1 (arrival) / set to 0
    # all pixels with arrival > t
    new_tensor = burner(tensor, 30)
    assert new_tensor[1].max() <= 30

    mask = new_tensor[0] != 0
    assert mask.sum() > 0
    assert (tensor[:, mask] == new_tensor[:, mask]).all()
