import torch
import numpy as np

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.forward_burn_process import ForwardBurnProcess

def test_model(dataset):
    model = MK_UNet_Regression(
        in_channels=13,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='relu'
    )

    burner = ForwardBurnProcess()
    input = burner(dataset[0], 30)
    input = np.pad(input, ((0, 0), (6, 6), (6, 6)), mode='constant', constant_values=0)

    input_tensor = torch.from_numpy(input).unsqueeze(0)

    out_tensor = model(input_tensor)
    assert out_tensor[0].shape == (1, 2, 512, 512)

