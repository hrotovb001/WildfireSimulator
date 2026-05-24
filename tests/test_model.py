import torch
import torch.nn.functional as F

from wildfire_simulator.datasets import WildfireDataset
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.models import MK_UNet_Regression

def test_model(dataloader):
    dataset = WildfireDataset(dataloader)

    model = MK_UNet_Regression(
        in_channels=14,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='relu'
    )

    burner = ForwardBurnProcess()
    input_tensor = burner(dataset[0], 30)
    input_tensor = F.pad(input_tensor, (6, 6, 6, 6, 0, 0), mode='constant', value=0)

    # add time channel to make model time aware
    time = torch.tensor(30.0).view(1, 1, 1).expand(1, 512, 512)
    input_tensor = torch.cat([input_tensor, time])

    input_tensor = input_tensor.unsqueeze(0)

    output_tensor = model(input_tensor)
    assert output_tensor[0].shape == (1, 2, 512, 512)

