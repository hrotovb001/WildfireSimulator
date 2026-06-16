import torch
import pytest
import copy

from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step

def test_burn_step():
    inputs = torch.zeros((1, 14, 512, 512))
    inputs_copy = copy.deepcopy(inputs)
    def model(data):
        chan1 = torch.full((1, 1, 512, 512), 1.0)
        chan2 = torch.full((1, 1, 512, 512), 2.0)
        return [torch.cat((chan1, chan2), dim=1)]
    outputs = fire_burn_step(0, model, inputs)
    assert (inputs == inputs_copy).all()
    assert (outputs[0][0] == 1.0).all()
    assert (outputs[0][1] == 2.0).all()


def test_simulator():
    def model(data):
        return data * 2

    def step(t, data, model):
        return model(t + data)

    class FakeTransform:
        def __call__(self, data):
            return self.transform(data)
        def transform(self, data):
            return data / 5
        # the inverse transform is defined incorrectly so that it isn't transparent
        def inverse(self, data):
            return data * 3

    transform = FakeTransform()

    simulator = ForwardBurnSimulator(
        data=10,
        model=model,
        step=step,
        transform=transform,
        dt=2,
        max_t=20
    )

    assert simulator.run_to(10) == pytest.approx(207.6)
    assert simulator.run_to(10, return_history=True) == pytest.approx([10, 12, 24.6, 50.4, 102.6, 207.6])

