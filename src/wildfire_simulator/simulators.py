import torch
import copy
import numpy as np

def fire_burn_step(t, model, inputs):
    inputs = copy.deepcopy(inputs)
    inputs[0][13] = t
    with torch.no_grad():
        pred = model(inputs)[0]
        inputs[0][0] = pred[0][0].detach()
        inputs[0][1] = pred[0][1].detach()
    return inputs


class ForwardBurnSimulator:
    def __init__(
        self,
        data,
        model,
        step,
        transform,
        dt,
        max_t
    ):
        self.data = data
        self.model = model
        self.step = step
        self.transform = transform
        self.dt = dt
        self.max_t = max_t

    def run_to(self, t, return_history=False):
        input = self.transform(self.data)
        history = [self.data]
        dt = self.dt/self.max_t
        for i in np.arange(0, t/self.max_t, dt):
            input = self.step(i, input, self.model)
            history.append(self.transform.inverse(input))
        return history if return_history else history[-1]
