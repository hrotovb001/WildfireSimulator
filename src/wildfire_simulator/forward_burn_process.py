import numpy as np
import torch

class ForwardBurnProcess:
    """
    Simulates forward burning up to a given time `t`.

    When called with a state frame (13×H×W) and a burn time `t`, it zeroes
    the fire-mask (channel 0) and arrival-time (channel 1) of every pixel
    whose original arrival time exceeds `t`.

    Accepts both NumPy arrays and torch.Tensor, returning the same type.
    """
    def __call__(self, frame, t: float):
        if isinstance(frame, torch.Tensor):
            np_frame = frame.cpu().numpy()
            not_burnt = np_frame[1] > t
            np_out = np_frame.copy()
            np_out[0][not_burnt] = 0.0
            np_out[1][not_burnt] = 0.0
            return torch.from_numpy(np_out).to(frame.device, dtype=frame.dtype)
        # numpy case
        out = frame.copy()
        not_burnt = out[1] > t
        out[0][not_burnt] = 0.0
        out[1][not_burnt] = 0.0
        return out
