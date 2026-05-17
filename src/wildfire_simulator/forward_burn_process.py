import numpy as np
import torch

class ForwardBurnProcess:
    """
    Simulates forward burning up to a given time `t`.

    When called with a state frame (13×H×W) and a burn time `t`, it zeroes
    the fire-mask (channel 8) and arrival-time (channel 9) of every pixel
    whose original arrival time exceeds `t`.

    Accepts both NumPy arrays and torch.Tensor, returning the same type.
    """
    def __call__(self, frame, t: float):
        if isinstance(frame, torch.Tensor):
            np_frame = frame.cpu().numpy()
            not_burnt = np_frame[9] > t
            np_out = np_frame.copy()
            np_out[8][not_burnt] = 0.0
            np_out[9][not_burnt] = 0.0
            return torch.from_numpy(np_out).to(frame.device, dtype=frame.dtype)
        # numpy case
        out = frame.copy()
        not_burnt = out[9] > t
        out[8][not_burnt] = 0.0
        out[9][not_burnt] = 0.0
        return out
