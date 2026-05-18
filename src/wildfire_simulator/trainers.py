import torch
import torch.nn.functional as F
import random

from wildfire_simulator.forward_burn_process import ForwardBurnProcess


def _pad_to_multiple(tensor, multiple=32):
    """Pad the last two spatial dimensions to the next multiple of `multiple`."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    # pad last dim (width) then second-last (height)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))
    return padded, h, w


class ForwardBurnTrainer:
    """
    Trainer that uses the ForwardBurnProcess to create temporal training pairs.

    dt   – time difference (minutes) between input and target frames
    max_t – maximum allowed burn time (minutes)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        callbacks=None,
        epochs=1,
        dt=30,
        max_t=1440,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.epochs = epochs
        self.dt = dt
        self.max_t = max_t
        self.burner = ForwardBurnProcess()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)

        for batch in self.train_loader:
            batch = batch.to(self.device)
            N = batch.size(0)
            # Use the static method to create training pairs (includes random t per sample)
            inputs_14, targets = ForwardBurnTrainer.prepare_batch(
                batch=batch, dt=self.dt, max_t=self.max_t
            )
            # Model expects 13 input channels (no t channel)
            inputs = inputs_14[:, :13, :, :]

            # Pad to a multiple of 32 so the model's internal attention gates
            # always receive tensors with compatible spatial sizes.
            inputs_padded, orig_h, orig_w = _pad_to_multiple(inputs, multiple=32)
            targets_padded, _, _ = _pad_to_multiple(targets, multiple=32)

            self.optimizer.zero_grad()
            preds_padded = self.model(inputs_padded)
            # Handle possible tuple/list output from model
            if isinstance(preds_padded, (tuple, list)):
                preds_padded = preds_padded[0]
            # Crop predictions back to the original spatial size
            preds = preds_padded[:, :, :orig_h, :orig_w]
            loss = self.loss_fn(preds, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * N

        return total_loss / n_samples

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        n_samples = len(self.val_loader.dataset)

        # Initialise a single generator to keep evaluation deterministic across calls
        val_gen = torch.Generator(device=torch.device('cpu'))
        val_gen.manual_seed(0)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                N = batch.size(0)

                inputs_14, targets = ForwardBurnTrainer.prepare_batch(
                    batch=batch, dt=self.dt, max_t=self.max_t, generator=val_gen
                )
                # Model expects 13 input channels
                inputs = inputs_14[:, :13, :, :]

                # Pad to a multiple of 32 so the model's internal attention gates
                # always receive tensors with compatible spatial sizes.
                inputs_padded, orig_h, orig_w = _pad_to_multiple(inputs, multiple=32)
                targets_padded, _, _ = _pad_to_multiple(targets, multiple=32)

                preds_padded = self.model(inputs_padded)
                # Handle possible tuple/list output from model
                if isinstance(preds_padded, (tuple, list)):
                    preds_padded = preds_padded[0]
                # Crop predictions back to the original spatial size
                preds = preds_padded[:, :, :orig_h, :orig_w]
                loss = self.loss_fn(preds, targets)
                total_loss += loss.item() * N

        return total_loss / n_samples

    def fit(self):
        for epoch in range(self.epochs):
            _train_loss = self._train_epoch()
            val_loss = self._validate()
            metrics = {'val_loss': val_loss}
            for cb in self.callbacks:
                cb.on_validation_end(epoch=epoch, metrics=metrics, model=self.model)

    def evaluate(self):
        """Return the current validation loss as a dict."""
        val_loss = self._validate()
        return {'val_loss': val_loss}

    @staticmethod
    def prepare_batch(batch, dt, max_t, generator=None):
        """
        Prepare training inputs and targets from a batch of frames (N,13,H,W).

        For each sample a random burn time `t` is sampled up to
        min(arrival_max, max_t - dt).  When `generator` is provided a torch
        Generator is used for reproducibility.  Returns `(inputs, targets)`:

        - inputs: (N,14,H,W) where the first 13 channels are the frame burned
          at time `t` and the 14th channel is a constant equal to `t`.
        - targets: (N,2,H,W) containing the fire mask (channel 8) and arrival
          (channel 9) of the frame burned at time `t + dt`.
        """

        burner = ForwardBurnProcess()
        N = batch.size(0)
        device = batch.device
        dtype = batch.dtype
        input_frames = []
        target_frames = []

        for i in range(N):
            frame = batch[i]                     # (13, H, W)
            arrival = frame[9]                   # arrival times
            max_arr = arrival.max().item()
            upper = min(max_arr, max_t - dt)

            if upper <= 0:
                t = 0.0
            else:
                # Use the provided torch Generator for reproducible randomness,
                # otherwise fall back to the built-in random module.
                if generator is not None:
                    r = torch.rand(1, generator=generator, device=torch.device('cpu')).item()
                    t = upper * r
                else:
                    t = random.uniform(0, upper)

            in_frame = burner(frame, t)
            out_frame = burner(frame, t + dt)

            # Build the 14-channel input: add t as a constant channel
            t_channel = torch.full((1, in_frame.shape[-2], in_frame.shape[-1]),
                                   t, dtype=dtype, device=device)
            in_with_t = torch.cat([in_frame, t_channel], dim=0)   # (14, H, W)
            target = torch.stack([out_frame[8], out_frame[9]], dim=0)   # (2, H, W)

            input_frames.append(in_with_t.unsqueeze(0))   # (1,14,H,W)
            target_frames.append(target.unsqueeze(0))     # (1,2,H,W)

        inputs = torch.cat(input_frames, dim=0)   # (N, 14, H, W)
        targets = torch.cat(target_frames, dim=0) # (N, 2, H, W)

        return inputs, targets
