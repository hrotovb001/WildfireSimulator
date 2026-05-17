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

            input_frames = []
            target_frames = []

            for i in range(N):
                frame = batch[i]               # (13, H, W)
                arrival_times = frame[9]        # arrival time channel

                max_arr = arrival_times.max().item()
                upper = min(max_arr, self.max_t - self.dt)
                if upper <= 0:
                    t = 0.0
                else:
                    t = random.uniform(0, upper)

                in_frame = self.burner(frame, t)          # (13, H, W)
                out_frame = self.burner(frame, t + self.dt)

                # Model only needs to predict fire mask (ch8) and arrival (ch9)
                target = torch.stack([out_frame[8], out_frame[9]], dim=0)   # (2, H, W)

                input_frames.append(in_frame.unsqueeze(0))          # (1, 13, H, W)
                target_frames.append(target.unsqueeze(0))           # (1, 2, H, W)

            inputs = torch.cat(input_frames, dim=0)                 # (N, 13, H, W)
            targets = torch.cat(target_frames, dim=0)               # (N, 2, H, W)

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

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                N = batch.size(0)

                input_frames = []
                target_frames = []

                for i in range(N):
                    frame = batch[i]
                    t = 0.0  # deterministic evaluation
                    in_frame = self.burner(frame, t)
                    out_frame = self.burner(frame, t + self.dt)

                    target = torch.stack([out_frame[8], out_frame[9]], dim=0)

                    input_frames.append(in_frame.unsqueeze(0))
                    target_frames.append(target.unsqueeze(0))

                inputs = torch.cat(input_frames, dim=0)
                targets = torch.cat(target_frames, dim=0)

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
