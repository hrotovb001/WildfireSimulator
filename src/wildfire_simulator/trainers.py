import torch
import torch.nn.functional as F

from tqdm import tqdm

from wildfire_simulator.scheduled_sampler import ScheduledSampler

import numpy as np

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


class BurnerBatchProcessor:
    def __init__(
        self,
        burner,
        dt,
        eval,
        sampler,
        rng
    ):
        self.burner = burner
        self.dt = dt
        self.rng = rng
        self.eval = eval
        self.sampler = sampler

    def __call__(self, pred, true, epoch, batch_idx, t):
        if self.eval:
            epoch = 0

        self.rng.seed(epoch * 10_000 + batch_idx)


        N = true.size(0)
        input_frames = []
        target_frames = []

        for i in range(N):
            use_pred = self.rng.rand().item() < self.sampler.get_prob(epoch) or self.eval
            in_frame = pred[i] if use_pred else self.burner(true[i], t)
            out_frame = self.burner(true[i], t + self.dt)

            # Build the 14-channel input: add t as a constant channel
            t_channel = torch.full((1, in_frame.shape[-2], in_frame.shape[-1]), t, device=in_frame.device)
            in_with_t = torch.cat([in_frame, t_channel], dim=0)   # (14, H, W)
            target = torch.stack([out_frame[0], out_frame[1]], dim=0)   # (2, H, W)

            input_frames.append(in_with_t.unsqueeze(0))   # (1,14,H,W)
            target_frames.append(target.unsqueeze(0))     # (1,2,H,W)

        inputs = torch.cat(input_frames, dim=0)   # (N, 14, H, W)
        targets = torch.cat(target_frames, dim=0) # (N, 2, H, W)

        # Pad spatial dimensions to the next multiple of 32 (the raw data is
        # 500×500 → 512×512).
        inputs, _, _ = _pad_to_multiple(inputs, multiple=32)
        targets, _, _ = _pad_to_multiple(targets, multiple=32)

        return inputs, targets


class ForwardBurnTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        train_batch_processor,
        val_batch_processor,
        callbacks=None,
        epochs=1,
        max_t=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_batch_processor = train_batch_processor
        self.val_batch_processor = val_batch_processor
        self.callbacks = callbacks or []
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.current_epoch = 0
        self.max_t = max_t

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)
        preds_padded = torch.zeros(2, 13, 512, 512)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            for t in np.arange(0, self.max_t, self.train_batch_processor.dt):
                inputs, targets = self.train_batch_processor(preds_padded, batch, epoch=epoch, batch_idx=batch_idx, t=t)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                N = inputs.size(0)

                self.optimizer.zero_grad()
                pred_out = self.model(inputs)
                if isinstance(pred_out, (list, tuple)):
                    pred_out = pred_out[0]
                loss = self.loss_fn(pred_out, targets)
                loss.backward()
                self.optimizer.step()

                preds_padded = inputs[:, :13, :, :].detach().cpu().clone()
                preds_padded[:, :2, :, :] = pred_out.detach().cpu()

                total_loss += loss.item() * N * self.train_batch_processor.dt / self.max_t
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_samples

    def _validate(self, epoch, total_epochs):
        self.model.eval()
        total_loss = 0.0
        n_samples = len(self.val_loader.dataset)

        pbar = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                preds_padded = None

                for t in np.arange(0, self.max_t, self.val_batch_processor.dt):
                    if preds_padded is None:
                        pred_input = batch
                    else:
                        pred_input = preds_padded

                    inputs, targets = self.val_batch_processor(
                        pred_input,
                        batch,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        t=t,
                    )

                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    N = inputs.size(0)

                    pred_out = self.model(inputs)
                    if isinstance(pred_out, (list, tuple)):
                        pred_out = pred_out[0]

                    preds_padded = inputs[:, :13, :, :].detach().cpu().clone()
                    preds_padded[:, :2, :, :] = pred_out.detach().cpu()

                loss = self.loss_fn(pred_out, targets)
                total_loss += loss.item() * N
                pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        return total_loss / n_samples

    def fit(self):
        total_epochs = self.epochs
        for epoch in range(self.current_epoch, total_epochs):
            train_loss = self._train_epoch(epoch, total_epochs)
            val_loss = self._validate(epoch, total_epochs)
            metrics = {'train_loss': train_loss, 'val_loss': val_loss}
            for cb in self.callbacks:
                cb.on_validation_end(epoch=epoch, metrics=metrics, model=self.model, optimizer=self.optimizer)
            self.current_epoch += 1

    def evaluate(self):
        val_loss = self._validate(epoch=0, total_epochs=1)
        return {'val_loss': val_loss}

