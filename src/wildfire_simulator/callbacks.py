import torch
import os

class ModelCheckpoint:
    """Simple model checkpoint callback. Saves the model when the monitored quantity improves."""

    def __init__(self, monitor='val_loss', mode='min', filename='best-model-{epoch:02d}-{val_loss:.2f}'):
        self.monitor = monitor
        self.mode = mode
        self.filename_template = filename
        self.best_metric = None
        self.best_path = None

        if self.mode == 'min':
            self.compare = lambda current, best: current < best
        elif self.mode == 'max':
            self.compare = lambda current, best: current > best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def on_validation_end(self, epoch, metrics, model):
        current = metrics.get(self.monitor)
        if current is None:
            return
        if self.best_metric is None or self.compare(current, self.best_metric):
            self.best_metric = current
            fname = self.filename_template.format(epoch=epoch, val_loss=current)
            os.makedirs('./checkpoints', exist_ok=True)
            path = os.path.join('./checkpoints', fname)
            torch.save(model.state_dict(), path)
            self.best_path = path
