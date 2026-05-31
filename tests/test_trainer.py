import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.nn as nn
from pathlib import Path
import os
import shutil
import re

from wildfire_simulator.callbacks import ModelCheckpoint, TensorBoardCallback
from wildfire_simulator.datasets import WildfireDataset
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor

def test_batch_processor(dataloader):
    dataset = WildfireDataset(dataloader)

    burner = ForwardBurnProcess()

    # random t with min(max(arrival_time), max_t - dt)  
    # input_tensor uses burner at t
    # output_tensor uses burner at t + dt
    # each item of batch uses a different t
    batch_processor = BurnerBatchProcessor(burner=burner, dt=30, max_t=1440, eval=False)
    batch = torch.stack([dataset[0], dataset[1]])
    input_tensor, output_tensor = batch_processor(batch, epoch=0, batch_idx=0)

    # the 14th channel is t broadcasted to all 512 x 512
    # the data is 500, 500 in last two dims but it must be
    # zero padded to a multiple of 32
    assert input_tensor.shape == (2, 14, 512, 512)
    assert output_tensor.shape == (2, 2, 512, 512)

    input_tensor2, output_tensor2 = batch_processor(batch, epoch=0, batch_idx=0)
    assert torch.equal(input_tensor, input_tensor2)
    assert torch.equal(output_tensor, output_tensor2)
    
    input_tensor3, output_tensor3 = batch_processor(batch, epoch=0, batch_idx=1)
    assert not torch.equal(input_tensor, input_tensor3)
    assert not torch.equal(output_tensor, output_tensor3)

    input_tensor4, output_tensor4 = batch_processor(batch, epoch=1, batch_idx=0)
    assert not torch.equal(input_tensor, input_tensor4)
    assert not torch.equal(output_tensor, output_tensor4)

    batch_processor = BurnerBatchProcessor(burner=burner, dt=30, max_t=1440, eval=True)

    input_tensor5, output_tensor5 = batch_processor(batch, epoch=1, batch_idx=0)
    assert torch.equal(input_tensor, input_tensor5)
    assert torch.equal(output_tensor, output_tensor5)

    input_tensor6, output_tensor6 = batch_processor(batch, epoch=1, batch_idx=1)
    assert not torch.equal(input_tensor, input_tensor6)
    assert not torch.equal(output_tensor, output_tensor6)


def test_trainer(dataloader):
    dataset = WildfireDataset(dataloader)

    burner = ForwardBurnProcess()

    # share same batch_processor for train and test to allow model to overfit
    batch_processor = BurnerBatchProcessor(burner=burner, dt=30, max_t=1440, eval=True)

    # share loader for the same reason as above
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    def get_trainer(epochs):
        model = MK_UNet_Regression(
            in_channels=14,
            out_channels=2,
            channels=[16, 32, 64, 96, 160],
            final_activation='relu'
        )

        checkpoint_cb = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filepath='./checkpoints_test/best-model-{epoch:02d}-{val_loss:.2f}.pt'
        )

        train_writer = SummaryWriter("training_test/train")
        val_writer = SummaryWriter("training_test/val")

        tensorboard_cb = TensorBoardCallback(
            train_writer=train_writer,
            val_writer=val_writer
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            5e-4,
            weight_decay=1e-4
        )

        trainer = ForwardBurnTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.L1Loss(),
            train_loader=loader,
            val_loader=loader,
            train_batch_processor = batch_processor,
            val_batch_processor = batch_processor,
            callbacks=[checkpoint_cb, tensorboard_cb],
            epochs=epochs
        )

        return trainer

    shutil.rmtree('./training_test', ignore_errors=True)

    trainer = get_trainer(epochs=10)

    eval_before = trainer.evaluate()
    assert isinstance(eval_before['val_loss'], float)

    shutil.rmtree('./checkpoints_test', ignore_errors=True)

    trainer.fit()

    eval_after = trainer.evaluate()
    assert eval_after['val_loss'] < eval_before['val_loss']

    folder = Path('./checkpoints_test')
    pattern = re.compile(r"best-model-\d{2}-\d+\.\d{2}\.pt")
    matching_files = [
        p for p in folder.iterdir() 
        if p.is_file() and pattern.fullmatch(p.name)
    ]
    assert matching_files

    last_checkpoint = max(matching_files, key=lambda p: p.name)

    trainer = get_trainer(epochs=20)
    trainer.load_checkpoint(last_checkpoint)

    eval_before_resumed = trainer.evaluate()
    assert eval_before_resumed['val_loss'] <= eval_after['val_loss']

    trainer.fit()

    eval_after_resumed = trainer.evaluate()
    assert eval_after_resumed['val_loss'] < eval_after['val_loss']

    train_acc = EventAccumulator("training_test/train")
    train_acc.Reload()
    assert len(set(s.step for s in train_acc.Scalars("Loss"))) == 20

    val_acc = EventAccumulator("training_test/val")
    val_acc.Reload()
    assert len(set(s.step for s in val_acc.Scalars("Loss"))) == 20

