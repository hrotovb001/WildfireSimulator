import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
import shutil
import re

from wildfire_simulator.callbacks import ModelCheckpoint
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
    batch_processor = BurnerBatchProcessor(burner=burner, dt=30, max_t=1440)
    batch = torch.stack([dataset[0], dataset[1]])
    input_tensor, output_tensor = batch_processor(batch, epoch=0, batch_idx=0, eval=False)

    # the 14th channel is t broadcasted to all 512 x 512
    # the data is 500, 500 in last two dims but it must be
    # zero padded to a multiple of 32
    assert input_tensor.shape == (2, 14, 512, 512)
    assert output_tensor.shape == (2, 2, 512, 512)

    input_tensor2, output_tensor2 = batch_processor(batch, epoch=0, batch_idx=0, eval=False)
    assert torch.equal(input_tensor, input_tensor2)
    assert torch.equal(output_tensor, output_tensor2)
    
    input_tensor3, output_tensor3 = batch_processor(batch, epoch=0, batch_idx=1, eval=False)
    assert not torch.equal(input_tensor, input_tensor3)
    assert not torch.equal(output_tensor, output_tensor3)

    input_tensor4, output_tensor4 = batch_processor(batch, epoch=1, batch_idx=0, eval=False)
    assert not torch.equal(input_tensor, input_tensor4)
    assert not torch.equal(output_tensor, output_tensor4)

    input_tensor5, output_tensor5 = batch_processor(batch, epoch=1, batch_idx=0, eval=True)
    assert torch.equal(input_tensor, input_tensor5)
    assert torch.equal(output_tensor, output_tensor5)

    input_tensor6, output_tensor6 = batch_processor(batch, epoch=1, batch_idx=1, eval=True)
    assert not torch.equal(input_tensor, input_tensor6)
    assert not torch.equal(output_tensor, output_tensor6)


def test_trainer(dataloader):
    dataset = WildfireDataset(dataloader)

    burner = ForwardBurnProcess()

    batch_processor = BurnerBatchProcessor(burner=burner, dt=30, max_t=1440)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    model = MK_UNet_Regression(
        in_channels=14,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='relu'
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename='best-model-{epoch:02d}-{val_loss:.2f}'
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
        train_loader=train_loader,
        val_loader=val_loader,
        batch_processor=batch_processor,
        callbacks=[checkpoint],
        epochs=3
    )

    eval_before = trainer.evaluate()
    assert isinstance(eval_before['val_loss'], float)

    shutil.rmtree('./checkpoints', ignore_errors=True)

    trainer.fit()

    eval_after = trainer.evaluate()
    assert isinstance(eval_after['val_loss'], float)
    assert eval_after['val_loss'] < eval_before['val_loss']

    folder = Path('./checkpoints')
    pattern = re.compile(r"best-model-\d{2}-\d+\.\d{2}\.pt")
    found = any(pattern.fullmatch(p.name) for p in folder.iterdir() if p.is_file())
    assert found
