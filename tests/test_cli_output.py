from wildfire_simulator.cli import format_train_description, format_val_description, format_results

def test_format_train():
    epoch = 3
    raw_loss = 0.456789
    desc = format_train_description(epoch, raw_loss)
    assert desc == "Epoch 3 | Loss: 0.4568"

def test_format_val():
    raw_loss = 0.456789
    desc = format_val_description(raw_loss)
    assert desc == "Validating | Loss: 0.4568"

def test_format_results():
    summary_metrics = {
        "best_epoch": 12,
        "train_loss": 0.02341,
        "val_loss": 0.03578,
        "duration_seconds": 145.2
    }

    result = format_results(summary_metrics)

    expected_output = (
        "=== Training Complete ===\n"
        "Best Epoch: 12\n"
        "Train Loss: 0.0234\n"
        "Val Loss: 0.0358%\n"
        "Time Elapsed: 145.20s"
    )
    assert result == expected_output

