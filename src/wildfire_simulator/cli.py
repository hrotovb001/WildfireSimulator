def format_train_description(epoch, loss):
    return f"Epoch {epoch} | Loss: {loss:.4f}"

def format_val_description(val_loss):
    return f"Validating | Loss: {val_loss:.4f}"

def format_results(summary):
    return (
        f"=== Training Complete ===\n"
        f"Best Epoch: {summary['best_epoch']}\n"
        f"Train Loss: {summary['train_loss']:.4f}\n"
        f"Val Loss: {summary['val_loss']:.4f}%\n"
        f"Time Elapsed: {summary['duration_seconds']:.2f}s"
    )
