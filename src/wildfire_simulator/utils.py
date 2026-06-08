import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_frame(data, filepath):
    """
    Save a multi-channel frame as a PNG image using matplotlib.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (C, H, W) where C is the number of channels.
    filepath : str or Path
        Path where the image will be saved.
    """
    n_channels = data.shape[0]

    # Determine a roughly square grid that can hold all channels
    ncols = int(np.ceil(np.sqrt(n_channels)))
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    # Flatten axes for easy indexing into a 1‑D array
    axes = np.atleast_1d(axes).flatten()

    # Plot each channel
    for i in range(n_channels):
        ax = axes[i]
        ax.imshow(data[i], cmap='gray', aspect='auto')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(n_channels, nrows * ncols):
        axes[j].axis('off')

    # Ensure the output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

