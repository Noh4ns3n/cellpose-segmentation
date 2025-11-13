import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from cellpose import models

def initialize_cellpose_model(model_type='cpsam'):
    """Initializes the Cellpose model with device detection (MPS/CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_gpu = True
        print("Using Apple MPS GPU acceleration.")
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("Using CPU (MPS not available).")

    model = models.CellposeModel(
        pretrained_model=model_type,
        gpu=use_gpu, 
        device=device
    )
    print(f'Initialized Cellpose model: {model}')
    return model

def save_masks_npz(masks, filename, output_dir):
    """Saves the segmentation mask array in compressed numpy format."""
    base_name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f'{base_name}_masks.npz')
    np.savez_compressed(output_path, masks=masks)
    # Return the path so the main script can use it if needed
    return output_path

def save_visualization(img, masks, filename, output_dir, count):
    """Creates and saves a composite image of the original image + masks."""
    # plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.imshow(masks, cmap='jet', alpha=0.5)
    ax.set_title(f'{count} cells detected in {filename}')
    ax.axis('off')
    
    base_name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f'{base_name}_viz.png')
    fig.savefig(output_path, bbox_inches='tight')
    # plt.close(fig)
    # plt.ion()
    return fig, output_path

def display_img(fig, delay = 1.0):
    """
    Displays the matplotlib figure in real-time, pauses, and closes it.
    Requires plt.show(block=False) to be called once beforehand in the main script.
    """
    plt.draw() 
    plt.pause(delay) 
    plt.close(fig)