import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from cellpose import models, io, plot
import csv

def initialize_cellpose_model(model_type='cpsam'):
    """Initializes the Cellpose model with device detection (CUDA/MPS/CPU)."""
    
    # 1. Check for NVIDIA CUDA GPU (Standard for Windows/Linux)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu = True
        print("Using NVIDIA CUDA GPU acceleration.")
    
    # 2. Check for Apple Silicon MPS (Standard for macOS M-series)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_gpu = True
        print("Using Apple MPS GPU acceleration.")
        
    # 3. Fallback to CPU
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("Using CPU (No GPU acceleration available).")

    # Initialize the Cellpose model
    model = models.CellposeModel(
        pretrained_model=model_type,
        gpu=use_gpu, 
        device=device
    )
    print(f'Initialized Cellpose model: {model}')
    return model

# Utils for segmentation.py
def get_image_files(input_folder, valid_extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    """Recursively finds all images in the input folder."""
    image_data = [] 
    for root, _, files in os.walk(input_folder):
        relative_dir = os.path.relpath(root, input_folder)
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                full_path = os.path.join(root, filename)
                image_data.append((full_path, relative_dir, filename))
    return image_data

def save_imagej_rois(masks, base_filename, output_dir):
    """
    Saves ROIs in a .zip format compatible with ImageJ/Fiji.
    """
    output_path_base = os.path.join(output_dir, base_filename)
    output_subdir = os.path.dirname(output_path_base)
    os.makedirs(output_subdir, exist_ok=True)
    
    # cellpose.io.save_rois automatically appends '_rois.zip'
    io.save_rois(masks, output_path_base)
    return f"{output_path_base}_rois.zip"

def save_visual_overlay(img, masks, base_filename, output_dir, count):
    """
    Saves a normalized visualization (useful for 12-bit/dark images).
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Normalize image for display (clip top/bottom 1% to handle dark 12-bit images)
    # If image is multi-channel, we assume the relevant channel was passed in 'img'
    if img.ndim == 2:
        p1, p99 = np.percentile(img, (1, 99))
        img_visual = np.clip((img - p1) / (p99 - p1), 0, 1)
        # Stack to RGB for overlay
        img_visual_rgb = np.stack([img_visual]*3, axis=-1)
    else:
        # Fallback for RGB
        img_visual_rgb = img

    overlay = plot.mask_overlay(img_visual_rgb, masks)
    
    ax.imshow(overlay)
    ax.set_title(f"{count} cells detected in {os.path.basename(base_filename)}")
    ax.axis('off')
    
    output_path_base = os.path.join(output_dir, base_filename)
    output_subdir = os.path.dirname(output_path_base)
    os.makedirs(output_subdir, exist_ok=True)
    
    final_output_path = f'{output_path_base}_overlay.png'
    plt.savefig(final_output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return final_output_path


# Utils for count.py

def save_results_csv(filename, bacteria_count, output_dir, csv_file_name='segmentation_summary.csv'):
    """
    Appends the image filename and detected count to a CSV file.
    """
    csv_path = os.path.join(output_dir, csv_file_name)
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'number_detected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'image_name': filename, 
            'number_detected': bacteria_count
        })
        
    return csv_path

def save_masks_npz(masks, base_filename, output_dir):
    """
    Saves the segmentation mask array in compressed numpy format.
    base_filename can contain subdirectory paths (e.g., 'batch1/imageA').
    """
    # Construct the full output path
    output_path_base = os.path.join(output_dir, base_filename)
    
    # Ensure the subdirectory path exists (e.g., creates 'output/batch1')
    output_subdir = os.path.dirname(output_path_base)
    os.makedirs(output_subdir, exist_ok=True)
    
    final_output_path = f'{output_path_base}_masks.npz'
    np.savez_compressed(final_output_path, masks=masks)
    return final_output_path

def save_count_visualization(img, masks, base_filename, output_dir, count):
    """
    Creates and saves a composite image of the original image + masks.
    base_filename can contain subdirectory paths (e.g., 'batch1/imageA').
    """
    # plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.imshow(masks, cmap='jet', alpha=0.5)
    ax.set_title(f'{count} cells detected in {os.path.basename(base_filename)}')
    ax.axis('off')
    
    # Construct the full output path
    output_path_base = os.path.join(output_dir, base_filename)

    # Ensure the subdirectory path exists
    output_subdir = os.path.dirname(output_path_base)
    os.makedirs(output_subdir, exist_ok=True)
    
    final_output_path = f'{output_path_base}_viz.png'
    fig.savefig(final_output_path, bbox_inches='tight')
    
    # plt.close(fig) is now in display_img()
    return fig, final_output_path

def display_img(fig, delay = 1.0):
    """
    Displays the matplotlib figure in real-time, pauses, and closes it.
    Requires plt.show(block=False) to be called once beforehand in the main script.
    """
    plt.draw() 
    plt.pause(delay) 
    plt.close(fig)