import time
import os
import sys
import numpy as np
from cellpose import io
import utils

# --- CONFIGURATION ---
# Choose: 'red', 'green', 'blue', or 'gray' (for single-channel images)
CHANNEL_NAME = 'green' 
# Choose model: cpsam (default) or other models you downloaded on your end
MODEL_TYPE = 'cpsam'
# Adjust diameter (mean diameter in px of your cells) if you see wrong results 
DIAMETER = 150

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(script_dir, '..', 'input')
OUTPUT_FOLDER = os.path.join(script_dir, '..', 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. SETUP MODEL ---
model = utils.initialize_cellpose_model(model_type=MODEL_TYPE)

# --- 2. GET IMAGES ---
image_data = utils.get_image_files(INPUT_FOLDER)

if not image_data:
    print(f"No images found in {INPUT_FOLDER}")
    sys.exit(0)

print(f"Found {len(image_data)} images to process. Channel selected: {CHANNEL_NAME}")

# --- 3. MAIN LOOP ---
channel_map = {'red': 0, 'green': 1, 'blue': 2}

for i, data in enumerate(image_data):
    full_path, relative_dir, filename = data
    
    # Construct base output name (handles subfolders)
    if relative_dir == '.':
        base_name = os.path.splitext(filename)[0]
    else:
        base_name = os.path.join(relative_dir, os.path.splitext(filename)[0])
        
    print(f"\nProcessing {i+1}/{len(image_data)}: {filename}...")
    start = time.time()

    try:
        # A. LOAD IMAGE
        img = io.imread(full_path)
        
        # Ensure dimensions are (Y, X, C)
        if img.ndim == 3 and img.shape[0] < 5:
            img = img.transpose(1, 2, 0)

        # B. EXTRACT CHANNEL
        if CHANNEL_NAME in channel_map and img.ndim == 3:
            idx = channel_map[CHANNEL_NAME]
            img_input = img[:, :, idx]
        else:
            img_input = img # Grayscale or fallback

        # C. RUN SEGMENTATION
        masks, flows, styles = model.eval(
            img_input, 
            diameter=DIAMETER,
            cellprob_threshold=-1.0,
            flow_threshold=0.6,
            normalize=True
        )
        
        num_cells = masks.max()
        print(f"  -> Found {num_cells} cells in {time.time()-start:.2f}s")

        # D. SAVE RESULTS
        if num_cells > 0:
            # 1. Save ImageJ ROIs
            roi_path = utils.save_imagej_rois(masks, base_name, OUTPUT_FOLDER)
            print(f"  -> Saved ROIs: {os.path.basename(roi_path)}")

            # 2. Save Visualization
            vis_path = utils.save_visual_overlay(img_input, masks, base_name, OUTPUT_FOLDER, num_cells)
            print(f"  -> Saved Visual: {os.path.basename(vis_path)}")
            
            # 3. Save Raw Masks (for reuse)
            utils.save_masks_npz(masks, base_name, OUTPUT_FOLDER)
            
            # 4. Save CSV stats
            utils.save_results_csv(filename, num_cells, OUTPUT_FOLDER)
        else:
            print("  -> WARNING: No cells found. Skipping file generation.")

    except Exception as e:
        print(f"  -> ERROR: {e}")

print("\nBatch processing complete.")