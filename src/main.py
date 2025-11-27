import numpy as np
from PIL import Image
import os
import sys
import utils
import matplotlib.pyplot as plt

# Get diameter argument, default to 10 if not provided
try:
    if len(sys.argv) > 1:
        DIAMETER = int(sys.argv[1])
        if DIAMETER <= 0:
             raise ValueError("Diameter must be a positive integer.")
    else:
        DIAMETER = 10
except ValueError as e:
    print(f"Error reading diameter argument: {e}")
    print("Usage: python src/main.py [DIAMETER_VALUE]")
    sys.exit(1)

print(f"--- Cellpose Segmentation (Diameter: {DIAMETER}px) ---")

# --- I/O Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(script_dir, '..', 'input')
OUTPUT_FOLDER = os.path.join(script_dir, '..', 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)

# --- File Collection (Recursive) ---
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
image_data = [] # Stores tuples: (full_path, relative_dir, filename)

for root, _, files in os.walk(INPUT_FOLDER):
    # relative_dir will be '' if root is INPUT_FOLDER, or 'folder_name/...' otherwise
    relative_dir = os.path.relpath(root, INPUT_FOLDER)
    
    for filename in files:
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(root, filename)
            image_data.append((full_path, relative_dir, filename))

if not image_data:
    print(f"ERROR: No images found in {INPUT_FOLDER}. Aborting.")
    sys.exit(0)

print(f"Found {len(image_data)} images to process...")

# --- Main Processing Loop ---
model = utils.initialize_cellpose_model(model_type='cpsam')

for i, data in enumerate(image_data):
    full_path, relative_dir, filename = data
    
    if relative_dir == '.':
        # File is directly in INPUT_FOLDER
        output_filename = filename
        print(f"\nProcessing {i+1}/{len(image_data)}: {filename}")
    else:
        # File is in a subfolder, use subfolder/filename
        output_filename = os.path.join(relative_dir, filename)
        print(f"\nProcessing {i+1}/{len(image_data)}: {output_filename}")


    try:
        img = np.array(Image.open(full_path))
        masks, flows, diams = model.eval(img, diameter=DIAMETER) 
        
        nombre_bacteries = masks.max() 
        print(f"Result: {nombre_bacteries} bacteria detected.")

        # --- Save Results ---
        utils.save_results_csv(output_filename, nombre_bacteries, OUTPUT_FOLDER)
        unique_base_name = os.path.splitext(output_filename)[0]
        utils.save_masks_npz(masks, unique_base_name, OUTPUT_FOLDER)
        fig, output_path = utils.save_visualization(img, masks, unique_base_name, OUTPUT_FOLDER, nombre_bacteries)
        
    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {output_filename}: {e}")
        
print("\nBatch processing complete!")