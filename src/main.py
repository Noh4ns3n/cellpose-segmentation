import numpy as np
from PIL import Image
import os
import sys
import utils
import matplotlib.pyplot as plt

# --- I/O Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(script_dir, '..', 'assets')
OUTPUT_FOLDER = os.path.join(script_dir, '..', 'output_segmentation')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 5. Main Processing Loop ---
plt.show(block=False)
model = utils.initialize_cellpose_model(model_type='cpsam')
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

if not image_files:
    print(f"\nERROR: No images found in {INPUT_FOLDER}. Aborting.")
    sys.exit(0)

print(f"\nFound {len(image_files)} images to process...")

for i, filename in enumerate(image_files):
    full_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\nProcessing {i+1}/{len(image_files)}: {filename}")

    try:
        img = np.array(Image.open(full_path))
        masks, flows, diams = model.eval(img, diameter=10)
        nombre_bacteries = masks.max()
        print(f"Result: {nombre_bacteries} bacteria detected.")

        utils.save_masks_npz(masks, filename, OUTPUT_FOLDER)
        utils.save_visualization(img, masks, filename, OUTPUT_FOLDER, nombre_bacteries)
        # fig, output_path = utils.save_visualization(img, masks, filename, OUTPUT_FOLDER, nombre_bacteries)
        # utils.display_img(fig, delay=3000)

    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {filename}: {e}")

print("\nBatch processing complete!")