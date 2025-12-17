# 🔬 Cellpose Segmentation & Counting Tools

This repository contains tools for automated cell segmentation using [Cellpose](https://www.cellpose.org/). It provides two main workflows:
1. **`count.py`**: A simple, automated counter for bacteria or cells.
2. **`segmentation.py`**: An advanced script for 12-bit/16-bit images, saving ImageJ ROIs, normalized visualizations, and batch processing.

---

## 1. Prerequisites

* **Python 3.9+** (Recommended to use the fast package manager `uv`).
* **Git** (For cloning the repository).
* **NVIDIA GPU (Optional but recommended)** or Apple Silicon (M1/M2/M3).

If you don't have UV installed: [Install UV](https://github.com/astral-sh/uv)

---

## 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Noh4ns3n/cellpose-segmentation.git](https://github.com/Noh4ns3n/cellpose-segmentation.git)
    cd cellpose-segmentation
    ```

2.  **Initialize Environment:**
    ```bash
    uv venv
    uv sync
    ```

3.  **Activate Environment:**
    * **macOS / Linux:** `source .venv/bin/activate`
    * **Windows:** `.\.venv\Scripts\activate`

---

## 3. Workflows

### Option A: Advanced Segmentation (`segmentation.py`)
*Best for: Confocal microscopy, 12-bit/16-bit TIFFs, fibrous structures (actin), and generating ImageJ ROIs.*

1.  **Configure:** Open `src/segmentation.py` and edit the top section:
    ```python
    CHANNEL_NAME = 'green'  # 'green', 'red', 'blue', or 'gray'
    MODEL_TYPE = 'cpsam'    # 'cpsam' (default) or any other model you downloaded on your end
    DIAMETER = 150          # Adjust with the mean diameter (in px) of your cells if you see wrong results
    ```
2.  **Run:**
    ```bash
    python src/segmentation.py
    ```
3.  **Output (`output/` folder):**
    * `_overlay.png`: Normalized visualization (visible even for dark images).
    * `_rois.zip`: Drag and drop this into ImageJ/Fiji to see editable masks.
    * `_masks.npz`: Raw data for Python analysis.

### Option B: Simple Counting (`count.py`)
*Best for: Quick counting of small objects (e.g., bacteria) in standard images.*

1.  **Run:**
    ```bash
    # Default diameter is 10px
    python src/count.py 
    
    # Or specify diameter manually (e.g. 30px)
    python src/count.py 30
    ```

---

## 4. Troubleshooting

### ❌ "CUDA GPU not detected" (NVIDIA Users)
If the script says "Using CPU" despite having an NVIDIA GPU, you likely have the CPU-only version of PyTorch installed by default.

**The Fix:**
Reinstall PyTorch with specific CUDA support inside your environment:

```bash
# 1. Activate your environment first!
source .venv/bin/activate

# 2. Uninstall the wrong version
uv pip uninstall torch torchvision torchaudio

# 3. Install the CUDA-enabled version (Adjust cu118 to your driver version if needed)
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

### ❌ Images look black / "White Blob"
Cause: Your image might be 12-bit or 16-bit (low pixel values) or saturated.

Solution: Use segmentation.py instead of count.py. It includes automatic normalization (clips top/bottom 1%) to make these images visible in the _overlay.png output.
