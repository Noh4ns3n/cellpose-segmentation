# 🔬 Cellpose Segmentation Script

This script uses the Cellpose library to automatically segment cells (specifically the `cpsam` model for generalist segmentation) across an entire folder of images.

---

## 1. Prerequisites

You need the following installed:

* **Python 3.9+** (Recommended to use the version manager `uv`).
* **Git** (For cloning the repository).

If you don't have UV installed, follow the installation instructions here: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

---

## 2. Setup and Initialization

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Noh4ns3n/cellpose-segmentation.git](https://github.com/Noh4ns3n/cellpose-segmentation.git)
    ```

2.  **Navigate into the project directory:**
    ```bash
    cd cellpose-segmentation
    ```

3.  **Create and activate the virtual environment (`.venv`):**
    This step installs all required dependencies. Required on the first time use only.

    ```bash
    # Create the environment and install dependencies
    uv venv
    uv sync
    ```

4.  **Activate the environment:**
    You must activate the virtual environment every time you open a new terminal session to run the script.
    * **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows (Command Prompt / PowerShell):**
        ```bash
        .\.venv\Scripts\activate
        ```

---

## 3. Prepare Data and Run

1.  **Add Images:**
    The script expects images to be in an `input` folder.
    **Create the `input` folder** at the root of the project (at the same level as the `src` folder) and place your images inside.
    *(Accepted formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`)*

2.  **Run the script:**
    ```bash
    python src/main.py
    ```

3.  **View Results:**
    Segmentation masks (as `.npz` files) and visualized results (as `.png` files) will be saved in the **`output`** folder.