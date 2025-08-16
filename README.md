# Pattern Diffusion WebUI

<img width="1988" height="966" alt="image" src="https://github.com/user-attachments/assets/a02a0a26-2ef0-4132-bbda-be7524192933" />

## Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/ToriiGate-batch`
2. (Optional) Create a virtual environment for your setup. Use python 3.10 to 3.11. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

## Requirements
- Tested on Python 3.12.
- Tested on Pytorch 2.80 w. CUDA 12.8.
- At batch size 1 it uses ~11gb VRAM

## Instructions
Run `py app.py`

