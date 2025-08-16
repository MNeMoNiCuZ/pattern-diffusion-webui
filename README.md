# Pattern Diffusion WebUI

This tool provides a Gradio WebUI for the [Arrexel/pattern-diffusion](https://huggingface.co/Arrexel/pattern-diffusion) model, designed to generate high-quality, seamlessly tiling patterns.

The WebUI allows for easy control over generation parameters and features a powerful dynamic prompt system to create endless variations.

<img width="1988" height="966" alt="image" src="https://github.com/user-attachments/assets/a02a0a26-2ef0-4132-bbda-be7524192933" />

## Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/pattern-diffusion-webui`
2. (Optional) Create a virtual environment for your setup. Use python 3.10 to 3.11. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

## Requirements
- Tested on Python 3.12.
- Tested on Pytorch 2.80 w. CUDA 12.8.
- At batch size 1 it uses ~11gb VRAM

## How to Use

### Web UI
To start the main interface, run the following command in your terminal:
```bash
py app.py
```
Then, open the URL provided in the terminal (usually `http://127.0.0.1:7860`) in your web browser.

### Command Line Interface (CLI)
You can also generate patterns directly from the command line using `inference.py`.
```bash
# Example:
python inference.py --prompt "a seamless pattern of cute cats" --width 1024 --height 1024 --num_inference_steps 50

# For a full list of options:
python inference.py --help
```

## Features

### Seamless Pattern Generation
The core feature of this tool is its ability to create images that tile perfectly. This is achieved by modifying the diffusion model's convolutional layers to use circular padding during the final 20% of the inference steps, ensuring the edges of the generated image wrap around seamlessly.

### Dynamic Prompts
To generate a wide variety of patterns from a single base prompt, the tool supports special syntax to introduce randomness and variation.

#### Wildcards
You can insert `__wildcard__` into your prompt. The tool will replace this with a random line from the corresponding `wildcards/wildcard.txt` file.

> **Example Prompt:**
> `Vibrant floral pattern with __color__ and __color__ flowers and __objects__ against a __color__ background.`

This will pick random lines from `wildcards/color.txt` and `wildcards/objects.txt`. You can easily create your own wildcard files.

#### Options Syntax
Use curly braces `{}` with options separated by pipes `|` to have the script randomly select one of them.

> **Example Prompt:**
> `A {modern|vintage|retro}, {geometric|floral|abstract} pattern.`

This will randomly choose between "modern", "vintage", or "retro", and between "geometric", "floral", or "abstract".

### Web UI Controls
- **Prompt**: The text describing the desired pattern. Supports all dynamic prompt features.
- **Width/Height**: The dimensions of the output image in pixels.
- **Number of Inference Steps**: How many steps the model takes to generate the image. Higher values can improve quality but take longer.
- **Seed**: A number to control the random generation. Use `-1` for a random seed. Using the same seed with the same prompt will produce the same image.
- **Batch Count**: The number of images to generate in a single run.
- **Generate**: Runs the generation process once for the specified batch count.
- **Generate Forever**: Continuously generates new images until the "Cancel" button is pressed. This is useful for quickly exploring different random variations of a prompt.
- **Cancel**: Stops the "Generate Forever" loop.
- **Processed Prompt**: Displays the final prompt after all wildcards and options have been resolved, showing you exactly what prompt was used for the generated image.