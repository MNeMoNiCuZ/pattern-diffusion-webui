import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional
from diffusers import StableDiffusionPipeline, DDPMScheduler
import diffusers
from PIL import Image
import os
import re
import random
from datetime import datetime


def asymmetricConv2DConvForward_circular(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    self.paddingX = (
        self._reversed_padding_repeated_twice[0],
        self._reversed_padding_repeated_twice[1],
        0,
        0
    )

    self.paddingY = (
        0,
        0,
        self._reversed_padding_repeated_twice[2],
        self._reversed_padding_repeated_twice[3]
    )
    working = F.pad(input, self.paddingX, mode="circular")
    working = F.pad(working, self.paddingY, mode="circular")

    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


# Sets the padding mode to circular on Conv2d
def make_seamless(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if isinstance(module, diffusers.models.lora.LoRACompatibleConv) and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


# Sets the padding mode back to default on Conv2d
def disable_seamless(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if isinstance(module, diffusers.models.lora.LoRACompatibleConv) and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


# Runs every inference step
def diffusion_callback(pipe, step_index, timestep, callback_kwargs):
    # Sets unet and VAE to have circular padding on conv2d for last 20% of steps
    if step_index == int(pipe.num_timesteps * 0.8):
        make_seamless(pipe.unet)
        make_seamless(pipe.vae)

    # Noise Rolling: For the first 80% of steps, this shifts the noise slightly and wraps around the edge
    if step_index < int(pipe.num_timesteps * 0.8):
        callback_kwargs["latents"] = torch.roll(callback_kwargs["latents"], shifts=(64, 64), dims=(2, 3))

    return callback_kwargs

def process_prompt(prompt: str) -> str:
    # Process __wildcard__ syntax
    def replace_wildcard(match):
        wildcard_name = match.group(1)
        wildcard_file = os.path.join("wildcards", f"{wildcard_name}.txt")
        if os.path.exists(wildcard_file):
            with open(wildcard_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                return random.choice(lines)
        return match.group(0)  # Return original if file not found or empty

    prompt = re.sub(r"__([a-zA-Z0-9_]+)__", replace_wildcard, prompt)

    # Process {option1|option2|option3} syntax
    def replace_options(match):
        options = match.group(1).split('|')
        return random.choice(options)

    prompt = re.sub(r"\{([^\}]+)\}", replace_options, prompt)

    return prompt

def generate_pattern(
    prompt: str,
    output_file: str = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 50,
    seed: int = -1,
    batch_count: int = 1,
):
    processed_prompt = process_prompt(prompt)

    pipe = StableDiffusionPipeline.from_pretrained(
        "Arrexel/pattern-diffusion",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Make sure to disable circular padding on conv2d before starting inference as it should only be enabled in last 20% of steps
    # This is not necessary if you are only generating a single image (as it is disabled by default when the pipe loads)
    disable_seamless(pipe.unet)
    disable_seamless(pipe.vae)

    generator = None
    if seed != -1:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator("cuda").manual_seed(random.randint(0, 2**32 - 1))

    all_images = []
    for i in range(batch_count):
        output = pipe(
            num_inference_steps=num_inference_steps,
            prompt=processed_prompt,
            width=width,
            height=height,
            callback_on_step_end=diffusion_callback,
            generator=generator,
        ).images[0]
        all_images.append(output)

        if output_file is None:
            # Auto-save to output/[DATE:YY-MM-DD]/[DATE:YY-MM-DD - HH.MM.SS.MS].png
            current_time = datetime.now()
            date_folder = current_time.strftime("%Y-%m-%d")
            file_name = current_time.strftime("%Y-%m-%d - %H.%M.%S.%f")[:-3] + ".png" # Include milliseconds
            
            output_dir = os.path.join("output", date_folder)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            output.save(output_path)
            print(f"Image saved to: {output_path}")
        else:
            # If output_file is provided, save only the first image to that specific file
            if i == 0:
                output.save(output_file)

    return all_images, processed_prompt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate seamless patterns using Pattern Diffusion.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Vibrant watercolor floral pattern with pink, purple, and blue flowers against a white background.",
        help="The text prompt for image generation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None, # Changed default to None for auto-saving
        help="The name of the output image file. If not specified, image will be auto-saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of the generated image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of the generated image.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The random seed for generation. Use -1 for a random seed.",
    )
    parser.add_argument(
        "--batch_count",
        type=int,
        default=1,
        help="The number of images to generate in a batch.",
    )
    args = parser.parse_args()

    images, final_prompt = generate_pattern(
        prompt=args.prompt,
        output_file=args.output_file,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        batch_count=args.batch_count,
    )
    print(f"Generated {len(images)} image(s) with prompt: {final_prompt}")

