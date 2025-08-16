import gradio as gr
from inference import generate_pattern, process_prompt
import time

def generate_single_image(prompt, width, height, num_inference_steps, seed, batch_count):
    # The generate_pattern function already handles saving to the dated output directory
    # and returns the PIL Image object.
    images, processed_prompt = generate_pattern(prompt, None, width, height, num_inference_steps, seed, batch_count)
    return images, processed_prompt

def generate_forever_loop(prompt, width, height, num_inference_steps, seed, batch_count, should_stop_state):
    while not should_stop_state.value:
        # Re-read the prompt each time for dynamic updates
        current_prompt = prompt
        images, processed_prompt = generate_pattern(current_prompt, None, width, height, num_inference_steps, seed, batch_count)
        yield images, processed_prompt # Yield the image and processed prompt to update the UI
        time.sleep(0.1) # Small delay to prevent busy-waiting

def get_processed_prompt(prompt):
    return process_prompt(prompt)

with gr.Blocks(title="Pattern Diffusion GUI") as demo:
    gr.Markdown("# Pattern Diffusion GUI")
    gr.Markdown("Generate seamless patterns using the Pattern Diffusion model.")

    should_stop = gr.State(False)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                value="Vibrant floral pattern with __color__ and __color__ flowers and __objects__ against a {dotted|striped|diagonal|__color__|grid} background.",
                lines=3,
            )
            width_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Width")
            height_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Height")
            steps_slider = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Number of Inference Steps")
            with gr.Row():
                seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                batch_count_input = gr.Number(label="Batch Count", value=1, minimum=1, precision=0)

            with gr.Row() as button_row:
                generate_btn = gr.Button("Generate", variant="primary")
                generate_forever_btn = gr.Button("Generate Forever", variant="primary")
            
            cancel_btn = gr.Button("Cancel", visible=False, variant="stop")
            
            processed_prompt_display = gr.Textbox(label="Processed Prompt", interactive=False, lines=1)

        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Patterns", show_label=True, elem_id="gallery")

    # Event Handlers
    # Single Generation
    generate_btn.click(
        fn=lambda: {
            generate_btn: gr.update(visible=False),
            generate_forever_btn: gr.update(visible=False),
        },
        outputs=[generate_btn, generate_forever_btn],
    ).then(
        fn=get_processed_prompt,
        inputs=[prompt_input],
        outputs=[processed_prompt_display],
    ).then(
        fn=generate_single_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, batch_count_input],
        outputs=[output_gallery, processed_prompt_display],
    ).then(
        fn=lambda: {
            generate_btn: gr.update(visible=True),
            generate_forever_btn: gr.update(visible=True),
        },
        outputs=[generate_btn, generate_forever_btn],
    )

    # Generate Forever
    generate_forever_btn.click(
        fn=lambda s: gr.State(False), # Reset stop flag
        inputs=[should_stop],
        outputs=should_stop,
    ).then(
        fn=lambda: {
            generate_btn: gr.update(visible=False),
            generate_forever_btn: gr.update(visible=False),
            cancel_btn: gr.update(visible=True),
        },
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
    ).then(
        fn=get_processed_prompt,
        inputs=[prompt_input],
        outputs=[processed_prompt_display],
    ).then(
        fn=generate_forever_loop,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, batch_count_input, should_stop],
        outputs=[output_gallery, processed_prompt_display],
    ).then(
        fn=lambda: {
            generate_btn: gr.update(visible=True),
            generate_forever_btn: gr.update(visible=True),
            cancel_btn: gr.update(visible=False),
        },
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
    )

    # Cancel Button
    cancel_btn.click(
        fn=lambda s: gr.State(True), # Set stop flag
        inputs=[should_stop],
        outputs=should_stop,
    ).then(
        fn=lambda: {
            generate_btn: gr.update(visible=True),
            generate_forever_btn: gr.update(visible=True),
            cancel_btn: gr.update(visible=False),
        },
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
    )

demo.launch()

def generate_and_display(prompt, width, height, num_inference_steps):
    # Generate a temporary output file name for Gradio
    output_file = "temp_output.png"
    generated_image = generate_pattern(prompt, output_file, width, height, num_inference_steps)
    return generated_image

def generate_single_image(prompt, width, height, num_inference_steps, seed, batch_count):
    # The generate_pattern function already handles saving to the dated output directory
    # and returns the PIL Image object.
    images, processed_prompt = generate_pattern(prompt, None, width, height, num_inference_steps, seed, batch_count)
    return images, processed_prompt

def generate_forever_loop(prompt, width, height, num_inference_steps, seed, batch_count, should_stop_state):
    while not should_stop_state.value:
        # Re-read the prompt each time for dynamic updates
        current_prompt = prompt
        images, processed_prompt = generate_pattern(current_prompt, None, width, height, num_inference_steps, seed, batch_count)
        yield images, processed_prompt # Yield the image and processed prompt to update the UI
        time.sleep(0.1) # Small delay to prevent busy-waiting

with gr.Blocks(title="Pattern Diffusion GUI") as demo:
    gr.Markdown("# Pattern Diffusion GUI")
    gr.Markdown("Generate seamless patterns using the Pattern Diffusion model.")

    should_stop = gr.State(False)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                value="Vibrant watercolor floral pattern with pink, purple, and blue flowers against a white background.",
                lines=3,
            )
            width_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Width")
            height_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Height")
            steps_slider = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Number of Inference Steps")
            with gr.Row():
                seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                batch_count_input = gr.Number(label="Batch Count", value=1, minimum=1, precision=0)

            with gr.Row() as button_row:
                generate_btn = gr.Button("Generate", variant="primary")
                generate_forever_btn = gr.Button("Generate Forever", variant="primary")
            
            cancel_btn = gr.Button("Cancel", visible=False, variant="stop")
            
            processed_prompt_display = gr.Textbox(label="Processed Prompt", interactive=False, lines=1)

        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Patterns", show_label=True, elem_id="gallery")

    # Event Handlers
    def set_buttons_generating():
        return {
            generate_btn: gr.update(visible=False),
            generate_forever_btn: gr.update(visible=False),
            cancel_btn: gr.update(visible=True),
        }

    def set_buttons_idle():
        return {
            generate_btn: gr.update(visible=True),
            generate_forever_btn: gr.update(visible=True),
            cancel_btn: gr.update(visible=False),
        }

    # Single Generation
    generate_btn.click(
        fn=lambda: {
            generate_btn: gr.update(visible=False),
            generate_forever_btn: gr.update(visible=False),
        },
        outputs=[generate_btn, generate_forever_btn],
        queue=False
    ).then(
        fn=generate_single_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, batch_count_input],
        outputs=[output_gallery, processed_prompt_display],
    ).then(
        fn=lambda: {
            generate_btn: gr.update(visible=True),
            generate_forever_btn: gr.update(visible=True),
        },
        outputs=[generate_btn, generate_forever_btn],
        queue=False
    )

    # Generate Forever
    generate_forever_btn.click(
        fn=lambda: False, # Reset stop flag
        inputs=None,
        outputs=should_stop,
        queue=False
    ).then(
        fn=set_buttons_generating,
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
        queue=False
    ).then(
        fn=generate_forever_loop,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, batch_count_input, should_stop],
        outputs=[output_gallery, processed_prompt_display],
    ).then(
        fn=set_buttons_idle,
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
        queue=False
    )

    # Cancel Button
    cancel_btn.click(
        fn=lambda: True, # Set stop flag
        inputs=None,
        outputs=should_stop,
        queue=False
    ).then(
        fn=set_buttons_idle,
        outputs=[generate_btn, generate_forever_btn, cancel_btn],
        queue=False
    )

demo.launch()