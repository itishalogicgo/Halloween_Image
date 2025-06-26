import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from pipeline_flux_kontext import FluxKontextPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

MAX_SEED = np.iinfo(np.int32).max

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", revision="refs/pr/2", torch_dtype=torch.bfloat16).to("cuda")

@spaces.GPU
def infer(input_image, prompt, seed=42, randomize_seed=False, guidance_scale=2.5, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing using the FLUX.1 Kontext pipeline.
    
    This function takes an input image and a text prompt to generate a modified version
    of the image based on the provided instructions. It uses the FLUX.1 Kontext model
    for contextual image editing tasks.
    
    Args:
        input_image (PIL.Image.Image): The input image to be edited. Will be converted
            to RGB format if not already in that format.
        prompt (str): Text description of the desired edit to apply to the image.
            Examples: "Remove glasses", "Add a hat", "Change background to beach".
        seed (int, optional): Random seed for reproducible generation. Defaults to 42.
            Must be between 0 and MAX_SEED (2^31 - 1).
        randomize_seed (bool, optional): If True, generates a random seed instead of
            using the provided seed value. Defaults to False.
        guidance_scale (float, optional): Controls how closely the model follows the
            prompt. Higher values mean stronger adherence to the prompt but may reduce
            image quality. Range: 1.0-10.0. Defaults to 2.5.
        steps (int, optional): Controls how many steps to run the diffusion model for.
            Range: 1-30. Defaults to 28.
        progress (gr.Progress, optional): Gradio progress tracker for monitoring
            generation progress. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A 3-tuple containing:
            - PIL.Image.Image: The generated/edited image
            - int: The seed value used for generation (useful when randomize_seed=True)
            - gr.update: Gradio update object to make the reuse button visible
    
    Example:
        >>> edited_image, used_seed, button_update = infer(
        ...     input_image=my_image,
        ...     prompt="Add sunglasses",
        ...     seed=123,
        ...     randomize_seed=False,
        ...     guidance_scale=2.5
        ... )
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    input_image = input_image.convert("RGB")
    image = pipe(
        image=input_image, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    return image, seed, gr.update(visible=True)

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Kontext [dev]
Image editing and manipulation model guidance-distilled from FLUX.1 Kontext [pro], [[blog]](https://bfl.ai/announcements/flux-1-kontext-dev) [[model]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload the image for editing", type="pil")
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt for editing (e.g., 'Remove glasses', 'Add a hat')",
                        container=False,
                    )
                    run_button = gr.Button("Run", scale=0)
                with gr.Accordion("Advanced Settings", open=False):
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )       
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=30,
                        value=28,
                        step=1
                    )
                    
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image", visible=False)
        
        
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [input_image, prompt, seed, randomize_seed, guidance_scale, steps],
        outputs = [result, seed, reuse_button]
    )
    reuse_button.click(
        fn = lambda image: image,
        inputs = [result],
        outputs = [input_image]
    )

demo.launch(mcp_server=True)