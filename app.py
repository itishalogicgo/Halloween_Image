# PyTorch 2.8 (temporary hack)
import os
os.system('pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 "torch<2.9" spaces')

# Actual demo code
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from optimization import optimize_pipeline_
import glob

# Preset Halloween themes that auto-fill the prompt when selected
THEME_PRESETS = {
    "Witch costume": "transform the person into a witch wearing a classic black hat and robe, add a moonlit night background",
    "Vampire": "make the person look like a vampire with pale skin, red eyes, and a dark cape, gothic castle background",
    "Ghost": "turn the person into a translucent ghost with a soft glow and misty background",
    "Zombie": "turn the person into a zombie with subtle decayed features and eerie lighting",
    "Skeleton": "stylize the person as a glowing skeleton with neon bones, dark background",
    "Pumpkin spirit": "add a pumpkin-head mask and autumn leaves background with warm cinematic lighting",
    "Monster": "turn the person into a friendly halloween monster with vibrant colors",
    "Pirate": "transform the person into a pirate with hat and eyepatch, wooden ship deck background",
    "Fairy": "make the person a halloween fairy with glowing wings and sparkly particles",
    "Cyberpunk": "apply a neon cyberpunk halloween theme with glowing accents and city night backdrop",
}

MAX_SEED = np.iinfo(np.int32).max
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token=os.getenv("HF_TOKEN")  # make sure HF_TOKEN is set in env
).to("cuda")

optimize_pipeline_(pipe, image=Image.new("RGB", (512, 512)), prompt='prompt')

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
    
    if input_image:
        input_image = input_image.convert("RGB")
        image = pipe(
            image=input_image, 
            prompt=prompt,
            guidance_scale=guidance_scale,
            width = input_image.size[0],
            height = input_image.size[1],
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    return image, seed, gr.Button(visible=True)

@spaces.GPU
def infer_example(input_image, prompt):
    image, seed, _ = infer(input_image, prompt)
    return image, seed

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
                    theme_dropdown = gr.Dropdown(
                        label="Theme",
                        choices=list(THEME_PRESETS.keys()),
                        value=None,
                        allow_custom_value=False,
                        scale=0,
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
        
            
        # Dynamically gather example images from current folder and 'Halloween Dress' subfolder
        image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        example_files: list[str] = []
        for pattern in image_patterns:
            example_files.extend(glob.glob(pattern))
            example_files.extend(glob.glob(os.path.join("Halloween Dress", pattern)))
        example_files = sorted(dict.fromkeys(example_files))  # de-duplicate while preserving order
        default_prompt = "make it spooky for halloween"
        dynamic_examples = [[path, default_prompt] for path in example_files]

        examples = gr.Examples(
            examples=dynamic_examples,
            inputs=[input_image, prompt],
            outputs=[result, seed],
            fn=infer_example,
            cache_examples="lazy"
        )
            
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [input_image, prompt, seed, randomize_seed, guidance_scale, steps],
        outputs = [result, seed, reuse_button]
    )
    
    def _apply_theme(theme_name: str):
        # Update the prompt textbox with the preset prompt corresponding to the selected theme
        return gr.update(value=THEME_PRESETS.get(theme_name, ""))

    theme_dropdown.change(
        fn=_apply_theme,
        inputs=[theme_dropdown],
        outputs=[prompt]
    )
    reuse_button.click(
        fn = lambda image: image,
        inputs = [result],
        outputs = [input_image]
    )

demo.launch(mcp_server=True)

