import gradio as gr
import numpy as np

import spaces
import torch
import random
from PIL import Image

from kontext_pipeline import FluxKontextPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image

from huggingface_hub import hf_hub_download


kontext_path = hf_hub_download(repo_id="diffusers/kontext", filename="kontext.safetensors")

MAX_SEED = np.iinfo(np.int32).max

transformer = FluxTransformer2DModel.from_single_file(kontext_path, torch_dtype=torch.bfloat16)
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

@spaces.GPU
def infer(input_image, prompt, seed=42, randomize_seed=False, guidance_scale=2.5, progress=gr.Progress(track_tqdm=True)):
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    input_image = input_image.convert("RGB")
    original_width, original_height = input_image.size
    
    if original_width >= original_height:
        new_width = 1024
        new_height = int(original_height * (new_width / original_width))
        new_height = round(new_height / 64) * 64
    else:
        new_height = 1024
        new_width = int(original_width * (new_height / original_height))
        new_width = round(new_width / 64) * 64
    
    input_image_resized = input_image.resize((new_width, new_height), Image.LANCZOS)
    image = pipe(
        image=input_image_resized, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        width=new_width,
        height=new_height,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    return image, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Kontext [dev]
        """)

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
        
        result = gr.Image(label="Result", show_label=False)
        
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

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [input_image, prompt, seed, randomize_seed, guidance_scale],
        outputs = [result, seed]
    )

demo.launch()