import torch
from torch import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import base64
from io import BytesIO
import os

def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    # repo = 'stabilityai/stable-diffusion-2-1-base'
    repo = 'prompthero/openjourney-v4'
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
    model = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, local_files_only=True,
                                              revision="main", #scheduler=scheduler,
                                               use_auth_token=HF_AUTH_TOKEN).to("cuda")    
    model.enable_xformers_memory_efficient_attention()
    
def inference(model_inputs:dict):
    global model
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    steps = model_inputs.get('steps', 25)
    neg_prompt = model_inputs.get('negative_prompt', "bad anatomy, low quality, ugly, blurry")
    guidance_scale = model_inputs.get('guidance_scale', 8)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        image = model(prompt, guidance_scale=guidance_scale, 
            negative_prompt=neg_prompt, height=height, width=width, 
            num_inference_steps=steps, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
