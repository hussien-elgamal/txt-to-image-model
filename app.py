import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter
import io

# Model and device setup
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Use CPU if GPU is not available

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token="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"
)
pipe.to(device)

# Function to generate and enhance images
def generate(prompt, guidance_scale=10.0):
    # Set a random seed for consistent results
    generator = torch.Generator(device).manual_seed(1024)
    
    with autocast(device):
        # Generate the image from the text prompt
        image = pipe(prompt, guidance_scale=guidance_scale, generator=generator).images[0]
    
    # Enhance the image by applying a sharpening filter
    image = image.filter(ImageFilter.SHARPEN)
    return image

# Streamlit app
st.title('Image Generation with Stable Diffusion')

# User input for prompt
prompt = st.text_input('Enter your description:', 'river nile with the pyramids great view')

if st.button('Generate Image'):
    if prompt:
        image = generate(prompt)
        st.image(image, caption='Generated Image', use_column_width=True)

        # Save the image to a buffer for downloading
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        st.download_button(label="Download Image", data=buffer, file_name='generatedimage.png', mime='image/png')
