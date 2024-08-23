import subprocess
import sys
import streamlit as st
from torch import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Function to install packages
def install_packages():
    packages = [
        'transformers',
        'diffusers',
        'accelerate',
        'torch',
        'Pillow'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Run the install function
install_packages()

# Streamlit app
st.title("Text to Image Model")
st.write("Welcome to the text-to-image model app!")

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    # Load Stable Diffusion model
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    model.to("cuda")  # Use GPU if available
    return model

# Initialize model
model = load_model()

# Text input from user
text_prompt = st.text_input("Enter a text prompt:", "A fantasy landscape")

# Generate image button
if st.button("Generate Image"):
    with autocast("cuda"):
        # Generate image
        image = model(text_prompt).images[0]
        
        # Display image
        st.image(image, caption="Generated Image")

# Option to download the image
if st.button("Download Image"):
    with autocast("cuda"):
        # Generate image again
        image = model(text_prompt).images[0]
        
        # Save image to file
        image.save("generated_image.png")
        
        # Provide download link
        st.download_button("Download Image", "generated_image.png", "image/png")
