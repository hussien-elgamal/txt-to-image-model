import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

# Initialize and cache the model
@st.cache_resource
def load_model():
    try:
        # Load the model with float32 precision for CPU usage
        model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
        model.to("cpu")  # Use CPU
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Streamlit app layout
st.title("Stable Diffusion Image Generation")

# Text input for prompt
text_prompt = st.text_input("Enter a text prompt:", "A fantasy landscape")

# Button to generate image
if st.button("Generate Image"):
    if model:
        if text_prompt:
            try:
                # Generate image from text prompt
                with torch.no_grad():
                    result = model(text_prompt)
                    image = result.images[0]

                # Display the generated image
                st.image(image, caption="Generated Image", use_column_width=True)

                # Save the image to a file
                image.save("generated_image.png")

                # Success message
                st.success("Image generated and saved as 'generated_image.png'")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.error("Please enter a text prompt.")
    else:
        st.error("Model not loaded properly.")

# Add a footer with a disclaimer or additional information
st.markdown("**Note:** The quality and content of the generated image may vary based on the text prompt.")
