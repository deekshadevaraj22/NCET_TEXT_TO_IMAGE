import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load model
@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

pipe = load_model()

# UI
st.title("🎨 AI Text to Image Generator")
st.write("Enter a prompt and generate an image!")

prompt = st.text_input("Enter your image prompt:")

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image... ⏳"):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image")
    else:
        st.warning("⚠️ Please enter a prompt.")
