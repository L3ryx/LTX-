import streamlit as st
import torch
from ltx2.inference import load_model, generate_video

st.title("LTX-2 Video Generator")

@st.cache_resource
def init_model():
    model = load_model()
    return model

model = init_model()

prompt = st.text_input("Enter prompt")

if st.button("Generate"):
    with st.spinner("Generating..."):
        video = generate_video(model, prompt)

        st.video(video)
