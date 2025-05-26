import gradio as ae
from tabs.main import utils_tabs
import os, sys

sys.path.append(os.getcwd())

with gr.Blocks(title=" Audio Editing", theme="Thatguy099/Sonix") as app:
    
    utils_tabs()

app.launch(share=True)
