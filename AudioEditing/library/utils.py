import os
import re
import sys
import codecs
import librosa
import logging

import numpy as np
import soundfile as sf

from pydub import AudioSegment

sys.path.append(os.getcwd())

from AudioEditing.tools import huggingface

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def check_audioldm2(model):
    for f in ["feature_extractor", "language_model", "projection_model", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae", "vocoder"]:
        folder_path = os.path.join("AudioEditing", "models", "audioldm2", model, f)

        if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)

    for f in ["feature_extractor/preprocessor_config.json","language_model/config.json","language_model/model.safetensors","model_index.json","projection_model/config.json","projection_model/diffusion_pytorch_model.safetensors","scheduler/scheduler_config.json","text_encoder/config.json","text_encoder/model.safetensors","text_encoder_2/config.json","text_encoder_2/model.safetensors","tokenizer/merges.txt","tokenizer/special_tokens_map.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/vocab.json","tokenizer_2/special_tokens_map.json","tokenizer_2/spiece.model","tokenizer_2/tokenizer.json","tokenizer_2/tokenizer_config.json","unet/config.json","unet/diffusion_pytorch_model.safetensors","vae/config.json","vae/diffusion_pytorch_model.safetensors","vocoder/config.json","vocoder/model.safetensors"]:
        model_path = os.path.join("AudioEditing", "models", "audioldm2", model, f)

        if not os.path.exists(model_path): huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/nhqvbyqz/", "rot13"), model, "/", f]), model_path)
