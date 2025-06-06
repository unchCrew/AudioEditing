import gradio as gr
import shutil
import os
from AudioEditing.gradio import *
import warnings


paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])

def change_audios_choices(input_audio): 
    audios = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), "choices": audios, "__type__": "update"}

def download_url(url):
    import yt_dlp

    if not url: return gr_warning("please provide url")
    if not os.path.exists("audios"): os.makedirs("audios", exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }],
            "quiet": True,            
            "no_warnings": True,
            "noplaylist": True,
            "verbose": False,
            "cookiefile": "assets/yt-dlp/config.txt"
        }
        gr_info("start".format(start="downloading music"))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join("audios", re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))
            if os.path.exists(audio_output): shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = audio_output + ".wav"
            if os.path.exists(audio_output): os.remove(audio_output)
            
            ydl.download([url])

        gr_info("success")
        return [audio_output, audio_output, "success"]


def utils_tabs():
    with gr.TabItem("Downloading Audio"):
        link_input = gr.Textbox(label="Download Youtube Audio")
        download_btn = gr.Button(" Download")
        download_btn.click(fn=download_url, inputs=link_input, outputs=None)
    with gr.TabItem("Audio Editing"):
        gr.Markdown("# Audio Editing Interface\nModify audio files with AudioLDM2 model.")
        
        # Input Section
        with gr.Group():
            gr.Markdown("**Upload Audio File**")
            with gr.Row(equal_height=True):
                drop_audio_file = gr.File(
                    label="Upload Audio",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"],
                    interactive=True
                )
                display_audio = gr.Audio(
                    label="Preview Input Audio",
                    show_download_button=True,
                    interactive=False
                )
        
        # Settings Section
        with gr.Accordion("Input/Output Settings", open=True):
            with gr.Row():
                input_audiopath = gr.Dropdown(
                    label="Audio Path",
                    choices=paths_for_files,
                    value="",
                    allow_custom_value=True,
                    info="Select or enter the path to the input audio.",
                    interactive=True
                )
                output_audiopath = gr.Textbox(
                    label="Output Path",
                    value="audios/output.wav",
                    placeholder="audios/output.wav",
                    info="Specify the path for the output audio.",
                    interactive=True
                )
            export_audio_format = gr.Radio(
                label="Export Format",
                choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"],
                value="wav",
                info="Choose the output audio format.",
                interactive=True
            )
            refresh_audio = gr.Button("Refresh Audio Paths")

        # Audio Editing Parameters
        with gr.Group():
            gr.Markdown("**Editing Parameters**")
            with gr.Row():
                with gr.Column():
                    tar_prompt = gr.Textbox(
                        label="Target Prompt",
                        placeholder="e.g., Piano and violin cover",
                        lines=3,
                        info="Describe the desired audio output.",
                        interactive=True
                    )
                    src_prompt = gr.Textbox(
                        label="Source Prompt",
                        placeholder="e.g., A recording of a happy upbeat classical music piece",
                        lines=2,
                        info="Describe the input audio.",
                        interactive=True
                    )
                with gr.Column():
                    cfg_scale_src = gr.Slider(
                        value=3,
                        minimum=0.5,
                        maximum=25,
                        label="Source CFG Scale",
                        info="Controls adherence to the source audio.",
                        interactive=True
                    )
                    cfg_scale_tar = gr.Slider(
                        value=12,
                        minimum=0.5,
                        maximum=25,
                        label="Target CFG Scale",
                        info="Controls adherence to the target prompt.",
                        interactive=True
                    )

        # Advanced Settings
        with gr.Accordion("Advanced Settings", open=False):
            audioldm2_model = gr.Radio(
                label="AudioLDM2 Model",
                choices=["audioldm2", "audioldm2-large", "audioldm2-music"],
                value="audioldm2-music",
                info="Select the AudioLDM2 model variant.",
                interactive=True
            )
            audioldm2_sample_rate = gr.Slider(
                minimum=8000,
                maximum=96000,
                value=44100,
                step=1,
                label="Sample Rate (Hz)",
                info="Set the audio sample rate.",
                interactive=True
            )
            t_start = gr.Slider(
                minimum=0,
                maximum=85,
                value=45,
                step=1,
                label="Start Time (seconds)",
                info="Start time for audio processing.",
                interactive=True
            )
            steps = gr.Slider(
                value=50,
                minimum=10,
                maximum=300,
                step=1,
                label="Processing Steps",
                info="Number of processing steps.",
                interactive=True
            )
            save_compute = gr.Checkbox(
                label="Save Compute Resources",
                value=True,
                interactive=True
            )

        # Action and Output
        with gr.Row():
            edit_button = gr.Button("Edit Audio", variant="primary")
        gr.Markdown("**Output Audio**")
        output_audioldm2 = gr.Audio(
            label="Edited Audio",
            show_download_button=True,
            interactive=False
        )

        # Event Handlers
        refresh_audio.click(
            fn=change_audios_choices,
            inputs=[input_audiopath],
            outputs=[input_audiopath]
        )
        drop_audio_file.upload(
            fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")),
            inputs=[drop_audio_file],
            outputs=[input_audiopath]
        )
        input_audiopath.change(
            fn=lambda audio: audio if os.path.isfile(audio) else None,
            inputs=[input_audiopath],
            outputs=[display_audio]
        )
        edit_button.click(
            fn=run_audioldm2,
            inputs=[
                input_audiopath,
                output_audiopath,
                export_audio_format,
                audioldm2_sample_rate,
                audioldm2_model,
                src_prompt,
                tar_prompt,
                steps,
                cfg_scale_src,
                cfg_scale_tar,
                t_start,
                save_compute
            ],
            outputs=[output_audioldm2],
            api_name="audioldm2"
        )
