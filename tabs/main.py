import gradio as gr
from AudioEditing.gradio import *

def utils_tabs():
    with gr.TabItem("utils"):
        with gr.Tabs():
            with gr.TabItem("Audio Editing", visible=configs.get("audioldm2", True)):
                gr.Markdown("Audio editing interface for modifying audio files.")
                with gr.Row():
                    gr.Markdown("Upload an audio file and specify the desired output.")
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                save_compute = gr.Checkbox(label="Save Compute", value=True, interactive=True)
                            tar_prompt = gr.Textbox(label="Target Prompt", info="Describe the desired audio output.", placeholder="Piano and violin cover", lines=5, interactive=True)
                    with gr.Column():
                        cfg_scale_src = gr.Slider(value=3, minimum=0.5, maximum=25, label="Source CFG Scale", info="Controls adherence to the source audio.", interactive=True)
                        cfg_scale_tar = gr.Slider(value=12, minimum=0.5, maximum=25, label="Target CFG Scale", info="Controls adherence to the target prompt.", interactive=True)
                with gr.Row():
                    edit_button = gr.Button("Edit Audio", variant="primary")
                with gr.Row():
                    with gr.Column():
                        drop_audio_file = gr.File(label="Upload Audio", file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])  
                        display_audio = gr.Audio(show_download_button=True, interactive=False, label="Input Audio")
                    with gr.Column():
                        with gr.Accordion("Input/Output Settings", open=False):
                            with gr.Column():
                                export_audio_format = gr.Radio(label="Export Format", info="Choose the output audio format.", choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                                input_audiopath = gr.Dropdown(label="Audio Path", value="", choices=paths_for_files, info="Select or enter the path to the input audio.", allow_custom_value=True, interactive=True)
                                output_audiopath = gr.Textbox(label="Output Path", value="audios/output.wav", placeholder="audios/output.wav", info="Specify the path for the output audio.", interactive=True)
                            with gr.Column():
                                refesh_audio = gr.Button("Refresh Audio")
                        with gr.Accordion("Advanced Settings", open=False):
                            audioldm2_model = gr.Radio(label="AudioLDM2 Model", info="Select the AudioLDM2 model variant.", choices=["audioldm2", "audioldm2-large", "audioldm2-music"], value="audioldm2-music", interactive=True)
                            with gr.Row():
                                src_prompt = gr.Textbox(label="Source Prompt", lines=2, interactive=True, info="Describe the input audio.", placeholder="A recording of a happy upbeat classical music piece")
                            with gr.Row():
                                with gr.Column(): 
                                    audioldm2_sample_rate = gr.Slider(minimum=8000, maximum=96000, label="Sample Rate", info="Set the audio sample rate (Hz).", value=44100, step=1, interactive=True)
                                    t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label="Start Time", interactive=True, info="Start time for audio processing (seconds).")
                                    steps = gr.Slider(value=50, step=1, minimum=10, maximum=300, label="Steps", info="Number of processing steps.", interactive=True)
                with gr.Row():
                    gr.Markdown("Output Audio")
                with gr.Row():
                    output_audioldm2 = gr.Audio(show_download_button=True, interactive=False, label="Output Audio")
                with gr.Row():
                    refesh_audio.click(fn=change_audios_choices, inputs=[input_audiopath], outputs=[input_audiopath])
                    drop_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[drop_audio_file], outputs=[input_audiopath])
                    input_audiopath.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audiopath], outputs=[display_audio])
                with gr.Row():
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
