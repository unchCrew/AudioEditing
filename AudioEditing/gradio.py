import os
import gradio as gr


def gr_info(message):
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=6)
    logger.error(message)



def run_audioldm2(input_path, output_path, export_format, sample_rate, audioldm_model, source_prompt, target_prompt, steps, cfg_scale_src, cfg_scale_tar, t_start, save_compute):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning("where the  audio input?")
        return None
        
    if not output_path:
        gr_warning("output audio not valid")
        return None
    
    output_path = output_path.replace("wav", export_format)

    if os.path.exists(output_path): os.remove(output_path)

    gr_info("start edit".format(input_path=input_path)
    subprocess.run([python, "main/inference/audioldm2.py", "--input_path", input_path, "--output_path", output_path, "--export_format", str(export_format), "--sample_rate", str(sample_rate), "--audioldm_model", audioldm_model, "--source_prompt", source_prompt, "--target_prompt", target_prompt, "--steps", str(steps), "--cfg_scale_src", str(cfg_scale_src), "--cfg_scale_tar", str(cfg_scale_tar), "--t_start", str(t_start), "--save_compute", str(save_compute)])
    
    gr_info("success!")
    return output_path
