from concurrent.futures import ProcessPoolExecutor
import logging
import sys
import gradio as gr
from tempfile import NamedTemporaryFile
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
import torch
import time
import warnings

from demos.FileCleaner import FileCleaner
from demos.LogSuppressor import LogSuppressor
from demos.ModelLoader import ModelLoader


pool = ProcessPoolExecutor(4)
pool.__enter__()

file_cleaner = FileCleaner()
log_suppressor = LogSuppressor()
log_suppressor.suppress_subprocess_logging()

model_loader = ModelLoader("facebook/musicgen-stereo-melody")
INTERRUPTING = False

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

def make_waveform(*args, **kwargs):
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out

def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None):
    model_loader.model.set_generation_params(duration=duration)
    target_sr = 32000
    processed_melodies = _process_melodies(melodies, duration, target_sr)
    
    try:
        if any(m is not None for m in processed_melodies):
            outputs = model_loader.model.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=False
            )
        else:
            outputs = model_loader.model.generate(texts, progress=progress, return_tokens=False)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])

    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, model_loader.model.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    return out_videos[0], out_wavs[0], None, None


def _process_melodies(melodies, duration, target_sr=32000, target_ac=1):
    processed_melodies = []
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(model_loader.model.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)
    return processed_melodies


def predict_full(text, melody, duration, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_loader.load_model()

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    model_loader.model.set_custom_progress_callback(_progress)

    return _do_predictions([text], [melody], duration, progress=True, gradio_progress=progress)

def create_ui():
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Endabgabe
            Dies ist zum Musik generieren
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    with gr.Column():
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                    interrupt_button = gr.Button("Interrupt")
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_full, inputs=[text, melody, duration],
                                               outputs=[output, audio_output])
        interrupt_button.click(interrupt, queue=False)

        interface.queue().launch()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    create_ui()
