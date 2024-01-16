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

genre_prompt = {
    "Classic": "Violin, Cello, Flute, Clarinet, Piano with Orchestral arrangements, intricate counterpoint, use of classical forms (sonata, symphony, etc.), dynamic contrast.",
"Metal": "Kick Drum, Reverse Bass with Build-ups and Drops, Energetic Melodies, Hard Percussion, Electric Guitar, Bass Guitar, Drum Kit, with Guitar Riffs, Double Bass Drumming, Distorted Sound, Time Signature Changes",
"Jazz": "Saxophone, Trumpet, Piano, Double Bass, Drum Kit with Improvisation, swing rhythm, syncopation, extended chords, call and response. Maintain the original song's melodic structure while infusing it with a classic jazz feel",
"Country": "Acoustic Guitar, Banjo, Fiddle, Pedal Steel Guitar with Twangy guitar, country scales, simple chord progressions, folk influences.",
"EDM": "Synthesizers, Drum Machines, Sampler with Electronic beats, synthesizer melodies, drops, build-ups, effects, rhythmic complexity.",
"Hardstyle/Eurodance": "Kick Drum, Synthesizers and Reverse Bass with Build-ups and Drops, Energetic Melodies, Pitched Vocals, Hard Percussion",
"Reggae": "Guitar, Bass, Drums, Organ with Offbeat rhythm (skank), dub effects",
"Rock": "Electric Guitar, Bass Guitar, Drum Kit with Power chords, guitar solos, strong backbeat, distortion, energetic performance."
}

input_file_path = "./assets/input_files/"
music_files = {
    "Bach": "bach.mp3",
    "Bolero": "bolero_ravel.mp3",
    "Eye of the Tiger": "eye_of_the_tiger.mp3",
    "Let it go Long": "let_it_go_input1.mp3",
    "Let it go": "let_it_go_input2.mp3",
    "Pokerface": "pokerface_input_long.mp3",
    "Pokerface Long": "pokerface_input1.mp3",
    "Game of Thrones": "game_of_thrones_input1.mp3"
}

def melody_change(file):
    return input_file_path + music_files.get(file)

def make_waveform(*args, **kwargs):
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def _do_predictions(genre, melodies, duration = 10, progress=False, gradio_progress=None):
    model_loader.model.set_generation_params(duration=duration)
    target_sr = 32000
    processed_melodies = _process_melodies(melodies, duration, target_sr)
    input_prompt = [genre_prompt.get(genre)]
    
    try:
        if any(m is not None for m in processed_melodies):
            outputs = model_loader.model.generate_with_chroma(
                descriptions=input_prompt,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=False
            )
        else:
            outputs = model_loader.model.generate([input_prompt], progress=progress, return_tokens=False)
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


def predict_full(genre, melody, progress=gr.Progress()):
    progress(0, desc="Loading model...")
    model_loader.load_model()

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))

    model_loader.model.set_custom_progress_callback(_progress)

    return _do_predictions(genre, [melody], progress=True, gradio_progress=progress)


def create_ui():
    css = """
    .container {
        height: 50px;
    }
    """

    with gr.Blocks() as demo:
        with gr.Column():
            gr.HTML(value="<div style = 'display: flex; justify-content: space-between; align-items: center'><img src = 'file/logo.png' alt='logo' width='150px'/><div style='font-size: 32px'>GenreMorpher</div><img src = 'https://upload.wikimedia.org/wikipedia/de/5/57/Hochschule_Furtwangen_HFU_logo.svg' alt='hfu_logo' width='300px'/></div>")
            with gr.Row(elem_classes=["container"]):
                melody = gr.Audio(label="Upload your audio", interactive=True, sources=["upload"], )
                otherMelody = gr.Dropdown(music_files.keys(), label="Or choose one of our audios")
            genre = gr.Dropdown(genre_prompt.keys(), label="Select Genre", value="Metal")
            btn = gr.Button(value="Generate")
            
            output = gr.Video(label="Generated Music", interactive=False, elem_classes=["container"])
            audio_output = gr.Audio(label="Generated Music (wav)", type='filepath', interactive=False)
            otherMelody.change(fn=melody_change, inputs=otherMelody, outputs=melody)
            btn.click(predict_full, inputs=[genre, melody],
                                               outputs=[output, audio_output])
            gr.Markdown(
            """
            [GenreMorpher](https://github.com/larimei/Music-KI) is a project for the course AI in Music
            from the Hochschule Furtwangen.
            This app changes the style and genre of the uploaded audio file and 
            generates a new one.
            Created by Jonathan Rissler, Lennard Hurst and Lara Meister
            """
        )

            demo.queue().launch(share=False, allowed_paths=["logo.png"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    create_ui()
