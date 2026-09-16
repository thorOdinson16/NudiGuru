import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import gradio as gr

# Load Whisper (use medium/large for Kannada)
model = whisper.load_model("medium")

audio_q = queue.Queue()

# Audio callback (push frames into queue)
def audio_callback(indata, frames, time, status):
    audio_q.put(indata.copy())

# Worker thread → convert queued audio to text
def transcribe_stream(callback):
    buffer = np.zeros((0, 1))
    samplerate = 16000

    while True:
        data = audio_q.get()
        buffer = np.concatenate((buffer, data))

        # Process every 1 second
        if buffer.shape[0] >= samplerate:
            samples = buffer.flatten().astype(np.float32)
            buffer = np.zeros((0, 1))

            audio = whisper.pad_or_trim(samples)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            result = model.decode(mel)
            text = result.text.strip()

            if text != "":
                callback(text)

# GUI handling
def start_listening():
    sd.default.samplerate = 16000
    sd.default.channels = 1

    sd.InputStream(callback=audio_callback).start()

    def generator():
        text_container = ""
        def update_text(t):
            nonlocal text_container
            text_container = t
        threading.Thread(target=transcribe_stream, args=(update_text,), daemon=True).start()

        while True:
            yield text_container

    return generator()

# Gradio UI
demo = gr.Interface(
    fn=start_listening,
    inputs=[],
    outputs=gr.Textbox(label="Live Speech-to-Text"),
    live=True
)

demo.launch()