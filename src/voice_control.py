import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras import models
from queue import Queue, Empty

model = models.load_model("model/voice_model.keras", compile=False)
labels = ['down', 'left', 'right', 'up']
voice_active = False
change_to = 'RIGHT'

def record_audio():
    fs = 16000
    seconds = 1
    print("Listening...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def predict_command(treshold=0.7):
    try: 
        audio = record_audio()
        spec = get_spectrogram(audio)
        spec = tf.expand_dims(spec, 0)
        prediction = model.predict(spec)
        predicted_index = tf.argmax(prediction[0]).numpy()
        confidence = prediction[0][predicted_index]

        if confidence >= treshold:
            command = labels[predicted_index]
            print(f"Godkjent kommando: {command} (tillit: {confidence:.2f})")
            return command
        else:
            print(f"Usikker kommando ignorert (tillit: {confidence:.2f})")
            return None
    except Exception as e:
        print(f"Feil ved stemmegjenkjenning: {e}")
        return None

def voice_control():
    global change_to, voice_active
    while voice_active:
        command = predict_command()
        if command in labels:
            change_to = command.upper()
        time.sleep(0.1)