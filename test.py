import numpy as np
import tensorflow as tf
import keras
from datasets import Dataset, load_dataset
import librosa

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Numpy version:", np.__version__)

dataset = load_dataset("google/speech_commands", "v0.01")

# Overview of the dataset
print(dataset)

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# Check
# print(train_dataset[0])
# print(validation_dataset[0])
# print(test_dataset[0])

def preprocess_audio(set):

    audio_array = set['audio']['array']
    sampling_rate = set['audio']['sampling_rate']

    mel_spectogram = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, n_mels=128)

    log_mel_spectogram = librosa.power_to_db(mel_spectogram)

    log_mel_spectogram = np.expand_dims(log_mel_spectogram, axis=-1)

    return {'audio': log_mel_spectogram}

train_dataset = dataset['train'].map(preprocess_audio)
validation_dataset = dataset['validation'].map(preprocess_audio)
test_dataset = dataset['test'].map(preprocess_audio)


# Access the first sample from the train dataset and inspect
sample = train_dataset[0]

# Convert 'audio' list into a NumPy array and check its shape
audio_data = np.array(sample['audio'])
print("Sample audio shape:", audio_data.shape)  # Should be (time_steps, n_mels, 1) = (128, 32, 1)
print("First 10 values of the audio data:", audio_data[:10]) # is a 10x10 grid, a list of lists




