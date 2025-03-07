import numpy as np
import tensorflow as tf
import keras
from datasets import Dataset, load_dataset
import librosa
from keras import layers, models

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Numpy version:", np.__version__)

dataset = load_dataset("google/speech_commands", "v0.01")

# Overview of the dataset
print(dataset)

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# Check available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)


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
# audio_data = np.array(sample['audio'])
# print("Sample audio shape:", audio_data.shape)  # Should be (time_steps, n_mels, 1) = (128, 32, 1)
# print("First 10 values of the audio data:", audio_data[:10]) # is a 10x10 grid, a list of lists


# A simple CNN model
model = models.Sequential([
    layers.InputLayer(shape=(128, 32, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(12, activation='softmax') # Number of possible commands
])

# Compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


def audio_generator(dataset):
    for sample in dataset:
        audio_features = sample['audio']
        label = sample['label']
        
        # Convert audio_features to a numpy array (if it's not already)
        audio_features = np.array(audio_features)
        
        # Ensure the audio features have the shape (128, 32, 1)
        # Pad or truncate if necessary (this assumes the audio data is 2D, with shape (128, n_features, 1))
        if audio_features.shape[1] < 32:
            # Pad the sequence if it's shorter than expected
            pad_width = 32 - audio_features.shape[1]
            audio_features = np.pad(audio_features, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        elif audio_features.shape[1] > 32:
            # Truncate the sequence if it's longer than expected
            audio_features = audio_features[:, :32, :]
        
        # Yield the audio features and label
        yield audio_features, label

def convert_to_tf_dataset(dataset):
    # Create a TensorFlow Dataset from the generator
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(dataset), 
        output_signature=(
            tf.TensorSpec(shape=(128, 32, 1), dtype=tf.float32),  # The expected shape of the audio data
            tf.TensorSpec(shape=(), dtype=tf.int64)  # Adjust dtype according to your label type (e.g., tf.int64 for class labels)
        )
    )
    return tf_dataset


# Convert the train, validation, and test datasets
train_tf_dataset = convert_to_tf_dataset(train_dataset)
validation_tf_dataset = convert_to_tf_dataset(validation_dataset)
test_tf_dataset = convert_to_tf_dataset(test_dataset)

print("Fitting the model now!!!")

model.fit(
    train_tf_dataset.batch(128),
    epochs=10,  # Can be changed
    validation_data=validation_tf_dataset.batch(128)
)

test_loss, test_accuracy = model.evaluate(test_tf_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

