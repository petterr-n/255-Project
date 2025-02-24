import numpy as np
import tensorflow as tf
import keras
from datasets import Dataset, load_dataset

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Numpy version:", np.__version__)

dataset = load_dataset("google/speech_commands", "v0.01")

# Overview of the dataset
print(dataset)

print(dataset['train'][0])

print(dataset['train'].features['label'])
print(dataset['train'][0]['audio'])
print(dataset['train'][0]['label'])