# %% [markdown]
# # Snake Voice‑Command Classification
# 
# This notebook trains two different models (CNN and CRNN) on a spoken‑command dataset, selects the best performer on the validation split, and evaluates it on an unseen test split.
# 
# This work is inspired by the TensorFlow keyword‑spotting tutorial: [Keyword Spotting](https://www.tensorflow.org/tutorials/audio/simple_audio)

# %% [markdown]
# ## Imports & Dependencies

# %%
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers
from IPython import display

from models.cnn_model import build_cnn_model
from models.crnn_model import build_crnn_model
from models.hyper_model import build_hypermodel
from models.baseline_model import build_baseline_model

# %%
SEED = 42   # ensures consistants for reproducibility and training
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "../data/snake_commands"
BATCH_SIZE = 64
EPOCHS = 30

# %% [markdown]
# ## Helper functions
# This section defines functions to prepare raw audio for model input and to visulaize results:
# 1. **Data Preprocessing**
#     - `squeeze(audio, labels)`: removes extra chennel dimension.
#     - `get_spectrogram(waveform)`: computes STFT and returns a magnitude spectrogram with shape `(time, freq, 1)`.
#     - `make_spec_ds(ds)`: maps `get_spectrogram`over a dataset pipeline with parallel calls.
# 2. **Visualization utilities**
#     - `plot_spectrogram(...)`: displays a log-scaled spectrogram.
#     - `plot_history(...)`: plots training/validation loss and accuracy.
#     - `plot_confusion_matrix(...)`: computes predictions, builds a confusion matrix, and visualizes it with Seaborn.

# %%
def squeeze(audio, labels):
    """Remove the last dimension of the audio tensor."""
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# %%
def augment_waveform(audio, labels):
    """On-the-fly waveform augmentation::
        1) Adds small Gaussian noise
        2) Applies random time-shift ±0.1 s
    """
    # 1) Background noise
    noise = tf.random.normal(tf.shape(audio), mean=0.0, stddev=0.001)
    audio = audio + noise

    #2) Tids-shift
    shift = tf.random.uniform([], -320, 320, dtype=tf.int32)
    audio = tf.roll(audio, shift, axis=0)

    return audio, labels

# %%
def get_spectrogram(waveform):
    """ Compute STFT and return magnitude spectrogram with axis."""
    spectogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectogram = tf.abs(spectogram)[..., tf.newaxis]
    return spectogram

# %%
def plot_spectrogram(spectogram, ax):
    """Plot log-scaled spectrogram."""
    if spectogram.ndim == 3:
        spectogram = np.squeeze(spectogram, axis=-1)
    log_spec = np.log(spectogram.T + np.finfo(float).eps)
    h, w = log_spec.shape
    X = np.linspace(0, np.size(spectogram), num=w, dtype=int)
    Y = range(h)
    ax.pcolormesh(X, Y, log_spec)

# %%
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# %%
def plot_history(history, prefix="Model"):
    """Plot training and validation loss & accuracy"""
    hist = history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(hist['loss'], label='train_loss')
    plt.plot(hist['val_loss'], label='val_loss')
    plt.title(f'{prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.array(hist['accuracy'])*100, label='train_accuracy')
    plt.plot(np.array(hist['val_accuracy'])*100, label='val_accuracy')
    plt.title(f'{prefix} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
def plot_confusion_matrix(model, ds, labels):
    """Compute and plot confusion matrix on dataset"""
    preds = model.predict(ds)
    y_pred = tf.argmax(preds, axis=1)
    y_true = tf.concat([l for _, l in ds.map(lambda s, l: (s, l))], axis=0)

    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,annot=True, fmt='g')
    plt.xlabel('Prediction') 
    plt.ylabel('Label')
    plt.show()

# %% [markdown]
# ## Data Loading & Augmentation

# %%
# List all subfolders in the data directory and filter out unwanted files

commands = np.array(tf.io.gfile.listdir(str(DATA_DIR)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Raw folders (commands):', commands)

# %% [markdown]
# Load and split the audio dataset into an 80/20 split, then shards validation set further to separate test and validation subsets.

# %%
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=SEED,
    output_sequence_length=16000,
    subset='both'
)

label_names = np.array(train_ds.class_names)
print("\nLabel names:", label_names)

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_waveform, tf.data.AUTOTUNE)

val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# %% [markdown]
# ## Example waveform and spectrogram

# %%
# Inspect shapes
for example_audio, example_labels in train_ds.take(1):
    print("Audio shape:", example_audio.shape)
    print("Labels shape:", example_labels.shape)

# %%
# Plot a few raw waveforms
plt.figure(figsize=(16, 10))
rows, cols = 3, 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i+1)
    plt.plot(example_audio[i])
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])
plt.show()

# %%
# Show waveform + spectrogram + play audio for one example
waveform = example_audio[0]
spectrogram = get_spectrogram(waveform)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set(title='Waveform', xlim=[0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label_names[example_labels[0]].title())
plt.show()

display.display(display.Audio(waveform, rate=16000))

# %% [markdown]
# ## Spectrogram Dataset Preparation
# 
# Convert each dataset split into spectrograms, then apply caching, shuffling, and prefetching to optimize the input pipeline.

# %%
train_spec_ds = make_spec_ds(train_ds).cache().shuffle(10000, seed=SEED).prefetch(tf.data.AUTOTUNE)
val_spec_ds   = make_spec_ds(val_ds).cache().prefetch(tf.data.AUTOTUNE)
test_spec_ds  = make_spec_ds(test_ds).cache().prefetch(tf.data.AUTOTUNE)

# %%
# Visual check of a few spectrgrams
for spec_batch, label_batch in train_spec_ds.take(1):
    example_specs = spec_batch.numpy()
    example_labs  = label_batch.numpy()
    break

rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
for i in range(rows*cols):
    r, c = divmod(i, cols)
    plot_spectrogram(example_specs[i], axes[r][c])
    axes[r][c].set_title(label_names[example_labs[i]])
plt.show()

# %% [markdown]
# ## Build and compile baseline-, CNN- and CRNN model

# %%
# Determine input shape and setup normalization
for specs, _ in train_spec_ds.take(1):
    input_shape = specs.shape[1:]
num_labels = len(label_names)
print("Input shape:", input_shape, "\nNumber of classes:", num_labels)

norm_layer = layers.Normalization()
norm_layer.adapt(train_spec_ds.map(lambda spec, lab: spec))

# %%
# Build models
models_dict = {
    'CNN':  build_cnn_model(input_shape, num_labels, norm_layer),
    'CRNN': build_crnn_model(input_shape, num_labels, norm_layer),
    'BASELINE': build_baseline_model(input_shape, num_labels)
}

for name, m in models_dict.items():
    print(f"\n{'='*10} Summary for {name} {'='*10}\n")
    m.summary()

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

# %% [markdown]
# ## Model training
# We train each model for up to 30 epochs, using early stopping (patience=5) to prevent overfitting.

# %%
# Train with early stopping
histories = {}
for name, m in models_dict.items():
    print(f"\n=== Training {name} ===")
    histories[name] = m.fit(
        train_spec_ds,
        validation_data=val_spec_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )

# %% [markdown]
# After training, we pick the model with the highest validation accuracy

# %%
# Best model
best_name = max(histories, key=lambda n: max(histories[n].history['val_accuracy']))
best_history = histories[best_name].history
best_model = models_dict[best_name]

best_val_acc = max(best_history['val_accuracy'])
best_val_loss = best_history['val_loss'][best_history['val_accuracy'].index(best_val_acc)]
best_epoch = best_history['val_accuracy'].index(best_val_acc) + 1
print(f"Best model: {best_name} - epoch {best_epoch} with validation accuracy = {best_val_acc:.3f} and validation loss = {best_val_loss:.3f}")

# Other model
for other_name, model in models_dict.items():
    if other_name == best_name:
        continue

    other_history = histories[other_name].history

    other_val_acc = max(other_history['val_accuracy'])
    other_val_loss = other_history['val_loss'][other_history['val_accuracy'].index(other_val_acc)]
    other_epoch = other_history['val_accuracy'].index(other_val_acc) + 1
    print(f"Other model: {other_name} - epoch {other_epoch} with validation accuracy = {other_val_acc:.3f} and validation loss = {other_val_loss:.3f}")

# %% [markdown]
# ## Hyperparameter tuning
# Configure a KerasTuner `RandomSearch`:
# - Objective: maximum validation accuracy.
# - Up to 20 tials
# - Uses our `build_hypermodel`wrapper to vary layer sizes, learning rate, etc.

# %%
hypermodel = partial(
    build_hypermodel,
    best_name=best_name,
    input_shape=input_shape,
    num_labels=num_labels,
    norm_layer=norm_layer
)

print(f"Starter hyperparameter tuning (random search) for modellen: {best_name}")
tuner = kt.RandomSearch(
    hypermodel,
    seed = SEED,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='kt_tuner',
    project_name=f'random_search_{best_name.lower()}'
)

# %%
# Kjør hypertuning
tuner.search(
    train_spec_ds,
    validation_data=val_spec_ds,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)]
)

# %% [markdown]
# ## Train Tuned Model
# Retrive the best hyperparameter set, display each chosen value, and rebuild the model accordingly.

# %%
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest hyperparameters:")
for name, value in best_hp.values.items():
    print(f"{name}: {value}")
tuned_model = tuner.hypermodel.build(best_hp)

# %%
# Tren final modell
history_tuned = tuned_model.fit(
    train_spec_ds,
    validation_data=val_spec_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
)

# %%
tuned_val_acc = max(history_tuned.history['val_accuracy'])
tuned_val_loss = history_tuned.history['val_loss'][history_tuned.history['val_accuracy'].index(tuned_val_acc)]
tuned_epoch = history_tuned.history['val_accuracy'].index(tuned_val_acc) + 1
print(f"Tuned model: epoch {tuned_epoch} with validation accuracy = {tuned_val_acc:.3f} and validation loss = {tuned_val_loss:.3f}")

# %% [markdown]
# ## Final model selection & evaluation
# Check if the tuned model outpreforms the original model, then select the model that has the higher validation accuracy. Then we evaluate that model on the test set.

# %%
if tuned_val_acc > best_val_acc:
    final_model = tuned_model
    final_history = history_tuned.history
    val_acc = tuned_val_acc
    val_loss = tuned_val_loss
    source = 'Tuned'
else:
    final_model = best_model
    final_history = histories[best_name].history
    val_acc = best_val_acc
    val_loss = best_val_loss
    source = 'Original'
print(f"Final model source: {source} with validation accuracy = {val_acc:.3f} and validation loss = {val_loss:.3f}")

# %%
print("=== Evaluating tuned model on test set ===")
results = final_model.evaluate(test_spec_ds, return_dict=True)
print(f"{source} modell test results:", results)

final_model.save(f"final_voice_model.keras")
print(f"\nSaved the {source} model.")

# %% [markdown]
# ## Visualization of final results

# %% [markdown]
# Visualize training vs. validation loss and accuracy for the chosen model to ensure stable convergence.

# %%
plot_history(final_history, prefix=source)

# %% [markdown]
# Generate and display a confusion matrix for the final model's predictions on the test set to identify which commands are most often misclassified.

# %%
plot_confusion_matrix(final_model, test_spec_ds, label_names)


