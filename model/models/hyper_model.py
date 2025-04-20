from tensorflow.keras import layers, models
import tensorflow as tf

def build_hypermodel(hp, best_name, input_shape, num_labels, norm_layer):
    # Definer arkitektur med variable hyperparametre
    if best_name == 'CNN':
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(filters=hp.Choice('conv1_filters', [16, 32, 64]),
                         kernel_size=hp.Choice('conv1_kernel', [3, 5]), activation='relu'),
            layers.Conv2D(filters=hp.Choice('conv2_filters', [32, 64, 128]),
                         kernel_size=hp.Choice('conv2_kernel', [3, 5]), activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(hp.Float('dropout_1', 0.2, 0.6, step=0.1)),
            layers.Flatten(),
            layers.Dense(units=hp.Choice('dense_units', [64, 128, 256]), activation='relu'),
            layers.Dropout(hp.Float('dropout_2', 0.2, 0.6, step=0.1)),
            layers.Dense(num_labels, activation='softmax')
        ])
    else:  # CRNN
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(filters=hp.Choice('conv1_filters', [16, 32, 64]),
                         kernel_size=hp.Choice('conv1_kernel', [3, 5]), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=hp.Choice('conv2_filters', [32, 64, 128]),
                         kernel_size=hp.Choice('conv2_kernel', [3, 5]), activation='relu'),
            layers.MaxPooling2D(),
            layers.Reshape(target_shape=(-1, hp.Choice('reshape_channels', [64, 128]))),
            layers.LSTM(units=hp.Choice('lstm_units_1', [32, 64, 128]), return_sequences=True),
            layers.LSTM(units=hp.Choice('lstm_units_2', [32, 64, 128])),
            layers.Dense(units=hp.Choice('dense_units', [64, 128, 256]), activation='relu'),
            layers.Dropout(hp.Float('dropout', 0.2, 0.6, step=0.1)),
            layers.Dense(num_labels, activation='softmax')
        ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model