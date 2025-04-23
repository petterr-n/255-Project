from tensorflow.keras import layers, models

def build_baseline_model(input_shape, num_labels):
    model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Rescaling(1./255),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_labels, activation='softmax')
])
    return model