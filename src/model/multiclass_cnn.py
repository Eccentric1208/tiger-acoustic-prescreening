"""
Multi-class CNN: species-level classifier.

Extends the binary model to distinguish sambar vs langur vs chital vs tiger
vs ambient. Same convolutional backbone as binary_cnn.py — only the head
changes: softmax over 5 classes instead of a single sigmoid.

Why the same backbone? Two reasons:
  1. If binary_cnn works and this doesn't, we know the problem is
     fine-grained class confusion, not the feature extractor.
  2. Reusing the architecture makes it possible to warm-start this model
     from binary weights later (transfer learning) — a nice-to-have we can
     add in the README as "future work".
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.preprocessing.dataset import NUM_CLASSES


def build_multiclass_cnn(
    input_shape=(128, None, 1),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile a CNN for 5-way species classification.

    Differences from build_binary_cnn:
      - Final layer: Dense(num_classes, softmax) instead of Dense(1, sigmoid)
      - Loss: sparse_categorical_crossentropy instead of binary_crossentropy
        ('sparse' means labels are integers 0-4, not one-hot vectors)
      - Metrics: only accuracy (precision/recall are per-class in multi-class
        settings — we compute those manually in the training script)
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram")

    # ---- Block 1 ----
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(momentum=0.9, name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ---- Block 2 ----
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(momentum=0.9, name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ---- Block 3 ----
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(momentum=0.9, name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ---- Aggregation ----
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)

    # ---- Classification head ----
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Softmax over num_classes = probability distribution across all species.
    # The 5 output values sum to 1.0; pick the index with the highest value
    # as the predicted class.
    outputs = layers.Dense(
        num_classes, activation="softmax", name="species_probabilities"
    )(x)

    model = Model(inputs, outputs, name="multiclass_species_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # 'sparse' variant expects int labels (0-4), not one-hot vectors.
        # Saves memory and matches what our dataset already produces.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_multiclass_cnn()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")