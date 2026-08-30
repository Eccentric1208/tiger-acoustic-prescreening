"""
Binary CNN: alarm-call vs ambient-noise classifier.

This is the simpler of our two classifiers. It answers ONE question:
"is there any prey alarm call (sambar/langur/chital/tiger) in this 3-second
window, or is it just forest background?"

Deliberately kept small (~100K parameters) so:
  - it trains in a few minutes on a CPU
  - it doesn't overfit our synthetic 1000-sample dataset
  - the architecture is legible when someone reads the README

If this baseline works, the same shape scales up cleanly to the 5-class
multi-class version in Phase 3 — we just swap the final layer.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_binary_cnn(
    input_shape=(128, None, 1),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile a small CNN for binary alarm-call classification.

    Args:
        input_shape: (n_mels, time_frames, channels). time_frames=None lets
            the model accept any clip length at inference (handy for sliding-
            window inference over long recordings later).
        dropout_rate: fraction of activations to zero during training.
            Regularization — prevents the model from memorizing training
            examples verbatim.
        learning_rate: how big a step the optimizer takes each batch. 1e-3
            is the default choice for Adam and rarely needs tuning early on.

    Returns:
        A compiled tf.keras.Model with binary crossentropy loss.
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram")

    # ---- Block 1: catch broad frequency-band patterns ----
    # 32 filters is deliberately modest at this stage; deeper layers
    # get progressively wider as spatial dims shrink.
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ---- Block 2 ----
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ---- Block 3 ----
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ---- Aggregation ----
    # GlobalAveragePooling2D collapses (H, W, C) -> (C,) by averaging over
    # spatial dims. This is what lets the model handle variable-length clips
    # at inference — a 3-second clip and a 5-second clip produce the same
    # vector shape after this layer.
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)

    # ---- Classification head ----
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Single sigmoid = probability of "alarm call present". Threshold at 0.5
    # (default) or tune later depending on false-alarm tolerance.
    outputs = layers.Dense(1, activation="sigmoid", name="alarm_probability")(x)

    model = Model(inputs, outputs, name="binary_alarm_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            # Precision/recall matter more than accuracy here: false negatives
            # (missing a real alarm) cost the ranger a camera-trap placement
            # opportunity; false positives (crying wolf) burn analyst time.
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check when the file is run directly.
    model = build_binary_cnn()
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")