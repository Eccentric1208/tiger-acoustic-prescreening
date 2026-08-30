"""
Training script for the binary alarm-call CNN.

Loads the 5-class dataset from Phase 1, remaps labels to binary
(ambient=0, everything-else=1), trains the CNN, saves the model and a
training-curve plot, and prints test-set metrics.

Run from project root:
    python -m src.model.train_binary
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — needed for saving without a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.preprocessing.dataset import (
    AlarmCallDataset,
    CLASS_NAMES,
    NUM_CLASSES,
)
from src.model.binary_cnn import build_binary_cnn


# ------------ config ------------

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# All outputs land under results/ per project convention.
MODEL_PATH = Path("results/models/binary_cnn.keras")
CURVES_PATH = Path("results/plots/binary_training_curves.png")

# Which of the 5 dataset classes count as "alarm" (label=1) for the binary task.
# ambient (index 4) becomes label 0; the four vocalization classes become label 1.
AMBIENT_LABEL = CLASS_NAMES.index("ambient")


# ------------ label remapping ------------

def to_binary_label(spec, label):
    """
    tf.data map function: rewrite the 5-class integer label into
    binary (ambient=0, alarm=1).

    tf.where + tf.equal keep everything inside the TensorFlow graph
    (fast, parallelisable) — no py_function needed here.
    """
    binary = tf.where(
        tf.equal(label, tf.constant(AMBIENT_LABEL, dtype=label.dtype)),
        tf.constant(0, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )
    # Cast to float32 and add a trailing dim so labels are (batch, 1) matching
    # the sigmoid output (batch, 1). Without expand_dims, labels are (batch,)
    # which Keras broadcasts silently — usually correct, occasionally not.
    # Explicit shape = fewer surprises.
    binary = tf.cast(binary, tf.float32)
    binary = tf.expand_dims(binary, axis=-1)
    return spec, binary


def remap_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply the binary relabel to every element of a tf.data.Dataset."""
    return ds.map(to_binary_label, num_parallel_calls=tf.data.AUTOTUNE)


# ------------ plotting ------------

def plot_training_curves(history, save_path: Path) -> None:
    """
    Two-panel figure: loss on the left, accuracy on the right.
    Both curves for train (solid) and validation (dashed) so overfitting
    is visually obvious — val curve peels off from train = overfitting.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(history.history["loss"]) + 1)

    ax_loss.plot(epochs_range, history.history["loss"], label="train")
    ax_loss.plot(epochs_range, history.history["val_loss"], linestyle="--", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("binary cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs_range, history.history["accuracy"], label="train")
    ax_acc.plot(epochs_range, history.history["val_accuracy"], linestyle="--", label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("Binary alarm CNN — training curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------ per-class evaluation ------------

def evaluate_per_class(model, test_ds_5class: tf.data.Dataset) -> None:
    """
    Confusion-style per-class breakdown on the TEST set, using the ORIGINAL
    5-class labels (before binary remap). Tells us not just "did the model
    work", but "which specific species does the model recognize best".
    """
    # Collect all predictions and both label formats in memory (small test set).
    all_probs = []
    all_binary_true = []
    all_5class_true = []

    for specs, labels_5 in test_ds_5class:
        probs = model.predict(specs, verbose=0).flatten()
        binary_true = (labels_5.numpy() != AMBIENT_LABEL).astype(np.int32)

        all_probs.append(probs)
        all_binary_true.append(binary_true)
        all_5class_true.append(labels_5.numpy())

    all_probs = np.concatenate(all_probs)
    all_binary_true = np.concatenate(all_binary_true)
    all_5class_true = np.concatenate(all_5class_true)

    preds = (all_probs >= 0.5).astype(np.int32)

    print("\nPer-class recall (on test set):")
    print(f"  {'class':<10} {'n':>5} {'correct':>8} {'recall':>8}")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = all_5class_true == class_idx
        n = int(mask.sum())
        if n == 0:
            continue
        # For alarm classes we want pred==1; for ambient we want pred==0.
        expected = 0 if class_idx == AMBIENT_LABEL else 1
        correct = int(((preds == expected) & mask).sum())
        recall = correct / n if n else 0.0
        print(f"  {class_name:<10} {n:>5} {correct:>8} {recall:>8.2%}")

    # Overall binary numbers
    tp = int(((preds == 1) & (all_binary_true == 1)).sum())
    fp = int(((preds == 1) & (all_binary_true == 0)).sum())
    fn = int(((preds == 0) & (all_binary_true == 1)).sum())
    tn = int(((preds == 0) & (all_binary_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_all = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / max(len(preds), 1)

    print("\nOverall (test set):")
    print(f"  accuracy:  {accuracy:.2%}")
    print(f"  precision: {precision:.2%}  (of predicted alarms, how many were real)")
    print(f"  recall:    {recall_all:.2%}  (of real alarms, how many did we catch)")
    print(f"  confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")


# ------------ main ------------

def main() -> None:
    # Reproducibility. Doesn't guarantee bitwise-identical runs (TF/threading
    # randomness sneaks in) but keeps results in the same ballpark.
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading dataset...")
    ds = AlarmCallDataset(data_dir="data/raw", random_seed=RANDOM_SEED)
    stats = ds.summary()
    print(f"  total files: {stats['total_files']}   "
          f"train/val/test: {stats['train_size']}/{stats['val_size']}/{stats['test_size']}")

    train_ds_5, val_ds_5, test_ds_5 = ds.build_tf_dataset(batch_size=BATCH_SIZE)

    # Binary-relabelled versions for training and validation
    train_ds = remap_dataset(train_ds_5)
    val_ds = remap_dataset(val_ds_5)
    # Keep the 5-class test set separate for per-species evaluation

    print("\nBuilding model...")
    model = build_binary_cnn(learning_rate=LEARNING_RATE)
    model.summary()
    print(f"  parameters: {model.count_params():,}")

    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,  # one line per epoch (cleaner than the default progress bar)
    )

    print("\nSaving training curves...")
    plot_training_curves(history, CURVES_PATH)
    print(f"  -> {CURVES_PATH}")

    print("\nSaving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"  -> {MODEL_PATH}")

    print("\nEvaluating on held-out test set...")
    evaluate_per_class(model, test_ds_5)

    print("\nDone.")


if __name__ == "__main__":
    main()