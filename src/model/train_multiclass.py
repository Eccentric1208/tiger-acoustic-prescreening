"""
Training script for the multi-class species CNN.

No label remap here — we keep the 5 original classes (sambar, langur, chital,
tiger, ambient) and train the model to distinguish all of them.

Outputs:
  results/models/multiclass_cnn.keras
  results/plots/multiclass_training_curves.png
  results/plots/multiclass_confusion_matrix.png

Run from project root:
    python -m src.model.train_multiclass
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.preprocessing.dataset import (
    AlarmCallDataset,
    CLASS_NAMES,
    NUM_CLASSES,
)
from src.model.multiclass_cnn import build_multiclass_cnn


# ------------ config ------------

BATCH_SIZE = 32
EPOCHS = 25 # slightly more than binary — 5-way is harder
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

MODEL_PATH = Path("results/models/multiclass_cnn.keras")
CURVES_PATH = Path("results/plots/multiclass_training_curves.png")
CONFUSION_PATH = Path("results/plots/multiclass_confusion_matrix.png")


# ------------ training curves plot ------------

def plot_training_curves(history, save_path: Path) -> None:
    """Loss + accuracy curves. Same shape as the binary version."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history.history["loss"]) + 1)

    ax_loss.plot(epochs_range, history.history["loss"], label="train")
    ax_loss.plot(epochs_range, history.history["val_loss"], linestyle="--", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("sparse categorical cross-entropy")
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

    fig.suptitle("Multi-class species CNN — training curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------ confusion matrix ------------

def compute_and_plot_confusion(
    model, test_ds: tf.data.Dataset, save_path: Path
) -> np.ndarray:
    """
    Build a confusion matrix from test-set predictions and save as a heatmap.

    Confusion matrix:
      - Rows = true class (what the sample actually was)
      - Cols = predicted class (what the model said)
      - Diagonal entries = correct predictions
      - Off-diagonal = mistakes; the pattern shows WHICH classes get confused

    E.g., if chital and langur have overlapping frequencies, we'd see a
    lit-up cell at (row=chital, col=langur) or (row=langur, col=chital).
    """
    all_true = []
    all_pred = []

    for specs, labels in test_ds:
        probs = model.predict(specs, verbose=0)  # (batch, num_classes)
        preds = np.argmax(probs, axis=1)         # (batch,) — index of max prob
        all_pred.append(preds)
        all_true.append(labels.numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Build the matrix by hand — no sklearn needed.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    # Print per-class recall and precision from the matrix.
    print("\nPer-class breakdown:")
    print(f"  {'class':<10} {'n_true':>7} {'correct':>8} {'recall':>8} {'precision':>10}")
    for i, name in enumerate(CLASS_NAMES):
        n_true = int(cm[i, :].sum())          # total actual samples of class i
        n_pred = int(cm[:, i].sum())          # total predicted as class i
        correct = int(cm[i, i])
        recall = correct / n_true if n_true else 0.0
        precision = correct / n_pred if n_pred else 0.0
        print(f"  {name:<10} {n_true:>7} {correct:>8} "
              f"{recall:>8.2%} {precision:>10.2%}")

    overall_acc = np.trace(cm) / cm.sum()
    print(f"\nOverall accuracy: {overall_acc:.2%}")

    # --- plot ---
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")

    # Annotate every cell with its count. White text on dark cells so it stays
    # readable against the colormap.
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return cm


# ------------ main ------------

def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading dataset...")
    ds = AlarmCallDataset(data_dir="data/raw", random_seed=RANDOM_SEED)
    stats = ds.summary()
    print(f"  total files: {stats['total_files']}   "
          f"train/val/test: {stats['train_size']}/{stats['val_size']}/{stats['test_size']}")

    # No relabeling — dataset already gives us int labels 0-4 which is exactly
    # what sparse_categorical_crossentropy expects.
    train_ds, val_ds, test_ds = ds.build_tf_dataset(batch_size=BATCH_SIZE)

    print("\nBuilding model...")
    model = build_multiclass_cnn(learning_rate=LEARNING_RATE)
    model.summary()
    print(f"  parameters: {model.count_params():,}")

    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
    )

    print("\nSaving training curves...")
    plot_training_curves(history, CURVES_PATH)
    print(f"  -> {CURVES_PATH}")

    print("\nSaving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"  -> {MODEL_PATH}")

    print("\nEvaluating on held-out test set...")
    compute_and_plot_confusion(model, test_ds, CONFUSION_PATH)
    print(f"  confusion matrix -> {CONFUSION_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()