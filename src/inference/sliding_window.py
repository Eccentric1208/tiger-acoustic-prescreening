"""
Sliding-window inference over long audio recordings.

Real AudioMoth recordings are hours long. Our CNN was trained on 3-second
clips. This module bridges the gap: it slides a fixed-length window across
a long recording, classifies each window, and produces a timestamped list
of detections.

Design choices:

  1. OVERLAP. Windows overlap by 50% by default. Without overlap, a call
     that straddles a window boundary can be cut in half and missed by both
     halves. 50% overlap guarantees every 3-second span of audio is fully
     contained in at least one window.

  2. MERGING. Overlapping windows can produce multiple detections for the
     same real event. We post-process by merging adjacent detections of the
     same class into a single event with a start/end time.

  3. WORKS FOR BOTH BINARY AND MULTI-CLASS MODELS. Detected automatically
     from the model's output shape: sigmoid (batch, 1) = binary,
     softmax (batch, num_classes) = multi-class.

Run from project root as a smoke test:
    python -m src.inference.sliding_window
which generates a synthetic 30-second recording, runs the multi-class model
over it, and prints detections.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import csv

import numpy as np
import tensorflow as tf

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.spectrogram import audio_to_melspectrogram
from src.preprocessing.noise_filter import full_pipeline
from src.preprocessing.dataset import CLASS_NAMES, SAMPLE_RATE


# ------------ config defaults ------------

WINDOW_SECONDS = 3.0     # matches CLIP_DURATION used during training
HOP_SECONDS = 1.5        # 50% overlap between consecutive windows
CONFIDENCE_THRESHOLD = 0.5   # min probability to count as a detection
MERGE_GAP_SECONDS = 2.0  # detections of same class within this gap get merged


# ------------ data types ------------

class Detection:
    """One detection event on the timeline."""
    def __init__(self, start: float, end: float, class_name: str,
                 confidence: float):
        self.start = start          # seconds from start of recording
        self.end = end
        self.class_name = class_name
        self.confidence = confidence

    def __repr__(self):
        return (f"Detection({self.start:.1f}-{self.end:.1f}s, "
                f"{self.class_name}, conf={self.confidence:.2f})")


# ------------ window generation ------------

def generate_windows(
    audio: np.ndarray,
    sr: int,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> List[Tuple[float, np.ndarray]]:
    """
    Chop a long audio array into overlapping fixed-length windows.

    Returns a list of (start_time_in_seconds, window_audio) tuples.
    The last window is zero-padded if the recording length isn't a clean
    multiple of the hop — this way we don't drop the tail.
    """
    window_samples = int(window_seconds * sr)
    hop_samples = int(hop_seconds * sr)
    n = len(audio)

    windows: List[Tuple[float, np.ndarray]] = []
    start = 0
    while start < n:
        end = start + window_samples
        if end <= n:
            chunk = audio[start:end]
        else:
            # Zero-pad the tail so the shape is consistent.
            chunk = np.zeros(window_samples, dtype=audio.dtype)
            chunk[:n - start] = audio[start:n]
        start_time = start / sr
        windows.append((start_time, chunk))
        start += hop_samples
        # Stop once we've covered the whole recording, no infinite tail
        if start >= n:
            break

    return windows


# ------------ per-window preprocessing ------------

def preprocess_window(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Turn one raw audio window into the (n_mels, time_frames, 1) spectrogram
    format the CNN expects. Same chain as dataset.py, minus the file I/O.
    Gate stays OFF here — inference-time gate would be a separate optimization.
    """
    cleaned, _ = full_pipeline(
        audio,
        sr=sr,
        apply_bandpass=True,
        apply_spectral_sub=True,
        apply_gate=False,
    )
    mel = audio_to_melspectrogram(cleaned, sr=sr)
    mel = mel[..., np.newaxis].astype(np.float32)
    return mel


# ------------ batched inference ------------

def classify_windows(
    model: tf.keras.Model,
    windows: List[Tuple[float, np.ndarray]],
    sr: int,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run the model over all windows in batches. Returns a probability array:
      - binary model:      shape (n_windows, 1)
      - multi-class model: shape (n_windows, num_classes)
    """
    specs = np.stack([preprocess_window(w, sr) for _, w in windows], axis=0)
    probs = model.predict(specs, batch_size=batch_size, verbose=0)
    return probs


# ------------ probability array -> Detection list ------------

def probs_to_detections(
    windows: List[Tuple[float, np.ndarray]],
    probs: np.ndarray,
    window_seconds: float,
    confidence_threshold: float,
) -> List[Detection]:
    """
    Convert a per-window probability array into a list of Detection events.

    Auto-detects binary vs multi-class from the probability array shape:
      - Last dim == 1 -> binary sigmoid. Threshold on prob; class name is
        just "alarm" (we don't know which species).
      - Last dim > 1  -> softmax. Use argmax to pick the class; threshold on
        the max prob. Windows where the model's top choice is "ambient" are
        dropped (we only care about events, not silence).
    """
    detections: List[Detection] = []
    n_classes = probs.shape[-1]

    for i, (start_time, _) in enumerate(windows):
        end_time = start_time + window_seconds

        if n_classes == 1:
            # Binary
            confidence = float(probs[i, 0])
            if confidence >= confidence_threshold:
                detections.append(Detection(
                    start_time, end_time, "alarm", confidence
                ))
        else:
            # Multi-class
            class_idx = int(np.argmax(probs[i]))
            confidence = float(probs[i, class_idx])
            class_name = CLASS_NAMES[class_idx]
            # Skip "ambient" — we only report events, not silence
            if class_name == "ambient":
                continue
            if confidence >= confidence_threshold:
                detections.append(Detection(
                    start_time, end_time, class_name, confidence
                ))

    return detections


# ------------ merging adjacent detections ------------

def merge_detections(
    detections: List[Detection],
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
) -> List[Detection]:
    """
    Because windows overlap 50%, one real call can trigger 2-3 consecutive
    detections. Merge same-class detections that are close in time into one
    event, keeping the max confidence.
    """
    if not detections:
        return []

    # Sort by start time (should already be, but defensive)
    detections = sorted(detections, key=lambda d: d.start)

    merged: List[Detection] = [detections[0]]
    for d in detections[1:]:
        prev = merged[-1]
        same_class = d.class_name == prev.class_name
        close_in_time = d.start - prev.end <= merge_gap_seconds
        if same_class and close_in_time:
            # Extend the previous detection to cover this one too.
            prev.end = max(prev.end, d.end)
            prev.confidence = max(prev.confidence, d.confidence)
        else:
            merged.append(d)
    return merged


# ------------ top-level function: file -> detections ------------

def detect_in_file(
    model: tf.keras.Model,
    audio_path: str,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
) -> List[Detection]:
    """
    Full pipeline: load a WAV, slide the window, classify, merge.
    This is the function you'll call for every AudioMoth recording.
    """
    audio, sr = load_audio(audio_path, target_sr=SAMPLE_RATE, normalize=True)
    windows = generate_windows(audio, sr, window_seconds, hop_seconds)
    probs = classify_windows(model, windows, sr)
    raw = probs_to_detections(windows, probs, window_seconds, confidence_threshold)
    merged = merge_detections(raw, merge_gap_seconds)
    return merged


# ------------ CSV export ------------

def save_detections_csv(detections: List[Detection], path: str) -> None:
    """Write detections to a CSV usable in Excel, R, QGIS, etc."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "class", "confidence"])
        for d in detections:
            writer.writerow([
                f"{d.start:.2f}",
                f"{d.end:.2f}",
                d.class_name,
                f"{d.confidence:.4f}",
            ])


# ------------ smoke test ------------

def _make_synthetic_long_recording(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Build a 30-second synthetic recording with planted alarm calls at known
    timestamps, so we can visually verify the sliding-window logic works.
    Uses the same generators as generate_samples.py.
    """
    from src.utils.generate_samples import (
        generate_ambient, generate_sambar_alarm, generate_langur_alarm,
    )
    total_seconds = 30.0
    total_samples = int(total_seconds * sr)
    recording = np.zeros(total_samples, dtype=np.float32)

    # Base layer: ambient throughout
    ambient_chunk_seconds = 3.0
    n_chunks = int(total_seconds / ambient_chunk_seconds)
    for i in range(n_chunks):
        start = int(i * ambient_chunk_seconds * sr)
        end = start + int(ambient_chunk_seconds * sr)
        recording[start:end] = generate_ambient(duration=ambient_chunk_seconds, sr=sr)

    # Plant a sambar alarm around t=8s and a langur alarm around t=20s
    for start_seconds, generator in [(8.0, generate_sambar_alarm),
                                     (20.0, generate_langur_alarm)]:
        call = generator(duration=3.0, sr=sr)
        start_idx = int(start_seconds * sr)
        recording[start_idx:start_idx + len(call)] += call

    return np.clip(recording, -1, 1).astype(np.float32)


if __name__ == "__main__":
    print("Sliding-window inference smoke test\n")

    model_path = Path("results/models/multiclass_cnn.keras")
    if not model_path.exists():
        # Fall back to binary if multi-class hasn't been trained yet
        model_path = Path("results/models/binary_cnn.keras")
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found at results/models/. "
            "Train binary or multi-class CNN first."
        )
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Build a synthetic long recording with planted calls at 8s and 20s
    print("Generating 30s synthetic test recording...")
    audio = _make_synthetic_long_recording()

    # Save it so we can also test load_audio -> full pipeline
    import soundfile as sf
    test_wav_path = Path("results/test_recording.wav")
    test_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(test_wav_path), audio, SAMPLE_RATE)
    print(f"  saved -> {test_wav_path}")

    print("Running sliding-window detection...")
    detections = detect_in_file(model, str(test_wav_path))

    print(f"\nFound {len(detections)} detection event(s):")
    for d in detections:
        print(f"  {d}")

    csv_path = Path("results/test_detections.csv")
    save_detections_csv(detections, str(csv_path))
    print(f"\nCSV -> {csv_path}")
    print("Expected: ~1 detection near 8s and ~1 near 20s.")