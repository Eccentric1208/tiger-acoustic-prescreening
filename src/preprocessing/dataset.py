"""
Dataset loader for the tiger acoustic pre-screening system.

Walks the data/raw/{species}/ folder tree, loads WAV files via the existing
preprocessing modules, computes mel spectrograms, and hands batches to
TensorFlow as a tf.data.Dataset.

Design choices worth knowing:

  1. LAZY loading. We store file paths + labels at startup, but WAVs are only
     read from disk when TensorFlow asks for the next batch. This means the
     code works identically whether the dataset has 1,000 files or 1,000,000.

  2. NOISE FILTER is applied, but with the energy gate DISABLED. The gate is
     an inference-time optimization designed to reject silence; during training
     we want every labeled example to reach the model, including quiet ones.

  3. NO AUGMENTATION in this file. That's a deliberate second pass, added only
     after the plain pipeline is proven to train a working model.

  4. STRATIFIED splits. Each class contributes proportionally to train/val/test.
     Without this, a random split could put all 40 tiger samples in the test
     set and leave the training set with none — the model would never learn
     what a tiger sounds like.

Run this file directly (python -m src.preprocessing.dataset) to see a
diagnostic summary: class counts, split sizes, one batch shape.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import tensorflow as tf

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.spectrogram import audio_to_melspectrogram
from src.preprocessing.noise_filter import full_pipeline


# -------- constants that other files (the CNN, inference) will import --------

# Order matters here: the index of each name IS its integer label.
# sambar=0, langur=1, chital=2, tiger=3, ambient=4. Keep this stable — if you
# reorder, previously-trained models silently become wrong.
CLASS_NAMES = ['sambar', 'langur', 'chital', 'tiger', 'ambient']
NUM_CLASSES = len(CLASS_NAMES)

# Fixed audio length in seconds. Any clip shorter is zero-padded; any clip
# longer is truncated. This has to be consistent across all training samples
# because the CNN wants a fixed input shape.
CLIP_DURATION = 3.0
SAMPLE_RATE = 22050


# ------------------------------- the loader class ----------------------------

class AlarmCallDataset:
    """
    Represents the on-disk dataset. Does not load any audio until you call
    build_tf_dataset().
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        clip_duration: float = CLIP_DURATION,
        sample_rate: int = SAMPLE_RATE,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        # test_frac is implicit: 1 - train_frac - val_frac
        random_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.random_seed = random_seed

        # Walk the folder tree once, at construction time. Cheap — just listing
        # filenames, not reading audio.
        self.filepaths, self.labels = self._scan_directory()

        # Compute stratified train/val/test splits. Each index in
        # self.train_idx / val_idx / test_idx points into self.filepaths.
        self.train_idx, self.val_idx, self.test_idx = self._stratified_split()

    # ---- private: directory scan ----

    def _scan_directory(self) -> Tuple[List[str], List[int]]:
        """
        Walk data/raw/, find every WAV, attach the correct integer label
        based on which class-folder it sits in.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Have you run 'python src/utils/generate_samples.py' yet?"
            )

        filepaths: List[str] = []
        labels: List[int] = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                # Not fatal — the user might be experimenting with a subset —
                # but noisy so they notice.
                print(f"  WARNING: no folder for class '{class_name}' at {class_dir}")
                continue

            # sorted() so the file order is deterministic across runs, which
            # combined with a fixed random_seed makes splits reproducible.
            wavs = sorted(class_dir.glob("*.wav"))
            for wav in wavs:
                filepaths.append(str(wav))
                labels.append(class_idx)

        if not filepaths:
            raise RuntimeError(f"No WAV files found under {self.data_dir}")

        return filepaths, labels

    # ---- private: stratified split ----

    def _stratified_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split file indices into train/val/test so that each class is
        represented proportionally in each split.

        Naive approach (DON'T do this):
            np.random.shuffle(all_indices)
            train = all_indices[:70%], val = ..., test = ...
        Problem: with random shuffling, an unlucky seed can put all 40 tiger
        samples in the test set. The model then never sees a tiger during
        training and can't learn to detect one.

        Stratified approach (what we do):
            For each class separately, shuffle its samples, then split THAT
            class 70/15/15. Concatenate across classes. Result: every split
            has ~70% / ~15% / ~15% of each class.
        """
        rng = np.random.default_rng(self.random_seed)
        labels_arr = np.array(self.labels)

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for class_idx in range(NUM_CLASSES):
            # Indices in self.filepaths where the label matches this class
            class_positions = np.where(labels_arr == class_idx)[0]
            rng.shuffle(class_positions)

            n = len(class_positions)
            n_train = int(n * self.train_frac)
            n_val = int(n * self.val_frac)
            # test gets the rest — avoids off-by-one from rounding

            train_idx.extend(class_positions[:n_train].tolist())
            val_idx.extend(class_positions[n_train:n_train + n_val].tolist())
            test_idx.extend(class_positions[n_train + n_val:].tolist())

        # Shuffle across-class within each split so a batch isn't 32 sambars
        # in a row followed by 32 langurs.
        train_idx_arr = np.array(train_idx); rng.shuffle(train_idx_arr)
        val_idx_arr = np.array(val_idx); rng.shuffle(val_idx_arr)
        test_idx_arr = np.array(test_idx); rng.shuffle(test_idx_arr)

        return train_idx_arr, val_idx_arr, test_idx_arr

    # ---- private: the actual work of turning one file into one spectrogram --

    def _load_and_transform(self, filepath: str) -> np.ndarray:
        """
        The full preprocessing chain for ONE audio file.

        Called lazily by tf.data — this function only runs when TensorFlow
        actually wants the next sample in a batch.

        Returns a spectrogram of shape (n_mels, time_frames, 1). The trailing
        1 is a channel dimension: Conv2D expects (height, width, channels).
        """
        # 1. Load and resample to a fixed duration.
        #    fix_length inside load_audio pads short clips and truncates long
        #    ones, so every waveform coming out is exactly clip_duration
        #    seconds long -> exactly the same array length every time.
        audio, sr = load_audio(
            filepath,
            target_sr=self.sample_rate,
            duration=self.clip_duration,
            normalize=True,
        )

        # 2. Noise filtering, but with the energy gate DISABLED.
        #    See the module docstring for why. full_pipeline returns
        #    (audio_or_None, metadata); with apply_gate=False the audio is
        #    never None, so we can unpack safely.
        cleaned, _meta = full_pipeline(
            audio,
            sr=sr,
            apply_bandpass=True,
            apply_spectral_sub=True,
            apply_gate=False,
        )

        # 3. Mel spectrogram in decibels. Shape: (n_mels, time_frames).
        mel = audio_to_melspectrogram(cleaned, sr=sr)

        # 4. Add channel dim -> (n_mels, time_frames, 1) so Conv2D is happy.
        mel = mel[..., np.newaxis].astype(np.float32)

        return mel

    # ---- public: build the tf.data.Dataset objects ----

    def build_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle_buffer: int = 256,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Turn our filepaths + labels into three tf.data.Dataset objects.

        Why tf.data.Dataset and not just a numpy array?
          - Lazy: only loads what the current batch needs
          - Handles shuffling, batching, prefetching for us
          - Runs the file I/O in parallel with the GPU/CPU training step,
            so the model is never waiting on disk

        Returns:
            (train_ds, val_ds, test_ds)
        """
        # Convert our python lists to arrays we can index cheaply.
        filepaths_arr = np.array(self.filepaths)
        labels_arr = np.array(self.labels)

        def make_split(indices: np.ndarray, shuffle: bool) -> tf.data.Dataset:
            """Build one tf.data pipeline for a given set of file indices."""
            paths = filepaths_arr[indices]
            lbls = labels_arr[indices]

            # from_tensor_slices: yields (path, label) pairs one at a time
            ds = tf.data.Dataset.from_tensor_slices((paths, lbls))

            # Shuffle BEFORE loading. Reshuffles every epoch. Only for train.
            if shuffle:
                ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(indices)),
                                seed=self.random_seed,
                                reshuffle_each_iteration=True)

            # tf.data can't call our python function directly — TensorFlow's
            # graph mode wants graph ops. tf.py_function is the escape hatch
            # that lets us run arbitrary python code inside a tf.data pipeline.
            # Slower than native tf ops but fine for I/O-bound preprocessing.
            def _load_py(path_tensor, label_tensor):
                path = path_tensor.numpy().decode("utf-8")
                spec = self._load_and_transform(path)
                return spec, label_tensor

            def _load_wrapper(path, label):
                spec, lbl = tf.py_function(
                    func=_load_py,
                    inp=[path, label],
                    Tout=(tf.float32, tf.int64),
                )
                # py_function loses shape info; declare it back so downstream
                # layers know what to expect. Time-axis is None because the
                # exact frame count depends on hop_length and clip_duration —
                # we'll pin it once we've inspected an actual sample below.
                spec.set_shape([None, None, 1])
                lbl.set_shape([])
                return spec, lbl

            # num_parallel_calls=AUTOTUNE = let TF pick how many files to load
            # in parallel based on available CPU cores.
            ds = ds.map(_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size)
            # prefetch = while GPU trains on batch N, load batch N+1 from disk
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_split(self.train_idx, shuffle=True)
        val_ds = make_split(self.val_idx, shuffle=False)
        test_ds = make_split(self.test_idx, shuffle=False)
        return train_ds, val_ds, test_ds

    # ---- public: diagnostic ----

    def summary(self) -> Dict:
        """Return a dict of dataset statistics for quick sanity checks."""
        labels_arr = np.array(self.labels)
        per_class = {
            CLASS_NAMES[i]: int((labels_arr == i).sum())
            for i in range(NUM_CLASSES)
        }
        return {
            "total_files": len(self.filepaths),
            "per_class": per_class,
            "train_size": len(self.train_idx),
            "val_size": len(self.val_idx),
            "test_size": len(self.test_idx),
        }


# ------------------- smoke test: run this file directly ----------------------

if __name__ == "__main__":
    print("Building dataset...\n")
    ds = AlarmCallDataset(data_dir="data/raw")

    stats = ds.summary()
    print(f"Total files:     {stats['total_files']}")
    print(f"Per class:       {stats['per_class']}")
    print(f"Train / Val / Test: "
          f"{stats['train_size']} / {stats['val_size']} / {stats['test_size']}")

    print("\nBuilding tf.data pipelines...")
    train_ds, val_ds, test_ds = ds.build_tf_dataset(batch_size=32)

    print("Grabbing one training batch to verify shapes...")
    for specs, labels in train_ds.take(1):
        print(f"  spectrogram batch shape: {specs.shape}   dtype: {specs.dtype}")
        print(f"  label batch shape:       {labels.shape}  dtype: {labels.dtype}")
        print(f"  label values in batch:   {sorted(set(labels.numpy().tolist()))}")
        print(f"  spec value range:        "
              f"[{float(tf.reduce_min(specs)):.2f}, "
              f"{float(tf.reduce_max(specs)):.2f}]  dB")

    print("\nDataset pipeline OK. Ready to build the CNN.")