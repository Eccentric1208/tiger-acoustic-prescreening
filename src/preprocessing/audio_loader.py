"""
Audio Loader Module
Loads WAV files from AudioMoth recordings or xeno-canto samples.
Handles resampling, normalization, and windowing.

AudioMoth records at 48kHz. Xeno-canto samples vary.
We standardize everything to 22050 Hz mono for ML processing.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


# Standard sample rate for all ML processing
# 22050 Hz captures all alarm call frequencies (up to 11kHz Nyquist)
# while keeping spectrogram computation fast
TARGET_SR = 22050

# Standard clip duration in seconds
# Long enough to capture one full alarm call burst
CLIP_DURATION = 3.0


def load_audio(
    filepath: str,
    target_sr: int = TARGET_SR,
    duration: Optional[float] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target rate, and normalize.

    Args:
        filepath: Path to WAV/MP3/FLAC file
        target_sr: Target sample rate (default 22050 Hz)
        duration: If set, truncate or pad to this duration in seconds
        normalize: If True, normalize amplitude to [-1, 1]

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    filepath = str(filepath)

    # librosa.load handles resampling automatically
    # mono=True converts stereo to mono (AudioMoth records mono anyway)
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    if normalize:
        audio = normalize_audio(audio)

    if duration is not None:
        audio = fix_length(audio, duration, sr)

    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak normalize audio to [-1, 1] range.
    Prevents clipping and ensures consistent input levels for the model.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def fix_length(audio: np.ndarray, duration: float, sr: int) -> np.ndarray:
    """
    Pad with zeros or truncate to exact duration.
    CNN expects fixed-size input, so every clip must be the same length.

    Args:
        audio: Audio array
        duration: Target duration in seconds
        sr: Sample rate

    Returns:
        Audio array of exact target length
    """
    target_length = int(duration * sr)

    if len(audio) < target_length:
        # Pad with zeros at the end
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    elif len(audio) > target_length:
        # Truncate
        audio = audio[:target_length]

    return audio


def load_directory(
    directory: str,
    target_sr: int = TARGET_SR,
    duration: float = CLIP_DURATION
) -> list:
    """
    Load all audio files from a directory.

    Args:
        directory: Path to directory containing audio files
        target_sr: Target sample rate
        duration: Fixed duration for each clip

    Returns:
        List of (audio_array, filename) tuples
    """
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    directory = Path(directory)

    results = []
    for filepath in sorted(directory.iterdir()):
        if filepath.suffix.lower() in supported_formats:
            try:
                audio, sr = load_audio(
                    filepath, target_sr=target_sr, duration=duration
                )
                results.append((audio, filepath.name))
            except Exception as e:
                print(f"  Warning: Could not load {filepath.name}: {e}")

    return results


def window_audio(
    audio: np.ndarray,
    sr: int,
    window_seconds: float = CLIP_DURATION,
    hop_seconds: float = 1.5
) -> list:
    """
    Slide a window across a long recording and extract clips.
    This is how we process full AudioMoth recordings —
    chop them into overlapping windows and classify each one.

    Args:
        audio: Full audio array (could be hours long)
        sr: Sample rate
        window_seconds: Window size in seconds
        hop_seconds: How far to slide between windows

    Returns:
        List of (audio_clip, start_time_seconds) tuples
    """
    window_size = int(window_seconds * sr)
    hop_size = int(hop_seconds * sr)

    clips = []
    start = 0
    while start + window_size <= len(audio):
        clip = audio[start:start + window_size]
        start_time = start / sr
        clips.append((clip, start_time))
        start += hop_size

    return clips


# ---- Quick test ----
if __name__ == "__main__":
    import sys

    test_file = "data/raw/sambar/sambar_sample_001.wav"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    print(f"Loading: {test_file}")
    audio, sr = load_audio(test_file, duration=CLIP_DURATION)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"  RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")
    print("  Audio loader working.")
