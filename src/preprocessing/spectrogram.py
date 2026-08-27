"""
Spectrogram Module
Converts audio to mel spectrograms for CNN input.

Key design decisions:
- 128 mel bands: standard for audio classification CNNs
- fmin=200 Hz: captures tiger roar fundamentals (80Hz overtones show above 200)
- fmax=8000 Hz: captures all alarm call energy, ignores ultrasonic noise
- These params are tuned for Indian forest alarm calls specifically.
  Kershenbaum et al. (2026) used similar ranges for chital detection in Nepal.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Spectrogram parameters tuned for alarm call classification
N_MELS = 128          # Number of mel bands (height of spectrogram image)
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # Hop between FFT windows
FMIN = 200            # Minimum frequency (Hz) - captures tiger roar harmonics
FMAX = 8000           # Maximum frequency (Hz) - above all alarm call energy


def audio_to_melspectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX
) -> np.ndarray:
    """
    Convert audio waveform to mel spectrogram in decibels.

    This is the core transformation. The CNN will learn patterns
    from these spectrograms — each species' alarm call has a
    distinct visual fingerprint in the mel spectrogram.

    Args:
        audio: Audio waveform array
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram in dB scale, shape (n_mels, time_frames)
    """
    # Generate mel spectrogram (power spectrum)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )

    # Convert power to decibels — this compresses the dynamic range
    # and makes quiet sounds visible alongside loud ones
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def spectrogram_to_image(
    mel_spec_db: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert mel spectrogram to normalized image array for CNN input.

    Args:
        mel_spec_db: Mel spectrogram in dB
        normalize: Scale to [0, 1] range

    Returns:
        2D array ready for CNN input
    """
    if normalize:
        # Scale to [0, 1] — neural networks prefer this range
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()
        if spec_max > spec_min:
            mel_spec_db = (mel_spec_db - spec_min) / (spec_max - spec_min)
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)

    return mel_spec_db


def plot_spectrogram(
    mel_spec_db: np.ndarray,
    sr: int = 22050,
    hop_length: int = HOP_LENGTH,
    title: str = "Mel Spectrogram",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> None:
    """
    Plot a mel spectrogram with proper axis labels.

    Args:
        mel_spec_db: Mel spectrogram in dB
        sr: Sample rate
        hop_length: Hop length used in generation
        title: Plot title
        save_path: If set, save figure to this path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    img = librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=hop_length,
        fmin=FMIN,
        fmax=FMAX,
        ax=ax,
        cmap='magma'
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity (dB)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_comparison(
    spectrograms: dict,
    sr: int = 22050,
    hop_length: int = HOP_LENGTH,
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple spectrograms side by side for visual comparison.
    This is how you verify the model can distinguish between species —
    if YOU can see the difference, the CNN can learn it.

    Args:
        spectrograms: Dict of {label: mel_spec_db}
        sr: Sample rate
        hop_length: Hop length
        save_path: If set, save figure
    """
    n = len(spectrograms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (label, spec) in zip(axes, spectrograms.items()):
        img = librosa.display.specshow(
            spec,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            hop_length=hop_length,
            fmin=FMIN,
            fmax=FMAX,
            ax=ax,
            cmap='magma'
        )
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        if ax == axes[0]:
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_ylabel('')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return fig


# ---- Quick test ----
if __name__ == "__main__":
    from audio_loader import load_audio, CLIP_DURATION

    species = ['sambar', 'langur', 'chital', 'tiger', 'ambient']
    spectrograms = {}

    for sp in species:
        filepath = f"data/raw/{sp}/{sp}_sample_001.wav"
        audio, sr = load_audio(filepath, duration=CLIP_DURATION)
        mel_spec = audio_to_melspectrogram(audio, sr)
        spectrograms[sp.capitalize()] = mel_spec
        print(f"  {sp}: spectrogram shape {mel_spec.shape}")

    plot_comparison(spectrograms, save_path="results/spectrogram_comparison.png")
    print("\nSpectrogram module working.")
