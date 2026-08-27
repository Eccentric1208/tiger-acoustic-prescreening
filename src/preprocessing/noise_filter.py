"""
Noise Filter Module
Handles the brutal reality of forest audio: rain, wind, cicadas, streams.

Two-stage approach:
  Stage 1 — Bandpass filter: keep only the frequency range where alarm calls live
  Stage 2 — Spectral subtraction: estimate and remove stationary background noise

This is where your system lives or dies in the field.
Clean xeno-canto recordings don't need this. A raw 72-hour AudioMoth
dump from Kaziranga in monsoon absolutely does.

References:
  - Kershenbaum et al. (2026) faced similar noise challenges in Nepal's Terai
  - scipy.signal for Butterworth filter design
  - librosa for spectral analysis
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from typing import Tuple, Optional


# ============================================================
# FREQUENCY BANDS FOR EACH SPECIES
# These define where each alarm call's energy lives.
# The bandpass filter uses the widest range to catch all species.
# ============================================================

SPECIES_BANDS = {
    'sambar':  (400, 2500),    # Deep resonant bark
    'langur':  (1000, 4500),   # High-pitched rapid calls
    'chital':  (1000, 3500),   # Mid-range whistle-bark
    'tiger':   (50, 500),      # Very low roar fundamentals
}

# Combined band that captures ALL alarm call species
# This is what we use for general filtering
ALARM_BAND = (200, 5000)

# Wider band that also includes tiger roar
FULL_DETECTION_BAND = (50, 5000)


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    lowcut: float = None,
    highcut: float = None,
    order: int = 5
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    This removes energy outside the frequency range where alarm calls live.
    Rain is broadband (all frequencies). Cicadas are typically above 5kHz.
    Wind rumble is below 100Hz. The bandpass cuts all of that.

    Args:
        audio: Audio waveform
        sr: Sample rate
        lowcut: Low frequency cutoff (Hz). Default uses FULL_DETECTION_BAND
        highcut: High frequency cutoff (Hz). Default uses FULL_DETECTION_BAND
        order: Filter order (higher = sharper cutoff, but can ring)

    Returns:
        Filtered audio
    """
    if lowcut is None:
        lowcut = FULL_DETECTION_BAND[0]
    if highcut is None:
        highcut = FULL_DETECTION_BAND[1]

    # Nyquist frequency
    nyquist = sr / 2.0

    # Normalize frequencies to Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    # Design Butterworth bandpass filter
    # Using second-order sections (sos) for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')

    # Apply filter
    filtered = sosfilt(sos, audio)

    return filtered.astype(np.float32)


def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_duration: float = 0.5,
    subtraction_factor: float = 2.0,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Estimate background noise from a quiet segment and subtract it.

    Assumes the first `noise_duration` seconds are "noise only" —
    no alarm calls, just ambient forest sound. This is usually true
    because alarm calls are events, not continuous.

    For a 72-hour recording, you'd estimate noise from multiple
    quiet segments and average them.

    Args:
        audio: Audio waveform
        sr: Sample rate
        noise_duration: Seconds of audio to use for noise estimation
        subtraction_factor: How aggressively to subtract (>1 = more aggressive)
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Noise-reduced audio
    """
    # Compute STFT of full signal
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise spectrum from first N seconds
    noise_samples = int(noise_duration * sr)
    noise_segment = audio[:noise_samples]
    noise_stft = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    # Subtract noise estimate from signal magnitude
    cleaned_magnitude = magnitude - (subtraction_factor * noise_magnitude)

    # Floor at zero — can't have negative energy
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

    # Reconstruct with original phase
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length)

    return cleaned_audio.astype(np.float32)


def energy_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40,
    frame_length: int = 2048,
    hop_length: int = 512
) -> bool:
    """
    Check if an audio clip has enough energy to be worth classifying.

    Most of a 72-hour recording is silence, wind, or rain.
    No point running the CNN on a clip that's clearly just noise.
    This gate rejects clips below an energy threshold.

    Args:
        audio: Audio clip
        sr: Sample rate
        threshold_db: Minimum RMS energy in decibels
        frame_length: Analysis frame length
        hop_length: Hop length

    Returns:
        True if clip has enough energy to classify, False to skip
    """
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    max_rms_db = np.max(rms_db)

    return max_rms_db > threshold_db


def add_rain_noise(
    audio: np.ndarray,
    sr: int,
    intensity: float = 0.3
) -> np.ndarray:
    """
    Simulate rain noise for testing noise filter robustness.
    Rain is broadband noise with slightly more energy in higher frequencies.

    Args:
        audio: Clean audio
        sr: Sample rate
        intensity: Rain intensity (0.0 to 1.0)

    Returns:
        Audio with simulated rain noise added
    """
    # Broadband noise
    rain = np.random.normal(0, intensity * 0.3, len(audio))

    # Rain has more high-frequency energy (patter of drops)
    # Apply a slight high-pass tilt
    from scipy.signal import butter, sosfilt
    sos = butter(2, 500 / (sr / 2), btype='high', output='sos')
    rain_hf = sosfilt(sos, rain)

    # Combine
    rain_combined = 0.6 * rain + 0.4 * rain_hf.astype(np.float64)

    noisy = audio.astype(np.float64) + rain_combined
    return np.clip(noisy, -1, 1).astype(np.float32)


def full_pipeline(
    audio: np.ndarray,
    sr: int,
    apply_bandpass: bool = True,
    apply_spectral_sub: bool = True,
    apply_gate: bool = True,
    gate_threshold_db: float = -40
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Run the complete noise filtering pipeline on an audio clip.

    This is what you call on every 3-5 second window from the AudioMoth
    recording. It filters, cleans, and gates — returning None for clips
    that are just noise, and cleaned audio for clips worth classifying.

    Args:
        audio: Raw audio clip
        sr: Sample rate
        apply_bandpass: Whether to apply bandpass filter
        apply_spectral_sub: Whether to apply spectral subtraction
        apply_gate: Whether to apply energy gate
        gate_threshold_db: Gate threshold

    Returns:
        Tuple of (cleaned_audio_or_None, metadata_dict)
    """
    metadata = {
        'original_rms': float(np.sqrt(np.mean(audio ** 2))),
        'passed_gate': True,
        'bandpass_applied': apply_bandpass,
        'spectral_sub_applied': apply_spectral_sub,
    }

    processed = audio.copy()

    # Stage 1: Bandpass filter
    if apply_bandpass:
        processed = bandpass_filter(processed, sr)

    # Stage 2: Spectral subtraction
    if apply_spectral_sub:
        processed = spectral_subtraction(processed, sr)

    # Stage 3: Energy gate
    if apply_gate:
        passed = energy_gate(processed, sr, threshold_db=gate_threshold_db)
        metadata['passed_gate'] = passed
        if not passed:
            return None, metadata

    metadata['processed_rms'] = float(np.sqrt(np.mean(processed ** 2)))

    return processed, metadata


# ---- Quick test ----
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from preprocessing.audio_loader import load_audio, CLIP_DURATION

    print("=== Noise Filter Test ===\n")

    # Load a clean sambar call
    audio, sr = load_audio("data/raw/sambar/sambar_sample_001.wav", duration=CLIP_DURATION)
    print(f"Original sambar call: RMS={np.sqrt(np.mean(audio**2)):.4f}")

    # Add rain noise
    noisy = add_rain_noise(audio, sr, intensity=0.5)
    print(f"With rain noise:      RMS={np.sqrt(np.mean(noisy**2)):.4f}")

    # Run through pipeline
    cleaned, meta = full_pipeline(noisy, sr)
    if cleaned is not None:
        print(f"After filtering:      RMS={meta['processed_rms']:.4f}")
        print(f"Gate passed: {meta['passed_gate']}")
    else:
        print("Clip rejected by energy gate")

    # Test ambient (should be rejected by gate or show low energy)
    ambient, sr = load_audio("data/raw/ambient/ambient_sample_001.wav", duration=CLIP_DURATION)
    cleaned_amb, meta_amb = full_pipeline(ambient, sr, gate_threshold_db=-30)
    print(f"\nAmbient clip gate passed: {meta_amb['passed_gate']}")

    print("\nNoise filter working.")
