"""
Sample Data Generator
Generates synthetic alarm call audio that matches real frequency profiles.

USE THIS FOR PIPELINE DEVELOPMENT ONLY.
Replace with real xeno-canto downloads for actual model training.

To download real data, search xeno-canto.org for:
  - Rusa unicolor (sambar deer) — filter by type: "alarm"
  - Semnopithecus (langur) — filter by type: "alarm"
  - Axis axis (chital/spotted deer) — filter by type: "alarm"
  - Panthera tigris — any vocalization type
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sambar_alarm(duration=3.0, sr=22050):
    """Sambar 'dhank' — deep resonant bark, 500Hz-2kHz, sharp onset."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    n_bursts = np.random.randint(2, 4)
    burst_starts = sorted(np.random.uniform(0.2, duration * 0.6, n_bursts))
    for bs in burst_starts:
        bd = np.random.uniform(0.25, 0.4)
        mask = (t >= bs) & (t < bs + bd)
        bt = t[mask] - bs
        f0 = np.random.uniform(700, 900)
        bark = (0.7 * np.sin(2 * np.pi * f0 * bt) +
                0.4 * np.sin(2 * np.pi * f0 * 1.5 * bt) +
                0.2 * np.sin(2 * np.pi * f0 * 2.0 * bt))
        envelope = np.exp(-3 * bt) * (1 - np.exp(-50 * bt))
        signal[mask] = bark * envelope
    noise = np.random.normal(0, 0.02, len(t))
    return np.clip(signal + noise, -1, 1).astype(np.float32)


def generate_langur_alarm(duration=3.0, sr=22050):
    """Langur 'khok-khok' — high-pitched rapid bark, 1kHz-4kHz."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    n_bursts = np.random.randint(4, 7)
    spacing = np.random.uniform(0.25, 0.4)
    start = np.random.uniform(0.1, 0.4)
    for i in range(n_bursts):
        bs = start + i * spacing
        if bs + 0.15 > duration:
            break
        bd = np.random.uniform(0.1, 0.18)
        mask = (t >= bs) & (t < bs + bd)
        bt = t[mask] - bs
        f0 = np.random.uniform(1800, 2200)
        bark = (0.6 * np.sin(2 * np.pi * f0 * bt) +
                0.4 * np.sin(2 * np.pi * f0 * 1.5 * bt) +
                0.2 * np.sin(2 * np.pi * f0 * 1.9 * bt))
        envelope = np.exp(-8 * bt) * (1 - np.exp(-80 * bt))
        signal[mask] = bark * envelope
    noise = np.random.normal(0, 0.02, len(t))
    return np.clip(signal + noise, -1, 1).astype(np.float32)


def generate_chital_alarm(duration=3.0, sr=22050):
    """Chital alarm bark — sharp whistle-bark, 1kHz-3kHz, frequency sweep."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    n_bursts = np.random.randint(2, 5)
    burst_starts = sorted(np.random.uniform(0.2, duration * 0.7, n_bursts))
    for bs in burst_starts:
        bd = np.random.uniform(0.2, 0.3)
        mask = (t >= bs) & (t < bs + bd)
        bt = t[mask] - bs
        f0 = np.random.uniform(1400, 1700)
        freq_sweep = f0 + 500 * bt / bd
        bark = (0.6 * np.sin(2 * np.pi * freq_sweep * bt) +
                0.3 * np.sin(2 * np.pi * 2200 * bt) +
                0.2 * np.sin(2 * np.pi * 2800 * bt))
        envelope = np.exp(-5 * bt) * (1 - np.exp(-60 * bt))
        signal[mask] = bark * envelope
    noise = np.random.normal(0, 0.02, len(t))
    return np.clip(signal + noise, -1, 1).astype(np.float32)


def generate_tiger_vocal(duration=4.0, sr=22050):
    """Tiger roar — very low frequency 50-200Hz, sustained, powerful."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    roar_start = np.random.uniform(0.3, 0.7)
    roar_dur = np.random.uniform(2.0, 3.0)
    mask = (t >= roar_start) & (t < roar_start + roar_dur)
    rt = t[mask] - roar_start
    f0 = np.random.uniform(70, 100)
    vibrato = 5 * np.sin(2 * np.pi * 4 * rt)
    roar = (0.8 * np.sin(2 * np.pi * (f0 + vibrato) * rt) +
            0.5 * np.sin(2 * np.pi * f0 * 2 * rt) +
            0.3 * np.sin(2 * np.pi * f0 * 3 * rt) +
            0.15 * np.sin(2 * np.pi * f0 * 4 * rt))
    envelope = (1 - np.exp(-3 * rt)) * np.exp(-0.5 * rt)
    signal[mask] = roar * envelope
    noise = np.random.normal(0, 0.02, len(t))
    return np.clip(signal + noise, -1, 1).astype(np.float32)


def generate_ambient(duration=3.0, sr=22050):
    """Forest ambient — cicadas, wind, general background. No alarm calls."""
    t = np.linspace(0, duration, int(sr * duration))
    cicada_freq = np.random.uniform(5500, 7000)
    cicada = 0.05 * np.sin(2 * np.pi * cicada_freq * t +
                           3 * np.sin(2 * np.pi * 8 * t))
    wind = 0.03 * np.sin(2 * np.pi * 50 * t +
                         2 * np.sin(2 * np.pi * 0.5 * t))
    noise = np.random.normal(0, 0.03, len(t))
    return np.clip(cicada + wind + noise, -1, 1).astype(np.float32)


GENERATORS = {
    'sambar': generate_sambar_alarm,
    'langur': generate_langur_alarm,
    'chital': generate_chital_alarm,
    'tiger': generate_tiger_vocal,
    'ambient': generate_ambient,
}


def generate_dataset(base_dir: str, samples_per_class: int = 20):
    """
    Generate a full synthetic dataset for pipeline development.

    Args:
        base_dir: Base directory (e.g., 'data/raw')
        samples_per_class: Number of samples per species
    """
    base = Path(base_dir)

    for species, generator in GENERATORS.items():
        species_dir = base / species
        species_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            audio = generator()
            filepath = species_dir / f"{species}_sample_{i+1:03d}.wav"
            sf.write(str(filepath), audio, 22050)

        print(f"  {species}: {samples_per_class} samples")

    total = samples_per_class * len(GENERATORS)
    print(f"\nTotal: {total} samples generated")
    print("NOTE: Replace with real xeno-canto data for model training.")


if __name__ == "__main__":
    print("Generating synthetic dataset...\n")
    generate_dataset("data/raw", samples_per_class=200)
