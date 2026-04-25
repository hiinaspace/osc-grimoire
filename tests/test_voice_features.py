from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.voice_features import extract_mfcc, extract_mfcc_from_array


def _sine_wave(
    freq: float = 440.0, duration_s: float = 1.0, sr: int = 16000
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_extract_mfcc_from_wav(tmp_path: Path) -> None:
    sr = 16000
    audio = _sine_wave(440.0, 1.0, sr)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), audio, sr)

    config = VoiceRecognitionConfig()
    features = extract_mfcc(wav_path, config, sample_rate=sr)
    assert features.ndim == 2
    expected_coefficients = config.n_mfcc - (1 if config.drop_mfcc_c0 else 0)
    assert features.shape[1] == expected_coefficients
    assert features.dtype == np.float32
    assert features.shape[0] > 0


def test_extract_mfcc_from_array_matches_wav(tmp_path: Path) -> None:
    sr = 16000
    audio = _sine_wave(880.0, 1.0, sr)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), audio, sr)

    config = VoiceRecognitionConfig()
    from_wav = extract_mfcc(wav_path, config, sample_rate=sr)
    from_array = extract_mfcc_from_array(audio, config, sample_rate=sr)
    assert from_wav.shape == from_array.shape
    # WAV round-trip introduces ~1e-2 float32 noise; coefficients are O(100).
    np.testing.assert_allclose(from_wav, from_array, atol=0.1)
