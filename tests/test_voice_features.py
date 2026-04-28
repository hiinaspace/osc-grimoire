from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.voice_features import (
    load_audio_mono,
    resample_audio,
    trim_voice_audio,
)


def _sine_wave(
    freq: float = 440.0, duration_s: float = 1.0, sr: int = 16000
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_load_audio_mono_resamples_wav(tmp_path: Path) -> None:
    source_rate = 8000
    audio = _sine_wave(440.0, 0.25, source_rate)
    stereo = np.column_stack([audio, audio * 0.5])
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), stereo, source_rate)

    loaded = load_audio_mono(wav_path, sample_rate=16000)

    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert abs(loaded.shape[0] - 4000) <= 1
    assert np.max(np.abs(loaded)) > 0.0


def test_resample_audio_preserves_duration() -> None:
    audio = _sine_wave(440.0, 0.5, 8000)

    resampled = resample_audio(audio, source_rate=8000, target_rate=16000)

    assert resampled.dtype == np.float32
    assert resampled.shape == (8000,)


def test_trim_voice_audio_removes_leading_and_trailing_silence() -> None:
    sr = 16000
    silence = np.zeros(sr // 4, dtype=np.float32)
    tone = _sine_wave(440.0, 0.5, sr)
    audio = np.concatenate([silence, tone, silence])

    trimmed = trim_voice_audio(audio, VoiceRecognitionConfig())

    assert 0 < trimmed.size < audio.size


def test_trim_voice_audio_uses_librosa_style_frame_indices() -> None:
    config = VoiceRecognitionConfig(trim_top_db=30.0)
    silence = np.zeros(1024, dtype=np.float32)
    tone = np.ones(4096, dtype=np.float32) * 0.25
    audio = np.concatenate([silence, tone, silence])

    trimmed = trim_voice_audio(audio, config)

    np.testing.assert_array_equal(trimmed, audio[512:6144])


def test_trim_voice_audio_keeps_all_silence() -> None:
    audio = np.zeros(1000, dtype=np.float32)

    trimmed = trim_voice_audio(audio, VoiceRecognitionConfig())

    np.testing.assert_array_equal(trimmed, audio)
