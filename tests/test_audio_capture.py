from __future__ import annotations

import numpy as np

from osc_grimoire.audio_capture import NonBlockingAudioRecorder
from osc_grimoire.config import AudioConfig


def test_nonblocking_recorder_captures_between_begin_and_end() -> None:
    states: list[bool] = []
    recorder = NonBlockingAudioRecorder(
        AudioConfig(sample_rate=16000),
        on_state_change=states.append,
    )

    recorder._audio_callback(np.ones((4, 1), dtype=np.float32), 4, None, _NoStatus())
    recorder.begin_recording()
    recorder._audio_callback(np.ones((4, 1), dtype=np.float32), 4, None, _NoStatus())
    recorder._audio_callback(
        np.full((4, 1), 2.0, dtype=np.float32), 4, None, _NoStatus()
    )
    audio = recorder.end_recording()

    assert states == [True, False]
    np.testing.assert_allclose(
        audio, np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
    )


def test_nonblocking_recorder_mixes_multichannel_audio() -> None:
    recorder = NonBlockingAudioRecorder(AudioConfig(sample_rate=16000, channels=2))
    recorder.begin_recording()
    recorder._audio_callback(
        np.array([[1.0, -1.0], [0.5, 1.0]], dtype=np.float32),
        2,
        None,
        _NoStatus(),
    )

    audio = recorder.end_recording()

    np.testing.assert_allclose(audio, np.array([0.0, 0.75], dtype=np.float32))


class _NoStatus:
    def __bool__(self) -> bool:
        return False
