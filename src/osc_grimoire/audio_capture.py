from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Self

# `keyboard` hooks at OS level. On Windows it works without elevation; on Linux
# it requires root. M1 targets Windows.
import keyboard
import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from .config import AudioConfig

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]


class PushToTalkRecorder:
    def __init__(
        self,
        audio_config: AudioConfig,
        hotkey: str = "space",
        on_state_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._audio_config = audio_config
        self._hotkey = hotkey
        self._on_state_change = on_state_change
        self._lock = threading.Lock()
        self._buffer: list[FloatArray] = []
        self._recording = False
        self._is_held = False
        self._press_event = threading.Event()
        self._release_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._press_hook = None
        self._release_hook = None

    def __enter__(self) -> Self:
        self._stream = sd.InputStream(
            samplerate=self._audio_config.sample_rate,
            channels=self._audio_config.channels,
            dtype="float32",
            device=self._audio_config.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._press_hook = keyboard.on_press_key(self._hotkey, self._on_press)
        self._release_hook = keyboard.on_release_key(self._hotkey, self._on_release)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # `keyboard` keys press/release hooks under the same _hooks[hotkey]
        # entry, so calling unhook twice on the matching pair raises KeyError.
        # Drop the pair via unhook() with each handle, swallowing the second
        # KeyError if the library already cleaned it up.
        for hook in (self._press_hook, self._release_hook):
            if hook is None:
                continue
            try:
                keyboard.unhook(hook)
            except KeyError:
                pass
        self._press_hook = None
        self._release_hook = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def record_one(self, max_seconds: float = 30.0) -> FloatArray:
        self._press_event.clear()
        self._release_event.clear()
        with self._lock:
            self._buffer.clear()
            self._recording = False
            self._is_held = False

        self._press_event.wait()
        if not self._release_event.wait(timeout=max_seconds):
            LOGGER.warning(
                "Recording exceeded %.1fs without release; cutting off.", max_seconds
            )
            with self._lock:
                self._recording = False

        with self._lock:
            chunks = list(self._buffer)
            self._buffer.clear()

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            LOGGER.debug("sounddevice status: %s", status)
        with self._lock:
            if not self._recording:
                return
            if indata.shape[1] == 1:
                self._buffer.append(indata[:, 0].copy())
            else:
                self._buffer.append(indata.mean(axis=1).astype(np.float32).copy())

    def _on_press(self, _event: object) -> None:
        if self._is_held:
            return
        self._is_held = True
        with self._lock:
            self._buffer.clear()
            self._recording = True
        self._release_event.clear()
        self._press_event.set()
        if self._on_state_change is not None:
            self._on_state_change(True)

    def _on_release(self, _event: object) -> None:
        if not self._is_held:
            return
        self._is_held = False
        with self._lock:
            self._recording = False
        self._release_event.set()
        if self._on_state_change is not None:
            self._on_state_change(False)
