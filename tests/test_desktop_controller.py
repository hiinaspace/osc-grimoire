from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf

from osc_grimoire.audio_playback import load_audio_for_playback
from osc_grimoire.config import (
    AppConfig,
    AudioConfig,
    GestureRecognitionConfig,
    VoiceRecognitionConfig,
)
from osc_grimoire.desktop_controller import VoiceTrainingController
from osc_grimoire.gesture_recognizer import load_gesture_points
from osc_grimoire.spellbook import load_spellbook
from osc_grimoire.voice_features import FloatArray
from osc_grimoire.voice_recognizer import VoiceTemplateBackend
from osc_grimoire.waveform import downsample_waveform, load_waveform_preview


def test_controller_create_cancel_draft_without_persisting(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    draft = controller.start_draft()
    controller.update_draft_name("Ignis")
    controller.cancel_draft()

    assert draft.name == "New Spell 1"
    assert controller.draft is None
    assert load_spellbook(tmp_path).spells == ()


def test_controller_persists_draft_on_first_sample_and_deletes_sample(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.start_draft()
    controller.update_draft_name("Ignis")

    spell = controller.add_sample_to_draft(_audio(440))

    assert spell.name == "Ignis"
    assert len(spell.voice_samples) == 1
    sample_path = tmp_path / spell.voice_samples[0]
    assert sample_path.exists()

    updated = controller.delete_sample(spell.id, spell.voice_samples[0])

    assert updated.voice_samples == ()
    assert not sample_path.exists()


def test_controller_deletes_spell_and_sample_directory(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.add_sample_to_draft(_audio(440))
    sample_path = tmp_path / spell.voice_samples[0]
    sample_dir = sample_path.parent

    deleted_name = controller.delete_spell(spell.id)

    assert deleted_name == spell.name
    assert load_spellbook(tmp_path).spells == ()
    assert not sample_dir.exists()
    assert controller.ui_log[-1].message == f"Deleted spell: {spell.name}"


def test_controller_does_not_persist_draft_on_empty_sample(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start_draft()
    controller.update_draft_name("Ignis")

    with pytest.raises(ValueError, match="No audio captured"):
        controller.add_sample_to_draft(np.zeros(0, dtype=np.float32))

    assert load_spellbook(tmp_path).spells == ()


def test_controller_renames_spell_and_keeps_names_unique(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    first = controller.add_sample_to_draft(_audio(440))
    controller.start_draft()
    controller.update_draft_name("Other")
    second = controller.add_sample_to_draft(_audio(660))

    renamed = controller.rename_spell(second.id, first.name)

    assert renamed.name == f"{first.name} 2"


def test_controller_recognizes_with_fake_backend(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start_draft()
    controller.update_draft_name("Low")
    controller.add_sample_to_draft(np.full(16000, 0.1, dtype=np.float32))
    controller.start_draft()
    controller.update_draft_name("High")
    controller.add_sample_to_draft(np.full(16000, 0.8, dtype=np.float32))

    result = controller.recognize(np.full(16000, 0.11, dtype=np.float32))

    assert result.decision.accepted
    assert result.ranking[0].name == "Low"
    assert "decision: ACCEPTED" in result.debug_text


def test_controller_pulses_spell_on_accepted_voice(tmp_path: Path) -> None:
    output = _FakeOutput()
    controller = _controller(tmp_path)
    controller.output = output
    controller.start_draft()
    controller.update_draft_name("Low")
    controller.add_sample_to_draft(np.full(16000, 0.1, dtype=np.float32))

    controller.recognize(np.full(16000, 0.11, dtype=np.float32))

    assert output.spell_pulses == ["Low"]
    assert output.fizzle_count == 0
    assert controller.ui_log[-1].message == "Voice cast: Low"


def test_controller_pulses_fizzle_on_rejected_voice(tmp_path: Path) -> None:
    output = _FakeOutput()
    controller = _controller(tmp_path)
    controller.output = output

    controller.recognize(np.full(16000, 0.11, dtype=np.float32))

    assert output.spell_pulses == []
    assert output.fizzle_count == 1
    assert controller.ui_log[-1].message.startswith("Voice fizzle:")


def test_controller_preloads_backend(tmp_path: Path) -> None:
    backend = _CountingBackend()
    controller = VoiceTrainingController(
        tmp_path,
        config=AppConfig(audio=AudioConfig(sample_rate=16000)),
        backend=backend.backend,
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
    )

    controller.preload_backend()

    assert backend.extract_array_calls == 1


def test_controller_updates_warm_feature_cache_incrementally(tmp_path: Path) -> None:
    backend = _CountingBackend()
    controller = VoiceTrainingController(
        tmp_path,
        config=AppConfig(audio=AudioConfig(sample_rate=16000)),
        backend=backend.backend,
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
    )
    spell = controller.add_sample_to_draft(_audio(440))
    controller._recognition_cache()
    backend.extract_path_calls = 0
    backend.extract_array_calls = 0

    controller.add_sample_to_spell(spell.id, _audio(660))

    assert backend.extract_path_calls == 0
    assert backend.extract_array_calls == 1


def test_controller_trims_saved_voice_samples(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()
    silence = np.zeros(4000, dtype=np.float32)
    audio = np.concatenate([silence, _audio(440), silence])

    updated = controller.add_sample_to_spell(spell.id, audio)

    saved, sample_rate = sf.read(
        str(tmp_path / updated.voice_samples[0]), dtype="float32"
    )
    assert sample_rate == 16000
    assert 0 < saved.size < audio.size


def test_load_audio_for_playback_reads_float32_sample(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    sf.write(str(path), _audio(440), 16000)

    audio, sample_rate = load_audio_for_playback(path)

    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.size > 0


def test_controller_plays_individual_and_random_samples(tmp_path: Path) -> None:
    player = _FakeAudioPlayer()
    controller = _controller(tmp_path, audio_player=player)
    spell = controller.add_sample_to_draft(_audio(440))

    controller.play_sample(spell.voice_samples[0])
    controller.play_random_sample(spell.id)

    expected_path = tmp_path / spell.voice_samples[0]
    assert player.paths == [expected_path, expected_path]
    assert controller.status == f"Playing {spell.name} sample."


def test_controller_saves_and_overwrites_gesture_sample(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(min_points=3),
    )
    spell = controller.add_sample_to_draft(_audio(440))

    controller.arm_gesture_recording(spell.id)
    controller.handle_gesture_stroke(_gesture_line())
    controller.arm_gesture_recording(spell.id)
    controller.handle_gesture_stroke(_gesture_zigzag())

    fresh = load_spellbook(tmp_path).spells[0]
    assert fresh.has_gesture
    assert len(fresh.gesture_samples) == 1
    points = load_gesture_points(tmp_path / fresh.gesture_samples[0])
    np.testing.assert_allclose(points, _gesture_zigzag())


def test_controller_clears_gesture_sample(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(min_points=3),
    )
    spell = controller.add_sample_to_draft(_audio(440))
    controller.save_gesture_sample(spell.id, _gesture_line())
    fresh = load_spellbook(tmp_path).spells[0]
    gesture_path = tmp_path / fresh.gesture_samples[0]

    updated = controller.clear_gesture_sample(spell.id)

    assert not updated.has_gesture
    assert updated.gesture_samples == ()
    assert not gesture_path.exists()
    assert controller.ui_log[-1].message == f"Cleared gesture: {spell.name}"


def test_controller_recognizes_gesture(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(
            min_points=3, score_min=0.5, margin_min=0.01
        ),
    )
    spell = controller.add_sample_to_draft(_audio(440))
    controller.save_gesture_sample(spell.id, _gesture_line())

    result = controller.recognize_gesture(_gesture_line())

    assert result.decision.accepted
    assert result.ranking[0].name == spell.name


def test_controller_pulses_outputs_for_gesture_results(tmp_path: Path) -> None:
    output = _FakeOutput()
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(
            min_points=3, score_min=0.5, margin_min=0.01
        ),
    )
    controller.output = output
    spell = controller.add_sample_to_draft(_audio(440))
    controller.save_gesture_sample(spell.id, _gesture_line())

    controller.recognize_gesture(_gesture_line())
    controller.recognize_gesture(np.zeros((2, 2), dtype=np.float32))

    assert output.spell_pulses == [spell.name]
    assert output.fizzle_count == 1
    assert controller.ui_log[-1].message == "Gesture rejected: gesture too short"


def test_controller_rejects_short_gesture_without_mutation(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(min_points=4),
    )
    spell = controller.add_sample_to_draft(_audio(440))

    with pytest.raises(ValueError):
        controller.save_gesture_sample(spell.id, np.zeros((2, 2), dtype=np.float32))

    assert load_spellbook(tmp_path).spells[0].gesture_samples == ()


def test_waveform_preview_downsamples_and_loads_wav(tmp_path: Path) -> None:
    audio = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    preview = downsample_waveform(audio, points=25)
    path = tmp_path / "sample.wav"
    sf.write(str(path), audio, 16000)
    loaded = load_waveform_preview(path, points=25)

    assert preview.shape == (25,)
    assert loaded.shape == (25,)
    assert np.max(np.abs(preview)) <= 1.0


def test_desktop_ui_import_smoke() -> None:
    import osc_grimoire.desktop_ui as desktop_ui

    assert desktop_ui.PAGE_MAIN == 0


def test_desktop_ui_pages_follow_spell_order(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import PAGE_DIAGNOSTICS, PAGE_MAIN, DesktopVoiceUi

    controller = _controller(tmp_path)
    first = controller.add_sample_to_draft(_audio(440))
    controller.start_draft()
    controller.update_draft_name("Other")
    controller.add_sample_to_draft(_audio(660))
    ui = DesktopVoiceUi(controller)

    assert ui._ordered_pages() == [PAGE_MAIN, 1, 2, PAGE_DIAGNOSTICS]
    ui._go_next_page()
    assert ui.page == 1
    assert ui.selected_spell_id == first.id


def test_desktop_ui_invalid_spell_page_does_not_auto_start_draft(
    tmp_path: Path,
) -> None:
    from osc_grimoire.desktop_ui import PAGE_MAIN, DesktopVoiceUi

    controller = _controller(tmp_path)
    ui = DesktopVoiceUi(controller)
    ui.page = 1

    ui._draw_spell_page()

    assert ui.page == PAGE_MAIN
    assert controller.draft is None


def test_desktop_ui_overlay_mode_disables_spell_name_editing(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    ui = DesktopVoiceUi(controller, overlay_mode=True)

    assert not ui._can_edit_spell_names()


def test_desktop_ui_overlay_keyboard_finish_updates_spell(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.add_sample_to_draft(_audio(440))
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    ui.keyboard_editing = True
    ui.keyboard_edit_spell_id = spell.id
    ui.edit_name = "Ignis"

    ui.finish_keyboard_name(commit=True)

    assert controller.spellbook.spells[0].name == "Ignis"
    assert ui.edit_name == "Ignis"


def test_desktop_ui_overlay_keyboard_cancel_restores_name(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.add_sample_to_draft(_audio(440))
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    close_count = 0

    def close_keyboard() -> None:
        nonlocal close_count
        close_count += 1

    ui.keyboard_close_handler = close_keyboard
    ui.keyboard_editing = True
    ui.keyboard_edit_spell_id = spell.id
    ui.keyboard_original_name = spell.name
    ui.edit_name = "Changed"

    ui.finish_keyboard_name(commit=False)

    assert controller.spellbook.spells[0].name == spell.name
    assert ui.edit_name == spell.name
    assert close_count == 1


def test_desktop_ui_spoken_name_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.add_sample_to_draft(_audio(440))
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    ui.selected_spell_id = spell.id
    ui.keyboard_editing = True
    ui.keyboard_edit_spell_id = spell.id
    recorder = _FakeRecorder()
    ui.recorder = cast("Any", recorder)
    monkeypatch.setattr(controller, "suggest_spell_name", lambda _audio: "Lumos")

    ui._begin_recording("name", "ui")
    ui._finish_recording("name", "ui")

    assert controller.spellbook.spells[0].name == spell.name
    assert ui.edit_name == "Lumos"
    assert ui.pending_spoken_name is None

    ui.finish_keyboard_name(commit=True)

    assert controller.spellbook.spells[0].name == "Lumos"


def test_desktop_ui_button_release_does_not_end_overlay_recording(
    tmp_path: Path,
) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    recorder = _FakeRecorder()
    ui.recorder = cast("Any", recorder)

    ui.begin_overlay_voice_recording()
    ui._update_hold_recording("recognize", held=False)

    assert ui.recording_mode == "recognize"
    assert ui.recording_source == "overlay"
    assert recorder.begin_count == 1
    assert recorder.end_count == 0

    ui.finish_overlay_voice_recording()

    assert ui.recording_mode is None
    assert ui.recording_source is None
    assert recorder.end_count == 1


def _controller(
    data_dir: Path,
    gesture_config: GestureRecognitionConfig | None = None,
    audio_player: Any | None = None,
) -> VoiceTrainingController:
    config = AppConfig(
        audio=AudioConfig(sample_rate=16000),
        gesture=gesture_config or GestureRecognitionConfig(),
    )
    return VoiceTrainingController(
        data_dir,
        config=config,
        backend=_fake_backend(),
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
        audio_player=audio_player,
    )


class _FakeAudioPlayer:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def play_file(self, path: Path) -> None:
        self.paths.append(path)


class _FakeRecorder:
    def __init__(self) -> None:
        self.begin_count = 0
        self.end_count = 0

    def begin_recording(self) -> None:
        self.begin_count += 1

    def end_recording(self) -> FloatArray:
        self.end_count += 1
        return np.zeros(1600, dtype=np.float32)


class _FakeOutput:
    status_text = "OSC target: fake"

    def __init__(self) -> None:
        self.voice_recording: list[bool] = []
        self.gesture_drawing: list[bool] = []
        self.spell_pulses: list[str] = []
        self.fizzle_count = 0
        self.tick_count = 0

    def set_voice_recording(self, recording: bool) -> None:
        self.voice_recording.append(recording)

    def set_gesture_drawing(self, drawing: bool) -> None:
        self.gesture_drawing.append(drawing)

    def pulse_spell(self, spell) -> None:
        self.spell_pulses.append(spell.name)

    def pulse_fizzle(self) -> None:
        self.fizzle_count += 1

    def tick(self, now=None) -> None:
        self.tick_count += 1


def _fake_backend() -> VoiceTemplateBackend:
    def extract_path(path: Path, _config: VoiceRecognitionConfig) -> FloatArray:
        audio, _sample_rate = sf.read(str(path), dtype="float32")
        return np.array([[float(np.mean(audio))]], dtype=np.float32)

    def extract_array(
        audio: FloatArray, _config: VoiceRecognitionConfig, _sample_rate: int
    ) -> FloatArray:
        return np.array([[float(np.mean(audio))]], dtype=np.float32)

    return VoiceTemplateBackend(
        name="fake",
        extract_path=extract_path,
        extract_array=extract_array,
        distance=lambda a, b: float(abs(a[0, 0] - b[0, 0])),
        aggregate=lambda distances: float(np.median(distances)),
    )


class _CountingBackend:
    def __init__(self) -> None:
        self.extract_array_calls = 0
        self.extract_path_calls = 0
        self.backend = VoiceTemplateBackend(
            name="counting",
            extract_path=self.extract_path,
            extract_array=self.extract_array,
            distance=lambda a, b: float(abs(a[0, 0] - b[0, 0])),
            aggregate=lambda distances: float(np.median(distances)),
        )

    def extract_path(self, _path: Path, _config: VoiceRecognitionConfig) -> FloatArray:
        self.extract_path_calls += 1
        return np.array([[0.0]], dtype=np.float32)

    def extract_array(
        self, audio: FloatArray, _config: VoiceRecognitionConfig, _sample_rate: int
    ) -> FloatArray:
        self.extract_array_calls += 1
        return np.array([[float(audio.size)]], dtype=np.float32)


def _audio(frequency: float) -> FloatArray:
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _gesture_line() -> FloatArray:
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    return np.column_stack([x, np.zeros_like(x)]).astype(np.float32)


def _gesture_zigzag() -> FloatArray:
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    y = np.where(np.arange(12) % 2 == 0, 0.0, 0.4).astype(np.float32)
    return np.column_stack([x, y]).astype(np.float32)
