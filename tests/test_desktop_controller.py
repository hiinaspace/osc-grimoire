from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from osc_grimoire.audio_playback import load_audio_for_playback
from osc_grimoire.config import (
    AppConfig,
    AudioConfig,
    GestureRecognitionConfig,
    VoiceRecognitionConfig,
)
from osc_grimoire.desktop_controller import GrimoireController
from osc_grimoire.gesture_recognizer import load_gesture_points
from osc_grimoire.spellbook import PRESET_SPELL_NAMES, OscAction, load_spellbook
from osc_grimoire.voice_features import FloatArray
from osc_grimoire.voice_recognizer import TextHypothesis, VoiceTemplateBackend
from osc_grimoire.waveform import downsample_waveform, load_waveform_preview


def test_controller_loads_preset_spells(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    assert tuple(s.name for s in controller.spellbook.spells) == PRESET_SPELL_NAMES


def test_controller_create_cancel_draft_without_persisting(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    draft = controller.start_draft()
    controller.update_draft_name("Ignis")
    controller.cancel_draft()

    assert draft.name == "New Spell 5"
    assert controller.draft is None
    assert load_spellbook(tmp_path).spells == controller.spellbook.spells


def test_controller_persists_draft_with_default_alias(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start_draft()
    controller.update_draft_name("Ignis")

    spell = controller.persist_draft()

    assert spell.name == "Ignis"
    assert spell.voice_aliases == ("Ignis",)
    assert load_spellbook(tmp_path).spells[-1].name == "Ignis"


def test_controller_renames_spell_and_updates_default_alias(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()

    renamed = controller.rename_spell(spell.id, "Ignis")

    assert renamed.name == "Ignis"
    assert renamed.voice_aliases == ("Ignis",)


def test_controller_adds_and_removes_voice_alias(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()

    updated = controller.add_voice_alias(spell.id, "In nis")
    assert updated.voice_aliases == (spell.name, "In nis")

    updated = controller.remove_voice_alias(spell.id, "In nis")
    assert updated.voice_aliases == (spell.name,)


def test_controller_rejects_invalid_voice_alias(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()

    with pytest.raises(ValueError):
        controller.add_voice_alias(spell.id, "")


def test_controller_updates_spell_osc_parameter(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()

    updated = controller.update_spell_osc_address(spell.id, "CustomFire")

    assert updated.osc_address == "CustomFire"
    assert controller.spell_osc_parameter_name(updated) == "CustomFire"
    assert load_spellbook(tmp_path).spells[-1].osc_address == "CustomFire"

    reset = controller.update_spell_osc_address(spell.id, "")

    assert reset.osc_address is None
    assert controller.spell_osc_parameter_name(reset).startswith("OSCGrimoireSpell")


def test_controller_updates_spell_osc_actions(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    spell = controller.persist_draft()

    updated = controller.update_spell_osc_actions(
        spell.id,
        on_cast=(OscAction("Spell", 3), OscAction("MagicPrepared", True)),
        after_cast=(),
    )

    assert updated.osc_on_cast == (
        OscAction("Spell", 3),
        OscAction("MagicPrepared", True),
    )
    assert updated.osc_after_cast == ()
    assert controller.spell_osc_signal_summary(updated) == "Spell=3, MagicPrepared=true"
    assert load_spellbook(tmp_path).spells[-1].osc_on_cast == updated.osc_on_cast


def test_controller_recognizes_with_fake_backend(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    result = controller.recognize(_audio_for_text("Alohomora"))

    assert result.decision.accepted
    assert result.ranking[0].name == "Alohomora"
    assert result.ranking[0].alias == "Alohomora"
    assert "decision: ACCEPTED" in result.debug_text
    assert result.text_hypotheses[0].text == "Alohomora"
    assert result.pending_incantations[0].text == "Alohomora"
    assert result.pending_incantations[0].distance == pytest.approx(0.0)


def test_controller_pulses_spell_on_accepted_voice(tmp_path: Path) -> None:
    output = _FakeOutput()
    controller = _controller(tmp_path)
    controller.output = output

    controller.recognize(_audio_for_text("Alohomora"))

    assert output.spell_pulses == ["Alohomora"]
    assert output.fizzle_count == 0
    assert (
        controller.ui_log[-1].message
        == "Accepted: Alohomora (osc: OSCGrimoireSpellAlohomora=true -> OSCGrimoireSpellAlohomora=false)"
    )


def test_controller_pulses_fizzle_on_rejected_voice(tmp_path: Path) -> None:
    output = _FakeOutput()
    controller = _controller(tmp_path)
    controller.output = output
    controller.voice_config = VoiceRecognitionConfig(voice_alias_distance_max=0.0)

    controller.recognize(_audio_for_text("Nope"))

    assert output.spell_pulses == []
    assert output.fizzle_count == 1
    assert controller.ui_log[-1].message.startswith("Fizzle (osc: OSCGrimoireFizzle):")


def test_controller_suggest_spell_name_uses_text_hypotheses(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    name = controller.suggest_spell_name(_audio_for_text("Alohomora"))

    assert name == "Alohomora"
    assert controller.last_name_hypotheses[0].text == "Alohomora"


def test_controller_local_input_toggles_combine_with_osc_input(tmp_path: Path) -> None:
    osc_input = _FakeInput()
    output = _FakeOutput()
    controller = _controller(tmp_path)
    controller.osc_input = osc_input
    controller.output = output

    controller.set_voice_enabled(False)
    controller.set_gesture_enabled(False)
    controller.set_ui_enabled(False)

    assert osc_input.voice_enabled is False
    assert osc_input.gesture_enabled is False
    assert osc_input.ui_enabled is False
    assert not controller.voice_enabled
    assert not controller.gesture_enabled
    assert output.voice_enabled == [False]
    assert output.gesture_enabled == [False]
    assert output.ui_enabled == [False]


def test_controller_voice_strictness_updates_voice_thresholds(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)

    controller.set_voice_strictness(0.0)
    assert controller.voice_config.relative_margin_min == 0.0
    assert controller.voice_config.voice_alias_distance_max == pytest.approx(9.0)

    controller.set_voice_strictness(0.30)
    assert controller.voice_config.relative_margin_min == pytest.approx(0.20)
    assert controller.voice_config.voice_alias_distance_max == pytest.approx(7.0)

    controller.set_voice_strictness(1.0)
    assert controller.voice_config.relative_margin_min == pytest.approx(0.45)
    assert controller.voice_config.voice_alias_distance_max == pytest.approx(5.0)


def test_controller_gesture_strictness_updates_gesture_thresholds(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)

    controller.set_gesture_strictness(0.0)
    assert controller.config.gesture.score_min == 0.0
    assert controller.config.gesture.margin_min == 0.0

    controller.set_gesture_strictness(0.30)
    assert controller.config.gesture.score_min == pytest.approx(0.20)
    assert controller.config.gesture.margin_min == pytest.approx(0.03)

    controller.set_gesture_strictness(1.0)
    assert controller.config.gesture.score_min == pytest.approx(0.70)
    assert controller.config.gesture.margin_min == pytest.approx(0.25)


def test_controller_preloads_backend(tmp_path: Path) -> None:
    backend = _CountingBackend()
    controller = GrimoireController(
        tmp_path,
        config=AppConfig(audio=AudioConfig(sample_rate=16000)),
        backend=backend.backend,
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
    )

    controller.preload_backend()

    assert backend.extract_array_calls == 1


def test_controller_saves_and_overwrites_gesture_sample(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(min_points=3),
    )
    spell = controller.persist_draft()

    controller.arm_gesture_recording(spell.id)
    controller.handle_gesture_stroke(_gesture_line())
    controller.arm_gesture_recording(spell.id)
    controller.handle_gesture_stroke(_gesture_zigzag())

    fresh = load_spellbook(tmp_path).spells[-1]
    assert fresh.has_gesture
    assert len(fresh.gesture_samples) == 1
    points = load_gesture_points(tmp_path / fresh.gesture_samples[0])
    np.testing.assert_allclose(points, _gesture_zigzag())


def test_controller_clears_gesture_sample(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        gesture_config=GestureRecognitionConfig(min_points=3),
    )
    spell = controller.persist_draft()
    controller.save_gesture_sample(spell.id, _gesture_line())
    fresh = load_spellbook(tmp_path).spells[-1]
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
    spell = controller.persist_draft()
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
    spell = controller.persist_draft()
    controller.save_gesture_sample(spell.id, _gesture_line())

    controller.recognize_gesture(_gesture_line())
    controller.recognize_gesture(np.zeros((2, 2), dtype=np.float32))

    assert output.spell_pulses == [spell.name]
    assert output.fizzle_count == 1
    assert (
        controller.ui_log[-1].message
        == "Fizzle (osc: OSCGrimoireFizzle): gesture too short"
    )


def test_load_audio_for_playback_reads_float32_sample(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "sample.wav"
    sf.write(str(path), np.zeros(100, dtype=np.float32), 16000)

    audio, sample_rate = load_audio_for_playback(path)

    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.size > 0


def test_waveform_preview_downsamples_and_loads_wav(tmp_path: Path) -> None:
    import soundfile as sf

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


def test_desktop_ui_extracts_osc_parameter_from_log() -> None:
    from osc_grimoire.desktop_ui import _osc_parameter_from_log

    assert (
        _osc_parameter_from_log("[12:00:01] Accepted: Lumos (osc: CustomLumos)")
        == "CustomLumos"
    )
    assert (
        _osc_parameter_from_log("[12:00:01] Accepted: Lumos (osc: Spell=3 -> 0)")
        == "Spell"
    )
    assert _osc_parameter_from_log("[12:00:01] Ready.") is None


def test_desktop_ui_pages_follow_spell_order(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import PAGE_DIAGNOSTICS, PAGE_MAIN, DesktopVoiceUi

    controller = _controller(tmp_path)
    first = controller.spellbook.spells[0]
    ui = DesktopVoiceUi(controller)

    assert ui._ordered_pages() == [PAGE_MAIN, 1, 2, 3, 4, PAGE_DIAGNOSTICS]
    ui._go_next_page()
    assert ui.page == 1
    assert ui.selected_spell_id == first.id


def test_desktop_ui_invalid_spell_page_does_not_auto_start_draft(
    tmp_path: Path,
) -> None:
    from osc_grimoire.desktop_ui import PAGE_MAIN, DesktopVoiceUi

    controller = _controller(tmp_path)
    ui = DesktopVoiceUi(controller)
    ui.page = 99
    ui.selected_spell_id = None

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
    spell = controller.spellbook.spells[0]
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
    spell = controller.spellbook.spells[0]
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


def test_desktop_ui_osc_edit_updates_spell_parameter(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.spellbook.spells[0]
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    ui.osc_editing = True
    ui.osc_edit_spell_id = spell.id
    ui.osc_on_cast_edit = "CustomFire=true"
    ui.osc_after_cast_edit = "CustomFire=false"

    ui._finish_osc_edit(commit=True)

    assert controller.spellbook.spells[0].osc_on_cast == (
        OscAction("CustomFire", True),
    )
    assert controller.spellbook.spells[0].osc_after_cast == (
        OscAction("CustomFire", False),
    )


def test_desktop_ui_osc_edit_updates_custom_spell_actions(tmp_path: Path) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.spellbook.spells[0]
    ui = DesktopVoiceUi(controller, overlay_mode=True)
    ui.osc_editing = True
    ui.osc_edit_spell_id = spell.id
    ui.osc_on_cast_edit = "Spell=4, MagicPrepared=true"
    ui.osc_after_cast_edit = ""

    ui._finish_osc_edit(commit=True)

    updated = controller.spellbook.spells[0]
    assert updated.osc_on_cast == (
        OscAction("Spell", 4),
        OscAction("MagicPrepared", True),
    )
    assert updated.osc_after_cast == ()


def test_desktop_ui_spoken_name_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osc_grimoire.desktop_ui import DesktopVoiceUi

    controller = _controller(tmp_path)
    spell = controller.spellbook.spells[0]
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
) -> GrimoireController:
    config = AppConfig(
        audio=AudioConfig(sample_rate=16000),
        gesture=gesture_config or GestureRecognitionConfig(),
    )
    return GrimoireController(
        data_dir,
        config=config,
        backend=_fake_backend(),
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
    )


def _tokenize(text: str) -> tuple[int, ...]:
    cleaned = "".join(c for c in text.casefold() if c.isalnum())
    if not cleaned:
        raise ValueError("empty")
    return tuple(ord(c) for c in cleaned)


def _text(tokens: tuple[int, ...]) -> str:
    return "".join(chr(t) for t in tokens).capitalize()


def _audio_for_text(text: str) -> FloatArray:
    return np.asarray(_tokenize(text), dtype=np.float32)


def _fake_backend() -> VoiceTemplateBackend:
    def distance_to_tokens(feature, token_ids: tuple[int, ...]) -> float:
        return (
            0.0
            if tuple(feature) == token_ids
            else float(abs(sum(feature) - sum(token_ids)) + 1)
        )

    def hypotheses(feature, _limit: int):
        return (TextHypothesis(_text(tuple(feature)), tuple(feature), -1.0),)

    return VoiceTemplateBackend(
        name="fake",
        extract_path=lambda _path, _config: _tokenize("alohomora"),
        extract_array=lambda audio, _config, _sample_rate: tuple(
            int(v) for v in np.asarray(audio).reshape(-1)
        ),
        distance=lambda _a, _b: 0.0,
        aggregate=lambda distances: float(np.median(distances)),
        tokenize_text=_tokenize,
        distance_to_tokens=distance_to_tokens,
        text_hypotheses=hypotheses,
    )


class _FakeRecorder:
    def __init__(self) -> None:
        self.begin_count = 0
        self.end_count = 0

    def begin_recording(self) -> None:
        self.begin_count += 1

    def end_recording(self) -> FloatArray:
        self.end_count += 1
        return _audio_for_text("Lumos")


class _FakeOutput:
    status_text = "OSC target: fake"

    def __init__(self) -> None:
        self.voice_recording: list[bool] = []
        self.gesture_drawing: list[bool] = []
        self.ui_enabled: list[bool] = []
        self.voice_enabled: list[bool] = []
        self.gesture_enabled: list[bool] = []
        self.enable_toggles: list[tuple[bool, bool, bool]] = []
        self.spell_pulses: list[str] = []
        self.fizzle_count = 0
        self.tick_count = 0

    def set_voice_recording(self, recording: bool) -> None:
        self.voice_recording.append(recording)

    def set_gesture_drawing(self, drawing: bool) -> None:
        self.gesture_drawing.append(drawing)

    def set_ui_enabled(self, enabled: bool) -> None:
        self.ui_enabled.append(enabled)

    def set_voice_enabled(self, enabled: bool) -> None:
        self.voice_enabled.append(enabled)

    def set_gesture_enabled(self, enabled: bool) -> None:
        self.gesture_enabled.append(enabled)

    def set_enable_toggles(
        self, *, ui_enabled: bool, gesture_enabled: bool, voice_enabled: bool
    ) -> None:
        self.enable_toggles.append((ui_enabled, gesture_enabled, voice_enabled))

    def pulse_spell(self, spell) -> None:
        self.spell_pulses.append(spell.name)

    def pulse_fizzle(self) -> None:
        self.fizzle_count += 1

    def tick(self, now=None) -> None:
        self.tick_count += 1


class _FakeInput:
    status_text = "OSC input: fake"

    def __init__(self) -> None:
        self.ui_enabled = True
        self.gesture_enabled = True
        self.voice_enabled = True

    def recent_messages(self) -> tuple[Any, ...]:
        return ()

    def set_enabled_state(
        self,
        *,
        ui_enabled: bool | None = None,
        gesture_enabled: bool | None = None,
        voice_enabled: bool | None = None,
    ) -> None:
        if ui_enabled is not None:
            self.ui_enabled = ui_enabled
        if gesture_enabled is not None:
            self.gesture_enabled = gesture_enabled
        if voice_enabled is not None:
            self.voice_enabled = voice_enabled

    def stop(self) -> None:
        pass


class _CountingBackend:
    def __init__(self) -> None:
        self.extract_array_calls = 0
        self.backend = VoiceTemplateBackend(
            name="counting",
            extract_path=lambda _path, _config: (),
            extract_array=self.extract_array,
            distance=lambda _a, _b: 0.0,
            aggregate=lambda distances: float(np.median(distances)),
            tokenize_text=_tokenize,
            distance_to_tokens=lambda _feature, _tokens: 0.0,
        )

    def extract_array(
        self, audio: FloatArray, _config: VoiceRecognitionConfig, _sample_rate: int
    ) -> tuple[int, ...]:
        self.extract_array_calls += 1
        return tuple(int(v) for v in np.asarray(audio).reshape(-1))


def _gesture_line() -> FloatArray:
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    return np.column_stack([x, np.zeros_like(x)]).astype(np.float32)


def _gesture_zigzag() -> FloatArray:
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    y = np.where(np.arange(12) % 2 == 0, 0.0, 0.4).astype(np.float32)
    return np.column_stack([x, y]).astype(np.float32)
