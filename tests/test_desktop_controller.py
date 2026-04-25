from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from osc_grimoire.config import AppConfig, AudioConfig, VoiceRecognitionConfig
from osc_grimoire.desktop_controller import VoiceTrainingController
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


def _controller(data_dir: Path) -> VoiceTrainingController:
    config = AppConfig(audio=AudioConfig(sample_rate=16000))
    return VoiceTrainingController(
        data_dir,
        config=config,
        backend=_fake_backend(),
        voice_config=VoiceRecognitionConfig(relative_margin_min=0.0),
    )


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
        self.backend = VoiceTemplateBackend(
            name="counting",
            extract_path=lambda _path, _config: np.array([[0.0]], dtype=np.float32),
            extract_array=self.extract_array,
            distance=lambda a, b: float(abs(a[0, 0] - b[0, 0])),
            aggregate=lambda distances: float(np.median(distances)),
        )

    def extract_array(
        self, audio: FloatArray, _config: VoiceRecognitionConfig, _sample_rate: int
    ) -> FloatArray:
        self.extract_array_calls += 1
        return np.array([[float(audio.size)]], dtype=np.float32)


def _audio(frequency: float) -> FloatArray:
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
