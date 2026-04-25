from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .audio_capture import NonBlockingAudioRecorder
from .desktop_controller import (
    DEFAULT_SAMPLE_TARGET,
    VoiceTrainingController,
)
from .paths import default_data_dir
from .spellbook import Spell
from .voice_features import FloatArray

LOGGER = logging.getLogger(__name__)

PAGE_MAIN = 0
PAGE_DIAGNOSTICS = -1


class DesktopVoiceUi:
    def __init__(self, controller: VoiceTrainingController) -> None:
        self.controller = controller
        self.selected_spell_id: str | None = (
            controller.spellbook.spells[0].id if controller.spellbook.spells else None
        )
        self.page = PAGE_MAIN
        self.edit_name = ""
        self.recording_mode: str | None = None
        self.recorder: NonBlockingAudioRecorder | None = None
        self.recorder_error: str | None = None
        self.waveform_cache: dict[tuple[str, int], FloatArray] = {}

    def draw(self) -> None:
        from imgui_bundle import imgui

        imgui.set_next_window_size(imgui.ImVec2(980, 720))
        _expanded, _open = imgui.begin(
            "OSC Grimoire",
            None,
            imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse,
        )
        self._draw_nav()
        imgui.separator()
        if self.page == PAGE_MAIN:
            self._draw_main_page()
        elif self.page == PAGE_DIAGNOSTICS:
            self._draw_diagnostics_page()
        else:
            self._draw_spell_page()
        imgui.separator()
        imgui.text(self.controller.status)
        if self.recorder_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.25, 1.0), self.recorder_error)
        imgui.end()

    def _draw_nav(self) -> None:
        from imgui_bundle import imgui

        if imgui.button("Main"):
            self.page = PAGE_MAIN
            if self.controller.draft is not None:
                self.controller.cancel_draft()
        imgui.same_line()
        if imgui.button("Prev"):
            self._go_prev_page()
        imgui.same_line()
        if imgui.button("Next"):
            self._go_next_page()
        imgui.same_line()
        imgui.text(f"Page: {self._page_title()}")
        imgui.same_line()
        if imgui.button("Add Spell"):
            self.controller.start_draft()
            self.selected_spell_id = None
            self.edit_name = self.controller.draft.name if self.controller.draft else ""
            self.page = len(self.controller.spellbook.spells) + 1
        imgui.same_line()
        if imgui.button("Diagnostics"):
            self.page = PAGE_DIAGNOSTICS

    def _draw_main_page(self) -> None:
        from imgui_bundle import imgui

        imgui.text("Spells")
        imgui.text("Use Prev/Next to flip to each spell page.")
        for spell in self.controller.spellbook.spells:
            label = f"{spell.name} ({len(spell.voice_samples)} samples)"
            selected = spell.id == self.selected_spell_id
            clicked, _selected = imgui.selectable(label, selected)
            if clicked:
                self.selected_spell_id = spell.id
                self.edit_name = spell.name
                self.page = self._page_for_spell_id(spell.id)
        imgui.separator()
        self._hold_button("Hold to Recognize (Space)", "recognize", allow_space=True)

        result = self.controller.last_result
        if result is not None:
            imgui.separator()
            imgui.text_unformatted(result.debug_text)

    def _draw_spell_page(self) -> None:
        from imgui_bundle import imgui

        spell = self._selected_spell()
        if spell is None and self.controller.draft is None:
            self.controller.start_draft()
            assert self.controller.draft is not None
            self.edit_name = self.controller.draft.name

        if spell is not None and not self.edit_name:
            self.edit_name = spell.name

        changed, new_name = imgui.input_text("Spell name", self.edit_name)
        if changed:
            self.edit_name = new_name
            if self.controller.draft is not None:
                self.controller.update_draft_name(new_name)

        if spell is None:
            if imgui.button("Save Empty Spell"):
                spell = self.controller.persist_draft()
                self.selected_spell_id = spell.id
                self.edit_name = spell.name
            imgui.same_line()
            if imgui.button("Cancel Draft"):
                self.controller.cancel_draft()
                self.selected_spell_id = None
                self.edit_name = ""
                self.page = PAGE_MAIN
        else:
            if imgui.button("Save Name"):
                spell = self.controller.rename_spell(spell.id, self.edit_name)
                self.selected_spell_id = spell.id
                self.edit_name = spell.name

        spell = self._selected_spell()
        sample_count = len(spell.voice_samples) if spell is not None else 0
        imgui.progress_bar(
            min(sample_count / DEFAULT_SAMPLE_TARGET, 1.0),
            imgui.ImVec2(360, 0),
            f"{sample_count}/{DEFAULT_SAMPLE_TARGET} samples",
        )

        self._hold_button("Hold to Record Sample (Space)", "sample", allow_space=True)
        imgui.same_line()
        self._hold_button("Hold to Test", "test", allow_space=False)

        if spell is not None:
            self._draw_samples(spell)

    def _draw_samples(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Samples")
        previews = self._sample_previews(spell)
        for index, relative_path in enumerate(spell.voice_samples):
            imgui.push_id(str(index))
            if imgui.button("X"):
                self.controller.delete_sample(spell.id, relative_path)
                self.waveform_cache.clear()
                imgui.pop_id()
                break
            imgui.same_line()
            imgui.text(Path(relative_path).name)
            imgui.same_line()
            if index < len(previews):
                imgui.plot_lines(
                    "##wave",
                    previews[index],
                    graph_size=imgui.ImVec2(420, 48),
                    scale_min=-1.0,
                    scale_max=1.0,
                )
            imgui.pop_id()

    def _draw_diagnostics_page(self) -> None:
        from imgui_bundle import imgui

        imgui.text(f"Backend: {self.controller.backend.name}")
        imgui.text(
            "relative_margin_min: "
            f"{self.controller.voice_config.relative_margin_min:.2f}"
        )
        imgui.separator()
        imgui.text("Last recognition")
        result = self.controller.last_result
        imgui.text_unformatted(result.debug_text if result is not None else "(none)")
        imgui.separator()
        imgui.text("Latent-space visualization placeholder")
        imgui.text("Future: PCA/UMAP scatter of samples, attempts, and negatives.")

    def _hold_button(self, label: str, mode: str, *, allow_space: bool) -> None:
        from imgui_bundle import imgui

        imgui.button(label, imgui.ImVec2(230, 42))
        button_held = imgui.is_item_active()
        space_held = allow_space and imgui.is_key_down(imgui.Key.space)
        held = button_held or space_held
        if held and self.recording_mode is None:
            self._begin_recording(mode)
        elif not held and self.recording_mode == mode:
            self._finish_recording(mode)

    def _begin_recording(self, mode: str) -> None:
        recorder = self._ensure_recorder()
        if recorder is None:
            return
        recorder.begin_recording()
        self.recording_mode = mode
        self.controller.status = f"Recording {mode}..."

    def _finish_recording(self, mode: str) -> None:
        if self.recorder is None:
            return
        audio = self.recorder.end_recording()
        self.recording_mode = None
        try:
            self._handle_recording(mode, audio)
        except Exception as exc:
            LOGGER.exception("Recording action failed")
            self.controller.status = str(exc)

    def _handle_recording(self, mode: str, audio: FloatArray) -> None:
        if mode == "recognize" or mode == "test":
            self.controller.recognize(audio)
            return
        if mode == "sample":
            spell = self._selected_spell()
            if spell is None:
                spell = self.controller.add_sample_to_draft(audio)
                self.selected_spell_id = spell.id
                self.edit_name = spell.name
                self.page = self._page_for_spell_id(spell.id)
            else:
                self.controller.add_sample_to_spell(spell.id, audio)
            self.waveform_cache.clear()

    def _ensure_recorder(self) -> NonBlockingAudioRecorder | None:
        if self.recorder is not None:
            return self.recorder
        try:
            self.recorder = NonBlockingAudioRecorder(self.controller.config.audio)
            self.recorder.start_stream()
            self.recorder_error = None
            return self.recorder
        except Exception as exc:
            LOGGER.exception("Could not start audio recorder")
            self.recorder_error = f"Audio recorder failed: {exc}"
            self.recorder = None
            return None

    def _selected_spell(self) -> Spell | None:
        if self.selected_spell_id is None:
            return None
        for spell in self.controller.spellbook.spells:
            if spell.id == self.selected_spell_id:
                return spell
        self.selected_spell_id = None
        return None

    def _go_prev_page(self) -> None:
        pages = self._ordered_pages()
        index = pages.index(self.page) if self.page in pages else 0
        self.page = pages[(index - 1) % len(pages)]
        self._sync_selection_to_page()

    def _go_next_page(self) -> None:
        pages = self._ordered_pages()
        index = pages.index(self.page) if self.page in pages else 0
        self.page = pages[(index + 1) % len(pages)]
        self._sync_selection_to_page()

    def _ordered_pages(self) -> list[int]:
        spell_pages = list(range(1, len(self.controller.spellbook.spells) + 1))
        pages = [PAGE_MAIN, *spell_pages]
        if self.controller.draft is not None:
            pages.append(len(self.controller.spellbook.spells) + 1)
        pages.append(PAGE_DIAGNOSTICS)
        return pages

    def _sync_selection_to_page(self) -> None:
        if self.page <= PAGE_MAIN:
            return
        index = self.page - 1
        if 0 <= index < len(self.controller.spellbook.spells):
            spell = self.controller.spellbook.spells[index]
            self.selected_spell_id = spell.id
            self.edit_name = spell.name

    def _page_title(self) -> str:
        if self.page == PAGE_MAIN:
            return "Main / Recognize"
        if self.page == PAGE_DIAGNOSTICS:
            return "Diagnostics"
        spell = self._selected_spell()
        if spell is not None:
            return spell.name
        if self.controller.draft is not None:
            return f"{self.controller.draft.name} (draft)"
        return "New Spell"

    def _page_for_spell_id(self, spell_id: str) -> int:
        for index, spell in enumerate(self.controller.spellbook.spells, start=1):
            if spell.id == spell_id:
                return index
        return PAGE_MAIN

    def _sample_previews(self, spell: Spell) -> list[FloatArray]:
        previews: list[FloatArray] = []
        for relative_path in spell.voice_samples:
            cache_key = (relative_path, 160)
            if cache_key not in self.waveform_cache:
                path = self.controller.data_dir / relative_path
                if path.exists():
                    self.waveform_cache[cache_key] = self.controller.sample_previews(
                        Spell(
                            id=spell.id,
                            name=spell.name,
                            voice_samples=(relative_path,),
                        )
                    )[0]
                else:
                    self.waveform_cache[cache_key] = np.zeros(160, dtype=np.float32)
            previews.append(self.waveform_cache[cache_key])
        return previews

    def shutdown(self) -> None:
        if self.recorder is not None:
            self.recorder.stop_stream()
            self.recorder = None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    controller = VoiceTrainingController(data_dir)
    controller.status = "Loading Whisper model..."
    controller.preload_backend()
    controller.status = "Ready."
    app = DesktopVoiceUi(controller)
    try:
        _run_imgui(app)
    finally:
        app.shutdown()
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSC Grimoire desktop voice UI")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _run_imgui(app: DesktopVoiceUi) -> None:
    from imgui_bundle import hello_imgui, immapp

    params = hello_imgui.SimpleRunnerParams()
    params.window_title = "OSC Grimoire"
    params.window_size = (1000, 760)
    params.gui_function = app.draw
    immapp.run(params)


if __name__ == "__main__":
    sys.exit(main())
