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
    EmbeddingPoint,
    VoiceTrainingController,
)
from .osc_input import OscInputService
from .osc_output import OscOutput
from .paths import default_data_dir
from .spellbook import Spell
from .voice_features import FloatArray

LOGGER = logging.getLogger(__name__)

PAGE_MAIN = 0
PAGE_DIAGNOSTICS = -1


class DesktopVoiceUi:
    def __init__(
        self,
        controller: VoiceTrainingController,
        *,
        overlay_mode: bool = False,
        surface_size: tuple[int, int] = (1000, 760),
    ) -> None:
        self.controller = controller
        self.overlay_mode = overlay_mode
        self.surface_size = surface_size
        self.selected_spell_id: str | None = (
            controller.spellbook.spells[0].id if controller.spellbook.spells else None
        )
        self.page = PAGE_MAIN
        self.edit_name = ""
        self.recording_mode: str | None = None
        self.recording_source: str | None = None
        self.recorder: NonBlockingAudioRecorder | None = None
        self.recorder_error: str | None = None
        self.waveform_cache: dict[tuple[str, int], FloatArray] = {}
        self._implot_context_created = False

    def draw(self) -> None:
        from imgui_bundle import imgui

        self.controller.tick_outputs()
        self._ensure_implot_context()
        window_flags = imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
        if self.overlay_mode:
            imgui.set_next_window_pos(imgui.ImVec2(0, 0))
            imgui.set_next_window_size(
                imgui.ImVec2(float(self.surface_size[0]), float(self.surface_size[1]))
            )
            window_flags |= (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_saved_settings
            )
        else:
            imgui.set_next_window_size(imgui.ImVec2(980, 720))
        _expanded, _open = imgui.begin(
            "OSC Grimoire",
            None,
            window_flags,
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
        if self.controller.output_status is not None:
            imgui.text(self.controller.output_status)
        if self.controller.input_status is not None:
            imgui.text(self.controller.input_status)
        if self.recorder_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.25, 1.0), self.recorder_error)
        imgui.end()
        if self.overlay_mode:
            self._draw_overlay_cursor()

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

        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##main_page_folio", 2, table_flags):
            return
        imgui.table_setup_column(
            "Recognition", imgui.TableColumnFlags_.width_stretch, 0.58
        )
        imgui.table_setup_column(
            "Visualization", imgui.TableColumnFlags_.width_stretch, 0.42
        )
        imgui.table_next_row()
        imgui.table_next_column()
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
        self._hold_button(
            "Hold to Recognize" if self.overlay_mode else "Hold to Recognize (Space)",
            "recognize",
            allow_space=not self.overlay_mode,
        )

        result = self.controller.last_result
        if result is not None:
            imgui.separator()
            imgui.text_unformatted(result.debug_text)
        gesture_result = self.controller.last_gesture_result
        if gesture_result is not None:
            imgui.separator()
            imgui.text_unformatted(gesture_result.debug_text)
        imgui.table_next_column()
        imgui.text("Embedding PCA")
        self._draw_embedding_plot()
        if self.controller.latest_gesture_points is not None:
            self._draw_gesture_preview(
                "Latest Gesture", self.controller.latest_gesture_points
            )
        imgui.end_table()

    def _draw_spell_page(self) -> None:
        from imgui_bundle import imgui

        spell = self._selected_spell()
        if spell is None and self.controller.draft is None:
            self.controller.start_draft()
            assert self.controller.draft is not None
            self.edit_name = self.controller.draft.name

        if spell is not None and not self.edit_name:
            self.edit_name = spell.name

        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##spell_page_folio", 2, table_flags):
            return
        imgui.table_setup_column("Spell", imgui.TableColumnFlags_.width_stretch, 0.58)
        imgui.table_setup_column(
            "Embedding", imgui.TableColumnFlags_.width_stretch, 0.42
        )
        imgui.table_next_row()
        imgui.table_next_column()
        self._draw_spell_controls(spell)
        spell = self._selected_spell()
        if spell is not None:
            self._draw_samples(spell)
        imgui.table_next_column()
        imgui.text("Embedding PCA")
        self._draw_embedding_plot()
        if spell is not None:
            self._draw_gesture_preview(
                "Saved Gesture", self.controller.gesture_preview(spell)
            )
        if self.controller.latest_gesture_points is not None:
            self._draw_gesture_preview(
                "Latest Gesture", self.controller.latest_gesture_points
            )
        imgui.end_table()

    def _draw_spell_controls(self, spell: Spell | None) -> None:
        from imgui_bundle import imgui

        if not self._can_edit_spell_names():
            imgui.text(f"Spell name: {self.edit_name}")
        else:
            changed, new_name = imgui.input_text("Spell name", self.edit_name)
            if changed:
                self.edit_name = new_name
                if self.controller.draft is not None:
                    self.controller.update_draft_name(new_name)

        if spell is None:
            if self._can_edit_spell_names():
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
            if self._can_edit_spell_names() and imgui.button("Save Name"):
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

        self._hold_button(
            "Hold to Record Sample"
            if self.overlay_mode
            else "Hold to Record Sample (Space)",
            "sample",
            allow_space=not self.overlay_mode,
        )
        imgui.same_line()
        self._hold_button("Hold to Test", "test", allow_space=False)
        if spell is not None:
            if imgui.button("Record / Replace Gesture", imgui.ImVec2(230, 42)):
                self.controller.arm_gesture_recording(spell.id)

    def _draw_samples(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Samples")
        previews = self._sample_previews(spell)
        available_width = imgui.get_content_region_avail().x
        columns = max(1, min(3, int(available_width // 190)))
        table_flags = (
            imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##sample_grid", columns, table_flags):
            return
        for index, relative_path in enumerate(spell.voice_samples):
            if index % columns == 0:
                imgui.table_next_row()
            imgui.table_next_column()
            imgui.push_id(str(index))
            if imgui.button("X"):
                self.controller.delete_sample(spell.id, relative_path)
                self.waveform_cache.clear()
                imgui.pop_id()
                break
            imgui.same_line()
            if index < len(previews):
                graph_width = max(110.0, imgui.get_content_region_avail().x - 6.0)
                imgui.plot_lines(
                    "##wave",
                    previews[index],
                    graph_size=imgui.ImVec2(graph_width, 34),
                    scale_min=-1.0,
                    scale_max=1.0,
                )
            imgui.pop_id()
        imgui.end_table()

    def _can_edit_spell_names(self) -> bool:
        return not self.overlay_mode

    def _draw_overlay_cursor(self) -> None:
        from imgui_bundle import imgui

        mouse_pos = imgui.get_mouse_pos()
        if mouse_pos.x < 0.0 or mouse_pos.y < 0.0:
            return
        if mouse_pos.x > self.surface_size[0] or mouse_pos.y > self.surface_size[1]:
            return
        draw_list = imgui.get_foreground_draw_list()
        center = imgui.ImVec2(mouse_pos.x, mouse_pos.y)
        outer = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.9))
        inner = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        draw_list.add_circle_filled(center, 7.0, outer, 16)
        draw_list.add_circle_filled(center, 4.0, inner, 16)

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
        imgui.text("Embedding PCA")
        self._draw_embedding_plot()
        gesture_result = self.controller.last_gesture_result
        if gesture_result is not None:
            imgui.separator()
            imgui.text_unformatted(gesture_result.debug_text)

    def _hold_button(self, label: str, mode: str, *, allow_space: bool) -> None:
        from imgui_bundle import imgui

        imgui.button(label, imgui.ImVec2(230, 42))
        button_held = imgui.is_item_active()
        space_held = allow_space and imgui.is_key_down(imgui.Key.space)
        held = button_held or space_held
        self._update_hold_recording(mode, held)

    def _update_hold_recording(self, mode: str, held: bool) -> None:
        if held and self.recording_mode is None:
            self._begin_recording(mode, "ui")
        elif not held and self.recording_mode == mode and self.recording_source == "ui":
            self._finish_recording(mode, "ui")

    def begin_overlay_voice_recording(self) -> None:
        if self.recording_mode is None:
            self._begin_recording("recognize", "overlay")

    def finish_overlay_voice_recording(self) -> None:
        if self.recording_mode == "recognize" and self.recording_source == "overlay":
            self._finish_recording("recognize", "overlay")

    def _begin_recording(self, mode: str, source: str) -> None:
        if (
            mode == "recognize" or mode == "test"
        ) and not self.controller.voice_enabled:
            self.controller.status = "Voice recognition disabled by OSC."
            return
        recorder = self._ensure_recorder()
        if recorder is None:
            return
        recorder.begin_recording()
        self.recording_mode = mode
        self.recording_source = source
        self.controller.set_voice_recording(True)
        self.controller.status = f"Recording {mode}..."

    def _finish_recording(self, mode: str, source: str) -> None:
        if self.recorder is None:
            return
        audio = self.recorder.end_recording()
        self.recording_mode = None
        self.recording_source = None
        self.controller.set_voice_recording(False)
        try:
            self._handle_recording(mode, audio)
        except ValueError as exc:
            if mode == "recognize" or mode == "test":
                self.controller.pulse_fizzle()
            self.controller.status = str(exc)
        except Exception as exc:
            LOGGER.exception("Recording action failed")
            if mode == "recognize" or mode == "test":
                self.controller.pulse_fizzle()
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

    def _draw_embedding_plot(self) -> None:
        from imgui_bundle import imgui, implot

        points = self.controller.embedding_points()
        if not points:
            imgui.text("Embedding PCA: no samples yet.")
            return

        imgui.text("Embedding PCA: spell samples are circles; latest attempt is X.")
        plot_flags = implot.Flags_.no_mouse_text
        available_width = imgui.get_content_region_avail().x
        plot_width = max(260.0, available_width)
        if implot.begin_plot(
            "##embedding_pca", imgui.ImVec2(plot_width, 260), plot_flags
        ):
            axis_flags = implot.AxisFlags_.no_decorations
            implot.setup_axes("", "", axis_flags, axis_flags)
            self._plot_sample_points(points)
            self._plot_query_points(points)
            implot.end_plot()

    def _draw_gesture_preview(
        self,
        label: str,
        points: FloatArray | None,
        size: tuple[float, float] = (260, 150),
    ) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text(label)
        draw_size = imgui.ImVec2(size[0], size[1])
        origin = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        bg = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.08, 0.07, 0.10, 1.0))
        border = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.35, 0.33, 0.42, 1.0))
        line = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.74, 0.88, 1.0, 1.0))
        draw_list.add_rect_filled(
            origin,
            imgui.ImVec2(origin.x + draw_size.x, origin.y + draw_size.y),
            bg,
        )
        draw_list.add_rect(
            origin,
            imgui.ImVec2(origin.x + draw_size.x, origin.y + draw_size.y),
            border,
        )
        array = (
            np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if points is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        if array.shape[0] >= 2:
            mapped = _map_gesture_points_to_rect(array, origin, draw_size)
            for start, end in zip(mapped[:-1], mapped[1:], strict=False):
                draw_list.add_line(start, end, line, 2.0)
        elif array.shape[0] == 0:
            text_pos = imgui.ImVec2(origin.x + 12.0, origin.y + draw_size.y * 0.45)
            text_color = imgui.color_convert_float4_to_u32(
                imgui.ImVec4(0.55, 0.55, 0.62, 1.0)
            )
            draw_list.add_text(text_pos, text_color, "(none)")
        imgui.dummy(imgui.ImVec2(draw_size.x, draw_size.y + 6.0))

    def _ensure_implot_context(self) -> None:
        if self._implot_context_created:
            return
        from imgui_bundle import implot

        if implot.get_current_context() is None:
            implot.create_context()
        self._implot_context_created = True

    def _plot_sample_points(self, points: tuple[EmbeddingPoint, ...]) -> None:
        from imgui_bundle import implot

        sample_groups = sorted({p.group for p in points if p.kind == "sample"})
        for index, group in enumerate(sample_groups):
            group_points = [
                p for p in points if p.kind == "sample" and p.group == group
            ]
            if not group_points:
                continue
            color = _palette_color(index, 1.0)
            spec = implot.Spec()
            spec.marker = implot.Marker_.circle
            spec.marker_size = 6.0
            spec.marker_fill_color = color
            spec.marker_line_color = color
            spec.fill_color = color
            spec.line_color = color
            spec.fill_alpha = 1.0
            spec.line_weight = 0.0
            xs = np.array([p.x for p in group_points], dtype=np.float64)
            ys = np.array([p.y for p in group_points], dtype=np.float64)
            implot.plot_scatter(group, xs, ys, spec)

    def _plot_query_points(self, points: tuple[EmbeddingPoint, ...]) -> None:
        from imgui_bundle import imgui, implot

        query_points = [p for p in points if p.kind == "query" and p.age == 0]
        for point in query_points:
            color = (
                imgui.ImVec4(0.25, 1.0, 0.45, 1.0)
                if point.accepted
                else imgui.ImVec4(1.0, 0.35, 0.25, 1.0)
            )
            xs_all = [p.x for p in points]
            ys_all = [p.y for p in points]
            span = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all), 1.0)
            delta = span * 0.025
            spec = implot.Spec()
            spec.line_color = color
            spec.line_weight = 2.0
            implot.plot_line(
                "latest attempt",
                np.array([point.x - delta, point.x + delta], dtype=np.float64),
                np.array([point.y - delta, point.y + delta], dtype=np.float64),
                spec,
            )
            implot.plot_line(
                "##latest_attempt_cross",
                np.array([point.x - delta, point.x + delta], dtype=np.float64),
                np.array([point.y + delta, point.y - delta], dtype=np.float64),
                spec,
            )

    def shutdown(self) -> None:
        if self.recorder is not None:
            self.recorder.stop_stream()
            self.recorder = None
        if self.controller.osc_input is not None:
            self.controller.osc_input.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    controller = VoiceTrainingController(data_dir)
    osc_output = OscOutput(controller.config.osc)
    controller.output = osc_output
    osc_input = OscInputService(controller.config.osc)
    osc_input.start()
    controller.osc_input = osc_input
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


def _palette_color(index: int, alpha: float):
    from imgui_bundle import imgui

    colors = (
        (0.32, 0.68, 1.00),
        (1.00, 0.62, 0.22),
        (0.67, 0.85, 0.28),
        (0.86, 0.44, 0.96),
        (0.21, 0.88, 0.76),
        (1.00, 0.42, 0.55),
    )
    r, g, b = colors[index % len(colors)]
    return imgui.ImVec4(r, g, b, alpha)


def _map_gesture_points_to_rect(points: np.ndarray, origin, size) -> list:
    from imgui_bundle import imgui

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = maximum - minimum
    scale = min(
        (size.x - 20.0) / max(float(span[0]), 1e-6),
        (size.y - 20.0) / max(float(span[1]), 1e-6),
    )
    centered = points - (minimum + maximum) * 0.5
    center = np.asarray([origin.x + size.x * 0.5, origin.y + size.y * 0.5])
    mapped = centered * scale
    return [
        imgui.ImVec2(float(center[0] + point[0]), float(center[1] - point[1]))
        for point in mapped
    ]


if __name__ == "__main__":
    sys.exit(main())
