from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_capture import NonBlockingAudioRecorder
from .desktop_controller import (
    DEFAULT_SAMPLE_TARGET,
    GestureResult,
    RecognitionResult,
    VoiceTrainingController,
)
from .gesture_recognizer import GestureRanking
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
        self.keyboard_request_handler: Callable[[str | None, str, str], bool] | None = (
            None
        )
        self.keyboard_close_handler: Callable[[], None] | None = None
        self.bindings_request_handler: Callable[[], bool] | None = None
        self.overlay_start_handler: Callable[[], bool] | None = None
        self.keyboard_edit_spell_id: str | None = None
        self.keyboard_editing = False
        self.keyboard_focus_pending = False
        self.keyboard_original_name = ""
        self.osc_edit_spell_id: str | None = None
        self.osc_editing = False
        self.osc_edit_value = ""
        self.osc_original_value = ""
        self.osc_focus_pending = False
        self.pending_spoken_name: str | None = None
        self.pending_spoken_name_spell_id: str | None = None
        self.delete_confirm_spell_id: str | None = None
        self.delete_confirm_ready_at = 0.0
        self.delete_confirm_expires_at = 0.0
        self.suppress_add_spell_until_mouse_up = False
        self._last_logged_input_status: str | None = None
        self._last_logged_output_status: str | None = None
        self._implot_context_created = False

    def draw(self) -> None:
        from imgui_bundle import imgui

        self.controller.tick_outputs()
        self._log_status_changes()
        self._ensure_implot_context()
        window_flags = imgui.WindowFlags_.no_collapse
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
            display_size = imgui.get_io().display_size
            imgui.set_next_window_pos(imgui.ImVec2(0, 0))
            imgui.set_next_window_size(display_size)
            window_flags |= (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_saved_settings
            )
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
            self._draw_settings_page()
        else:
            self._draw_spell_page()
        imgui.separator()
        self._draw_bottom_log()
        if self.recorder_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.25, 1.0), self.recorder_error)
        imgui.end()
        if self.overlay_mode:
            self._draw_overlay_cursor()

    def _draw_nav(self) -> None:
        from imgui_bundle import imgui

        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##top_bar", 4, table_flags):
            return
        navigation_width = (
            430
            if not self.overlay_mode and self.overlay_start_handler is not None
            else 320
        )
        imgui.table_setup_column(
            "Navigation", imgui.TableColumnFlags_.width_fixed, navigation_width
        )
        imgui.table_setup_column("Page", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("Activity", imgui.TableColumnFlags_.width_fixed, 210)
        status_width = 320 if self.overlay_mode else 430
        imgui.table_setup_column(
            "Status", imgui.TableColumnFlags_.width_fixed, status_width
        )
        imgui.table_next_row()
        imgui.table_next_column()
        if imgui.button("Main"):
            self.page = PAGE_MAIN
            if self.controller.draft is not None:
                self.controller.cancel_draft()
                self.suppress_add_spell_until_mouse_up = True
        imgui.same_line()
        if imgui.button("Settings"):
            self.page = PAGE_DIAGNOSTICS
        imgui.same_line()
        if imgui.button("Prev"):
            self._go_prev_page()
        imgui.same_line()
        if imgui.button("Next"):
            self._go_next_page()
        if not self.overlay_mode and self.overlay_start_handler is not None:
            imgui.same_line()
            if imgui.button("Start Overlay"):
                self._request_overlay_start()
        imgui.table_next_column()
        imgui.text(f"Page: {self._page_title()}")
        imgui.table_next_column()
        self._draw_centered_activity_status()
        imgui.table_next_column()
        self._draw_top_status()
        imgui.end_table()

    def _draw_main_page(self) -> None:
        from imgui_bundle import imgui

        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##main_page_folio", 2, table_flags):
            return
        recognition_width = 0.64 if self.overlay_mode else 0.72
        visualization_width = 1.0 - recognition_width
        imgui.table_setup_column(
            "Recognition", imgui.TableColumnFlags_.width_stretch, recognition_width
        )
        imgui.table_setup_column(
            "Visualization", imgui.TableColumnFlags_.width_stretch, visualization_width
        )
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("Spells")
        if not self.overlay_mode:
            self._hold_button(
                "Hold to Recognize",
                "recognize",
                allow_space=False,
                size=(230, 36),
            )
        if not self.controller.spellbook.spells:
            self._draw_main_empty_state()
        self._draw_spell_summary_table()
        imgui.table_next_column()
        self._draw_latest_score_panel()
        if self.controller.latest_gesture_points is not None:
            self._draw_gesture_preview(
                "Latest Gesture", self.controller.latest_gesture_points
            )
        imgui.end_table()

    def _draw_main_empty_state(self) -> None:
        from imgui_bundle import imgui

        imgui.text("No spells are trained yet.")
        imgui.text_disabled("Add a spell, then train:")
        imgui.bullet_text("Voice-only: record spoken samples.")
        imgui.bullet_text("Gesture-only: record a wand motion.")
        imgui.bullet_text("Both: train voice and gesture.")

    def _draw_spell_summary_table(self) -> None:
        from imgui_bundle import imgui

        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.no_saved_settings
        )
        if not imgui.begin_table("##spell_summary", 3, table_flags):
            return
        imgui.table_setup_column("Gesture", imgui.TableColumnFlags_.width_fixed, 140)
        imgui.table_setup_column("Spell", imgui.TableColumnFlags_.width_stretch)
        osc_width = 245 if self.overlay_mode else 300
        imgui.table_setup_column("OSC", imgui.TableColumnFlags_.width_fixed, osc_width)
        for spell in self.controller.spellbook.spells:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.push_id(spell.id)
            self._draw_gesture_canvas(
                self.controller.gesture_preview(spell), size=(76, 42)
            )
            if imgui.is_item_clicked():
                self._open_spell_page(spell)
            imgui.same_line()
            if imgui.button("Play"):
                self._play_random_sample(spell)
            imgui.table_next_column()
            clicked, _selected = imgui.selectable(
                f"{spell.name}##row",
                spell.id == self.selected_spell_id,
            )
            if clicked:
                self._open_spell_page(spell)
            self._draw_spell_row_match(spell)
            imgui.table_next_column()
            osc_parameter = self.controller.spell_osc_parameter_name(spell)
            if not self.overlay_mode:
                self._draw_copy_button(osc_parameter, tooltip="OSC parameter")
                imgui.same_line()
            osc_label = f"osc: {osc_parameter}"
            imgui.text_disabled(
                _left_elide_text_to_width(
                    osc_label,
                    imgui.get_content_region_avail().x,
                )
            )
            imgui.pop_id()
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_disabled("+")
        imgui.table_next_column()
        if self.suppress_add_spell_until_mouse_up:
            if not imgui.is_mouse_down(imgui.MouseButton_.left):
                self.suppress_add_spell_until_mouse_up = False
            imgui.begin_disabled()
            imgui.button("Add Spell")
            imgui.end_disabled()
        elif imgui.button("Add Spell"):
            self._start_add_spell()
        imgui.table_next_column()
        imgui.end_table()

    def _draw_spell_row_match(self, spell: Spell) -> None:
        latest_voice = self.controller.last_result
        latest_gesture = self.controller.last_gesture_result
        if self.controller.last_match_kind == "voice" and latest_voice is not None:
            row = _voice_ranking_for_spell(latest_voice, spell.id)
            if row is not None:
                score = _voice_match_score(latest_voice, row)
                conflict = _voice_margin_conflict(latest_voice, spell.id)
                state = (
                    "conflict"
                    if conflict
                    else "normal"
                    if latest_voice.decision.accepted
                    else "rejected"
                )
                if latest_voice.decision.accepted:
                    state = "accepted" if row == latest_voice.ranking[0] else "muted"
                self._draw_threshold_bar(
                    score,
                    0.0,
                    label="voice",
                    show_marker=False,
                    state=state,
                    size=(260, 14),
                )
            return
        if self.controller.last_match_kind == "gesture" and latest_gesture is not None:
            row = _gesture_ranking_for_spell(latest_gesture, spell.id)
            if row is not None:
                conflict = _gesture_margin_conflict(
                    latest_gesture, spell.id, self.controller.config.gesture.margin_min
                )
                state = (
                    "conflict"
                    if conflict
                    else "normal"
                    if latest_gesture.decision.accepted
                    else "rejected"
                )
                if latest_gesture.decision.accepted:
                    state = (
                        "accepted"
                        if latest_gesture.decision.best_spell_id == spell.id
                        else "muted"
                    )
                self._draw_threshold_bar(
                    row.score,
                    self.controller.config.gesture.score_min,
                    label="gesture",
                    show_marker=False,
                    state=state,
                    size=(260, 14),
                )

    def _draw_spell_page(self) -> None:
        from imgui_bundle import imgui

        spell = self._selected_spell()
        if spell is None and self.controller.draft is None:
            self.page = PAGE_MAIN
            return

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
        imgui.table_setup_column("Gesture", imgui.TableColumnFlags_.width_stretch, 0.42)
        imgui.table_next_row()
        imgui.table_next_column()
        self._draw_spell_controls(spell)
        spell = self._selected_spell()
        if spell is not None:
            self._draw_samples(spell)
        elif self.controller.draft is not None:
            self._draw_draft_sample_controls()
        imgui.table_next_column()
        if spell is not None:
            self._draw_saved_gesture_section(spell)
        elif self.controller.draft is not None:
            self._draw_draft_gesture_guidance()
        if self.controller.latest_gesture_points is not None:
            self._draw_gesture_preview(
                "Latest Gesture", self.controller.latest_gesture_points
            )
        imgui.end_table()

    def _draw_spell_controls(self, spell: Spell | None) -> None:
        from imgui_bundle import imgui

        if not self._can_edit_spell_names():
            imgui.text(f"Spell name: {self.edit_name}")
            self._draw_overlay_rename_controls(spell)
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
                self.suppress_add_spell_until_mouse_up = True
        else:
            if self._can_edit_spell_names() and imgui.button("Save Name"):
                spell = self.controller.rename_spell(spell.id, self.edit_name)
                self.selected_spell_id = spell.id
                self.edit_name = spell.name
            self._draw_delete_spell_controls(spell)

        spell = self._selected_spell()
        sample_count = len(spell.voice_samples) if spell is not None else 0
        imgui.progress_bar(
            min(sample_count / DEFAULT_SAMPLE_TARGET, 1.0),
            imgui.ImVec2(360, 0),
            f"{sample_count}/{DEFAULT_SAMPLE_TARGET} samples",
        )
        if spell is not None:
            self._draw_spell_osc_controls(spell)

    def _draw_overlay_rename_controls(self, spell: Spell | None) -> None:
        from imgui_bundle import imgui

        target_spell_id = spell.id if spell is not None else None
        if self.keyboard_editing and self.keyboard_edit_spell_id == target_spell_id:
            if self.keyboard_focus_pending:
                imgui.set_keyboard_focus_here()
                self.keyboard_focus_pending = False
            _submitted, new_name = imgui.input_text(
                "Spell name",
                self.edit_name,
                imgui.InputTextFlags_.enter_returns_true,
            )
            self.edit_name = new_name
            if imgui.button("Rename"):
                self.finish_keyboard_name(commit=True)
            imgui.same_line()
            if imgui.button("Cancel"):
                self.finish_keyboard_name(commit=False)
            imgui.same_line()
            if imgui.button("Show Keyboard"):
                self._request_keyboard_rename(target_spell_id)
            if imgui.get_content_region_avail().x < 140:
                pass
            else:
                imgui.same_line()
            self._hold_button(
                "Speak Name",
                "name",
                allow_space=False,
                size=(120, 0),
            )
        else:
            imgui.same_line()
            if imgui.button("Edit"):
                self._request_keyboard_rename(target_spell_id)
        if (
            self.pending_spoken_name is None
            or self.pending_spoken_name_spell_id != target_spell_id
        ):
            return
        imgui.text(f"Heard: {self.pending_spoken_name}")
        if imgui.button("Use Spoken Name"):
            self.edit_name = self.pending_spoken_name
            if not self.keyboard_editing:
                self._apply_spell_name(target_spell_id, self.pending_spoken_name)
            self.pending_spoken_name = None
            self.pending_spoken_name_spell_id = None
        imgui.same_line()
        if imgui.button("Discard"):
            self.pending_spoken_name = None
            self.pending_spoken_name_spell_id = None
            self.controller.status = "Spoken name discarded."

    def _request_keyboard_rename(self, target_spell_id: str | None) -> None:
        if self.keyboard_request_handler is None:
            self.controller.status = "SteamVR keyboard is unavailable."
            return
        if self.keyboard_request_handler(target_spell_id, self.edit_name, "Spell name"):
            self.keyboard_edit_spell_id = target_spell_id
            self.keyboard_editing = True
            self.keyboard_focus_pending = True
            self.keyboard_original_name = self.edit_name
            self.controller.status = "SteamVR keyboard opened."
        else:
            self.controller.status = "Could not open SteamVR keyboard."

    def finish_keyboard_name(self, *, commit: bool) -> None:
        if not self.keyboard_editing:
            return
        target_spell_id = self.keyboard_edit_spell_id
        self.keyboard_editing = False
        self.keyboard_edit_spell_id = None
        self.keyboard_focus_pending = False
        if self.keyboard_close_handler is not None:
            self.keyboard_close_handler()
        if not commit:
            self.edit_name = self.keyboard_original_name
            self.keyboard_original_name = ""
            self.cancel_keyboard_name()
            return
        self._apply_spell_name(target_spell_id, self.edit_name)
        self.keyboard_original_name = ""

    def cancel_keyboard_name(self) -> None:
        self.keyboard_editing = False
        self.keyboard_edit_spell_id = None
        self.keyboard_focus_pending = False
        if self.keyboard_original_name:
            self.edit_name = self.keyboard_original_name
            self.keyboard_original_name = ""
        self.controller.status = "Keyboard rename cancelled."

    def _apply_spell_name(self, target_spell_id: str | None, name: str) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Spell name cannot be empty")
        if target_spell_id is None:
            self.controller.update_draft_name(clean_name)
            self.edit_name = clean_name
            self.controller.status = f"Draft renamed to {clean_name}."
            return
        spell = self.controller.rename_spell(target_spell_id, clean_name)
        self.selected_spell_id = spell.id
        self.edit_name = spell.name

    def _draw_spell_osc_controls(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        osc_parameter = self.controller.spell_osc_parameter_name(spell)
        imgui.text(f"OSC parameter: {osc_parameter}")
        if not self.overlay_mode:
            imgui.same_line()
            self._draw_copy_button(osc_parameter, tooltip="OSC parameter")
        if self.osc_editing and self.osc_edit_spell_id == spell.id:
            if self.osc_focus_pending:
                imgui.set_keyboard_focus_here()
                self.osc_focus_pending = False
            _changed, value = imgui.input_text("OSC parameter", self.osc_edit_value)
            self.osc_edit_value = value
            if imgui.button("Save OSC"):
                self._finish_osc_edit(commit=True)
            imgui.same_line()
            if imgui.button("Cancel OSC"):
                self._finish_osc_edit(commit=False)
            imgui.same_line()
            if imgui.button("Show Keyboard"):
                self._request_osc_keyboard(spell)
            imgui.same_line()
            if imgui.button("Use Default"):
                self.osc_edit_value = ""
                self._finish_osc_edit(commit=True)
            imgui.text_disabled(
                "Leave blank to use the default derived from the spell name."
            )
            return
        if imgui.button("Edit OSC"):
            self.osc_edit_spell_id = spell.id
            self.osc_editing = True
            self.osc_edit_value = (
                spell.osc_address or self.controller.spell_osc_parameter_name(spell)
            )
            self.osc_original_value = spell.osc_address or ""
            self.osc_focus_pending = True

    def _request_osc_keyboard(self, spell: Spell) -> None:
        if self.keyboard_request_handler is None:
            self.controller.status = "SteamVR keyboard is unavailable."
            return
        if self.keyboard_request_handler(
            spell.id, self.osc_edit_value, "OSC parameter"
        ):
            self.osc_focus_pending = True
            self.controller.status = "SteamVR keyboard opened."
        else:
            self.controller.status = "Could not open SteamVR keyboard."

    def _finish_osc_edit(self, *, commit: bool) -> None:
        spell_id = self.osc_edit_spell_id
        if self.keyboard_close_handler is not None:
            self.keyboard_close_handler()
        self.osc_editing = False
        self.osc_edit_spell_id = None
        self.osc_focus_pending = False
        if spell_id is None:
            return
        if not commit:
            self.osc_edit_value = self.osc_original_value
            self.osc_original_value = ""
            self.controller.status = "OSC parameter edit cancelled."
            return
        spell = self.controller.update_spell_osc_address(spell_id, self.osc_edit_value)
        self.selected_spell_id = spell.id
        self.osc_original_value = ""

    def _draw_samples(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Samples")
        imgui.same_line()
        self._hold_button(
            "Record Sample",
            "sample",
            allow_space=False,
            size=(150, 0),
        )
        if not self.overlay_mode:
            imgui.same_line()
            self._hold_button(
                "Hold to Recognize",
                "test",
                allow_space=False,
                size=(170, 0),
            )
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
            if imgui.button("Play"):
                self._play_sample(relative_path)
            imgui.same_line()
            if imgui.button("X"):
                self.controller.delete_sample(spell.id, relative_path)
                self.waveform_cache.clear()
                imgui.pop_id()
                break
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

    def _play_sample(self, relative_path: str) -> None:
        try:
            self.controller.play_sample(relative_path)
        except Exception as exc:
            LOGGER.exception("Audio playback failed")
            self.controller.status = f"Playback failed: {exc}"

    def _play_random_sample(self, spell: Spell) -> None:
        try:
            self.controller.play_random_sample(spell.id)
        except Exception as exc:
            LOGGER.exception("Audio playback failed")
            self.controller.status = f"Playback failed: {exc}"

    def _draw_draft_sample_controls(self) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Samples")
        imgui.same_line()
        self._hold_button(
            "Record First Sample",
            "sample",
            allow_space=False,
            size=(190, 0),
        )
        imgui.text_wrapped(
            "Record a voice sample to create a voice spell. Aim for a normal, "
            "repeatable pronunciation; you can add more samples after it is created."
        )
        imgui.bullet_text("Use 5-10 samples if recognition feels unreliable.")
        imgui.bullet_text(
            "Vary pace and volume a little, but keep the same incantation."
        )

    def _draw_draft_gesture_guidance(self) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Gesture")
        if imgui.button("Record First Gesture"):
            self.controller.armed_gesture_spell_id = "__draft__"
            self.controller.status = (
                "Armed gesture recording for draft. Hold grip and draw."
            )
        imgui.text_wrapped(
            "Record a gesture to create a gesture spell. One clear, repeatable "
            "stroke is enough to start; you can add voice samples later if wanted."
        )

    def _draw_saved_gesture_section(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text("Saved Gesture")
        imgui.same_line()
        if imgui.button("Record Gesture"):
            self.controller.arm_gesture_recording(spell.id)
        imgui.same_line()
        if imgui.button("Clear Gesture"):
            self.controller.clear_gesture_sample(spell.id)
        self._draw_gesture_canvas(self.controller.gesture_preview(spell))

    def _draw_delete_spell_controls(self, spell: Spell) -> None:
        from imgui_bundle import imgui

        now = time.monotonic()
        if (
            self.delete_confirm_spell_id != spell.id
            or now > self.delete_confirm_expires_at
        ):
            if imgui.button("Delete Spell"):
                self.delete_confirm_spell_id = spell.id
                self.delete_confirm_ready_at = now + 1.5
                self.delete_confirm_expires_at = now + 7.0
                self.controller.status = "Delete armed. Wait, then confirm."
            return

        remaining = max(0.0, self.delete_confirm_ready_at - now)
        if remaining > 0.0:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.72, 0.25, 1.0),
                f"Confirm delete in {remaining:.1f}s",
            )
            return

        imgui.text_colored(
            imgui.ImVec4(1.0, 0.45, 0.35, 1.0),
            f"Delete {spell.name}?",
        )
        if imgui.button("Confirm Delete"):
            self.controller.delete_spell(spell.id)
            self.delete_confirm_spell_id = None
            self.selected_spell_id = (
                self.controller.spellbook.spells[0].id
                if self.controller.spellbook.spells
                else None
            )
            self.edit_name = ""
            self.page = PAGE_MAIN
            self.waveform_cache.clear()
        imgui.same_line()
        if imgui.button("Cancel Delete"):
            self.delete_confirm_spell_id = None
            self.controller.status = "Delete cancelled."

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

    def _draw_settings_page(self) -> None:
        from imgui_bundle import imgui

        imgui.text("Settings")
        imgui.separator()
        imgui.text("Casting hand")
        imgui.text_disabled("The spellbook appears on the opposite hand.")
        current_hand = self.controller.config.openvr.pointer_hand
        if imgui.radio_button("Left", current_hand == "left"):
            self.controller.set_casting_hand("left")
        imgui.same_line()
        if imgui.radio_button("Right", current_hand == "right"):
            self.controller.set_casting_hand("right")

        imgui.separator()
        imgui.text("Controller bindings")
        imgui.text_disabled("Default bindings:")
        imgui.bullet_text("Voice: hold trigger on the casting hand.")
        imgui.bullet_text("Gesture: hold grip on the casting hand.")
        imgui.bullet_text("Show/hide spellbook: hold both B buttons.")
        imgui.text_wrapped(
            "Use SteamVR bindings to change buttons or inspect the active controller profile."
        )
        if imgui.button("Change Bindings"):
            self._request_binding_settings()

        imgui.separator()
        imgui.text("Recognition tuning")
        imgui.text_disabled(
            "Move left to accept more attempts; move right to reject uncertain matches."
        )
        self._draw_strictness_slider(
            "Voice",
            self.controller.voice_strictness,
            self.controller.set_voice_strictness,
        )
        self._draw_strictness_slider(
            "Gesture",
            self.controller.gesture_strictness,
            self.controller.set_gesture_strictness,
        )

        imgui.separator()
        imgui.text("OSC")
        prefix = self.controller.config.osc.parameter_prefix
        imgui.text_disabled("Outputs sent to VRChat:")
        self._draw_osc_help_parameter(
            f"{prefix}VoiceRecording", "true while voice is recording."
        )
        self._draw_osc_help_parameter(
            f"{prefix}GestureDrawing", "true while gesture is drawing."
        )
        self._draw_osc_help_parameter(
            f"{prefix}Fizzle", "brief pulse when recognition rejects."
        )
        self._draw_osc_help_parameter(
            f"{prefix}Spell<Name>",
            "brief pulse when a spell is accepted; customizable on each spell page.",
        )
        imgui.text_disabled("Inputs accepted from VRChat:")
        self._draw_osc_help_parameter(
            f"{prefix}UIEnabled", "show or hide the spellbook."
        )
        self._draw_osc_help_parameter(
            f"{prefix}GestureEnabled", "enable or disable gesture input."
        )
        self._draw_osc_help_parameter(
            f"{prefix}VoiceEnabled", "enable or disable voice input."
        )

        imgui.separator()
        if imgui.collapsing_header("Diagnostics"):
            self._draw_diagnostics_details()

    def _draw_strictness_slider(
        self, label: str, value: float, setter: Callable[[float], None]
    ) -> None:
        from imgui_bundle import imgui

        imgui.text(label)
        imgui.text("Lenient")
        imgui.same_line()
        slider_width = max(220.0, imgui.get_content_region_avail().x - 58.0)
        imgui.set_next_item_width(slider_width)
        changed, value = imgui.slider_float(
            f"##{label.lower()}_strictness",
            value,
            0.0,
            1.0,
            "",
        )
        if changed:
            setter(value)
        imgui.same_line()
        imgui.text("Strict")

    def _draw_osc_help_parameter(self, parameter: str, description: str) -> None:
        from imgui_bundle import imgui

        imgui.bullet_text(f"{parameter}: {description}")
        if self.overlay_mode:
            return
        imgui.same_line()
        imgui.push_id(f"copy_{parameter}")
        self._draw_copy_button(parameter, tooltip="OSC parameter")
        imgui.pop_id()

    def _request_binding_settings(self) -> None:
        if self.bindings_request_handler is None:
            self.controller.status = "SteamVR binding UI is unavailable."
            return
        if self.bindings_request_handler():
            self.controller.status = "Opening SteamVR bindings..."
        else:
            self.controller.status = "Could not open SteamVR bindings."

    def _request_overlay_start(self) -> None:
        if self.overlay_start_handler is None:
            self.controller.status = "Overlay startup is unavailable."
            return
        if self.overlay_start_handler():
            self.controller.status = "Starting SteamVR overlay..."
        else:
            self.controller.status = "Could not start SteamVR overlay."

    def _draw_diagnostics_details(self) -> None:
        from imgui_bundle import imgui

        imgui.text("Diagnostics")
        imgui.text(f"Backend: {self.controller.backend.name}")
        imgui.text(
            "relative_margin_min: "
            f"{self.controller.voice_config.relative_margin_min:.2f}"
        )
        imgui.text(
            "gesture score/margin: "
            f"{self.controller.config.gesture.score_min:.2f}/"
            f"{self.controller.config.gesture.margin_min:.2f}"
        )
        if self.controller.output_status is not None:
            imgui.text(self.controller.output_status)
        if self.controller.input_status is not None:
            imgui.text(self.controller.input_status)
        imgui.separator()
        imgui.text("Last recognition")
        result = self.controller.last_result
        imgui.text_unformatted(result.debug_text if result is not None else "(none)")
        gesture_result = self.controller.last_gesture_result
        if gesture_result is not None:
            imgui.separator()
            imgui.text_unformatted(gesture_result.debug_text)

    def _draw_latest_score_panel(self) -> None:
        from imgui_bundle import imgui

        imgui.text("Latest Match Scores")
        if (
            self.controller.last_match_kind == "voice"
            and self.controller.last_result is not None
        ):
            self._draw_voice_score_panel(self.controller.last_result)
            return
        if (
            self.controller.last_match_kind == "gesture"
            and self.controller.last_gesture_result is not None
        ):
            self._draw_gesture_score_panel(self.controller.last_gesture_result)
            return
        imgui.text_disabled("No match attempts yet.")

    def _draw_voice_score_panel(self, result: RecognitionResult) -> None:
        from imgui_bundle import imgui

        imgui.text(
            "Voice accepted" if result.decision.accepted else "Voice rejected/fizzle"
        )
        if not result.ranking:
            imgui.text_disabled(result.decision.reason)
            return
        for row in result.ranking:
            conflict = _voice_margin_conflict(result, row.spell_id)
            state = (
                "conflict"
                if conflict
                else "normal"
                if result.decision.accepted
                else "rejected"
            )
            if result.decision.accepted:
                state = "accepted" if row == result.ranking[0] else "muted"
            self._draw_threshold_bar(
                _voice_match_score(result, row),
                0.0,
                label=row.name,
                show_marker=False,
                state=state,
            )
        if _voice_low_confidence(result):
            imgui.text_disabled(f"Closest spell is weak: {result.ranking[0].name}")
        elif _voice_margin_failure(result):
            imgui.text_disabled(
                f"Too close to choose: {result.ranking[0].name} / "
                f"{result.ranking[1].name}"
            )

    def _draw_gesture_score_panel(self, result: GestureResult) -> None:
        from imgui_bundle import imgui

        imgui.text(
            "Gesture accepted"
            if result.decision.accepted
            else "Gesture rejected/fizzle"
        )
        if not result.ranking:
            imgui.text_disabled(result.decision.reason)
            return
        for row in result.ranking:
            conflict = _gesture_margin_conflict(
                result, row.spell_id, self.controller.config.gesture.margin_min
            )
            state = (
                "conflict"
                if conflict
                else "normal"
                if result.decision.accepted
                else "rejected"
            )
            if result.decision.accepted:
                state = (
                    "accepted"
                    if result.decision.best_spell_id == row.spell_id
                    else "muted"
                )
            self._draw_threshold_bar(
                row.score,
                self.controller.config.gesture.score_min,
                label=row.name,
                state=state,
            )
        if _gesture_margin_failure(result, self.controller.config.gesture.margin_min):
            imgui.text_disabled(
                f"Too close to choose: {result.ranking[0].name} / "
                f"{result.ranking[1].name}"
            )

    def _draw_threshold_bar(
        self,
        value: float,
        threshold: float,
        *,
        label: str,
        higher_is_better: bool = True,
        show_marker: bool = True,
        scale_max: float = 1.0,
        size: tuple[float, float] = (260, 18),
        state: str = "normal",
    ) -> None:
        from imgui_bundle import imgui

        scale_max = max(scale_max, value, threshold, 1e-6)
        width = min(size[0], max(120.0, imgui.get_content_region_avail().x - 8.0))
        height = max(size[1], imgui.get_text_line_height() * 1.3)
        origin = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        bg = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.12, 0.11, 0.15, 1.0))
        ok_color = imgui.ImVec4(0.28, 0.78, 0.38, 1.0)
        bad_color = imgui.ImVec4(0.92, 0.35, 0.26, 1.0)
        conflict_color = imgui.ImVec4(0.95, 0.68, 0.22, 1.0)
        accepted_color = imgui.ImVec4(0.22, 0.88, 0.34, 1.0)
        muted_color = imgui.ImVec4(0.23, 0.32, 0.25, 1.0)
        rejected_color = imgui.ImVec4(0.38, 0.36, 0.44, 1.0)
        ok = imgui.color_convert_float4_to_u32(ok_color)
        bad = imgui.color_convert_float4_to_u32(bad_color)
        conflict = imgui.color_convert_float4_to_u32(conflict_color)
        accepted = imgui.color_convert_float4_to_u32(accepted_color)
        muted = imgui.color_convert_float4_to_u32(muted_color)
        rejected = imgui.color_convert_float4_to_u32(rejected_color)
        marker = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.92, 0.35, 1.0))
        text = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.92, 0.92, 0.96, 1.0))
        passes = value >= threshold if higher_is_better else value <= threshold
        fill_width = width * min(max(value / scale_max, 0.0), 1.0)
        draw_list.add_rect_filled(
            origin, imgui.ImVec2(origin.x + width, origin.y + height), bg, 3.0
        )
        draw_list.add_rect_filled(
            origin,
            imgui.ImVec2(origin.x + fill_width, origin.y + height),
            (
                accepted
                if state == "accepted"
                else muted
                if state == "muted"
                else rejected
                if state == "rejected"
                else conflict
                if state == "conflict"
                else ok
                if passes
                else bad
            ),
            3.0,
        )
        if show_marker:
            marker_x = origin.x + width * min(max(threshold / scale_max, 0.0), 1.0)
            draw_list.add_line(
                imgui.ImVec2(marker_x, origin.y - 2.0),
                imgui.ImVec2(marker_x, origin.y + height + 2.0),
                marker,
                2.0,
            )
        draw_list.add_text(
            imgui.ImVec2(
                origin.x + 5.0,
                origin.y + max(1.0, (height - imgui.get_text_line_height()) * 0.5),
            ),
            text,
            f"{label}: {value:.2f}",
        )
        imgui.dummy(imgui.ImVec2(width, height + 4.0))

    def _draw_bottom_log(self) -> None:
        from imgui_bundle import imgui

        entries = list(self.controller.ui_log)[-4:]
        if not entries:
            imgui.text_disabled("(no recent events)")
            return
        for index, entry in enumerate(entries):
            line = entry.format()
            imgui.text_unformatted(line)
            if self.overlay_mode:
                continue
            copy_text = _osc_parameter_from_log(line) or line
            imgui.same_line()
            imgui.push_id(f"log_copy_{index}")
            self._draw_copy_button(copy_text, tooltip="log OSC parameter")
            imgui.pop_id()

    def _draw_copy_button(self, text: str, *, tooltip: str) -> None:
        from imgui_bundle import imgui

        if imgui.small_button("Copy"):
            imgui.set_clipboard_text(text)
            self.controller.status = f"Copied {text}."
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Copy {tooltip}")

    def _draw_top_status(self) -> None:
        from imgui_bundle import imgui

        summary = self._osc_status_summary()
        if not self.overlay_mode:
            ui_enabled = self.controller.ui_enabled
            changed, ui_enabled = imgui.checkbox("Spellbook", ui_enabled)
            if changed:
                self.controller.set_ui_enabled(ui_enabled)
            imgui.same_line()
        voice_enabled = self.controller.voice_enabled
        changed, voice_enabled = imgui.checkbox("Voice", voice_enabled)
        if changed:
            self.controller.set_voice_enabled(voice_enabled)
        imgui.same_line()
        gesture_enabled = self.controller.gesture_enabled
        changed, gesture_enabled = imgui.checkbox("Gesture", gesture_enabled)
        if changed:
            self.controller.set_gesture_enabled(gesture_enabled)
        imgui.same_line()
        imgui.text_disabled(summary)

    def _draw_centered_activity_status(self) -> None:
        from imgui_bundle import imgui

        status = self._activity_status()
        width = imgui.calc_text_size(status).x
        avail = imgui.get_content_region_avail().x
        if avail > width:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail - width) * 0.5)
        imgui.text_disabled(status)

    def _hold_button(
        self,
        label: str,
        mode: str,
        *,
        allow_space: bool,
        size: tuple[float, float] = (230, 42),
    ) -> None:
        from imgui_bundle import imgui

        imgui.button(label, imgui.ImVec2(size[0], size[1]))
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
            mode == "recognize" or mode == "test" or mode == "name"
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
                self.controller.add_log(f"Voice fizzle: {exc}")
            self.controller.status = str(exc)
        except Exception as exc:
            LOGGER.exception("Recording action failed")
            if mode == "recognize" or mode == "test":
                self.controller.pulse_fizzle()
                self.controller.add_log(f"Voice fizzle: {exc}")
            self.controller.status = str(exc)

    def _handle_recording(self, mode: str, audio: FloatArray) -> None:
        if mode == "recognize" or mode == "test":
            self.controller.recognize(audio)
            return
        if mode == "name":
            spoken_name = self.controller.suggest_spell_name(audio)
            if self.keyboard_editing:
                self.edit_name = spoken_name
                self.pending_spoken_name = None
                self.pending_spoken_name_spell_id = None
            else:
                self.pending_spoken_name = spoken_name
                spell = self._selected_spell()
                self.pending_spoken_name_spell_id = (
                    spell.id if spell is not None else None
                )
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

    def _log_status_changes(self) -> None:
        changed = False
        input_status = self.controller.input_status
        if input_status is not None and input_status != self._last_logged_input_status:
            self._last_logged_input_status = input_status
            changed = True
        output_status = self.controller.output_status
        if (
            output_status is not None
            and output_status != self._last_logged_output_status
        ):
            self._last_logged_output_status = output_status
            changed = True
        if changed:
            self.controller.add_log(self._osc_status_summary())

    def _activity_status(self) -> str:
        if self.recording_mode is not None:
            if self.recording_mode == "recognize":
                return "voice: recording"
            if self.recording_mode == "sample":
                return "sample: recording"
            if self.recording_mode == "name":
                return "name: recording"
            return f"{self.recording_mode}: recording"
        if self.controller.armed_gesture_spell_id is not None:
            return "gesture: armed"
        return self.controller.status

    def _osc_status_summary(self) -> str:
        enabled = (
            self.controller.output_status is not None
            or self.controller.osc_input is not None
        )
        return f"OSC: {'on' if enabled else 'off'}"

    def _selected_spell(self) -> Spell | None:
        if self.selected_spell_id is None:
            return None
        for spell in self.controller.spellbook.spells:
            if spell.id == self.selected_spell_id:
                return spell
        self.selected_spell_id = None
        return None

    def _open_spell_page(self, spell: Spell) -> None:
        self.selected_spell_id = spell.id
        self.edit_name = spell.name
        self.page = self._page_for_spell_id(spell.id)

    def open_spell_after_gesture_action(self, result: object) -> None:
        if isinstance(result, Spell):
            self._open_spell_page(result)

    def _start_add_spell(self) -> None:
        self.controller.start_draft()
        self.selected_spell_id = None
        self.edit_name = self.controller.draft.name if self.controller.draft else ""
        self.page = len(self.controller.spellbook.spells) + 1

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
            return "Settings"
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

    def _draw_gesture_preview(
        self,
        label: str,
        points: FloatArray | None,
        size: tuple[float, float] = (260, 150),
    ) -> None:
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text(label)
        self._draw_gesture_canvas(points, size=size)

    def _draw_gesture_canvas(
        self,
        points: FloatArray | None,
        size: tuple[float, float] = (260, 150),
    ) -> None:
        from imgui_bundle import imgui

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
    controller.osc_input = osc_input
    osc_input.on_avatar_change = controller.sync_enable_toggles_to_output
    osc_input.start()
    controller.sync_enable_toggles_to_output()
    controller.status = "Loading voice model..."
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
    params.window_size = (1280, 860)
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


def _gesture_margin(ranking: tuple[GestureRanking, ...]) -> float | None:
    if len(ranking) < 2:
        return None
    second = ranking[1]
    if second.distance <= 1e-6:
        return 0.0
    return float((second.distance - ranking[0].distance) / second.distance)


def _gesture_margin_failure(result: GestureResult, threshold: float) -> bool:
    margin = _gesture_margin(result.ranking)
    return margin is not None and margin < threshold


def _gesture_margin_conflict(
    result: GestureResult, spell_id: str, threshold: float
) -> bool:
    if not _gesture_margin_failure(result, threshold):
        return False
    return any(row.spell_id == spell_id for row in result.ranking[:2])


def _gesture_ranking_for_spell(
    result: GestureResult, spell_id: str
) -> GestureRanking | None:
    for row in result.ranking:
        if row.spell_id == spell_id:
            return row
    return None


def _voice_ranking_for_spell(result: RecognitionResult, spell_id: str):
    for row in result.ranking:
        if row.spell_id == spell_id:
            return row
    return None


def _voice_match_score(result: RecognitionResult, row) -> float:
    if not result.ranking:
        return 0.0
    best_distance = max(result.ranking[0].aggregate_distance, 1e-6)
    return min(best_distance / max(row.aggregate_distance, 1e-6), 1.0)


def _voice_low_confidence(result: RecognitionResult) -> bool:
    decision = result.decision
    return (
        decision.intra_ratio is not None
        and decision.intra_ratio > decision.intra_ratio_max
    )


def _voice_margin_failure(result: RecognitionResult) -> bool:
    decision = result.decision
    return (
        decision.margin_ratio is not None
        and decision.margin_ratio < decision.margin_ratio_min
        and len(result.ranking) > 1
    )


def _voice_margin_conflict(result: RecognitionResult, spell_id: str) -> bool:
    if not _voice_margin_failure(result):
        return False
    return any(row.spell_id == spell_id for row in result.ranking[:2])


def _shorten_ui_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"...{text[-(max_chars - 3) :]}"


def _left_elide_text_to_width(text: str, max_width: float) -> str:
    from imgui_bundle import imgui

    if max_width <= 0.0:
        return ""
    if imgui.calc_text_size(text).x <= max_width:
        return text
    prefix = "..."
    if imgui.calc_text_size(prefix).x > max_width:
        return ""
    low = 0
    high = len(text)
    while low < high:
        keep = (low + high + 1) // 2
        candidate = f"{prefix}{text[-keep:]}"
        if imgui.calc_text_size(candidate).x <= max_width:
            low = keep
        else:
            high = keep - 1
    return f"{prefix}{text[-low:]}" if low > 0 else prefix


def _osc_parameter_from_log(text: str) -> str | None:
    marker = "(osc: "
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find(")", start)
    if end < 0:
        return None
    parameter = text[start:end].strip()
    return parameter or None


def _checkbox_width(label: str) -> float:
    from imgui_bundle import imgui

    style = imgui.get_style()
    return (
        imgui.get_frame_height()
        + style.item_inner_spacing.x
        + imgui.calc_text_size(label).x
    )


if __name__ == "__main__":
    sys.exit(main())
