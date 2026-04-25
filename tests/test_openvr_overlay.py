from __future__ import annotations

from typing import cast

import pytest

from osc_grimoire.config import OpenVrOverlayConfig
from osc_grimoire.desktop_ui import DesktopVoiceUi
from osc_grimoire.openvr_overlay import (
    OpenVrOverlayRunner,
    OverlayMouseState,
    is_trigger_pressed,
    next_mouse_events,
    overlay_transform_matrix,
    uv_to_imgui,
)


def test_uv_to_imgui_preserves_overlay_y_axis() -> None:
    assert uv_to_imgui(0.25, 0.75, 1000, 760) == (250.0, 570.0)


def test_trigger_pressed_uses_openvr_button_mask() -> None:
    assert is_trigger_pressed(1 << 33)
    assert not is_trigger_pressed(1 << 32)


def test_next_mouse_events_preserves_release_after_hover_loss() -> None:
    state = OverlayMouseState()

    events = next_mouse_events(
        state, hovering=True, trigger_down=True, position=(10.0, 20.0)
    )
    assert events == [("pos", (10.0, 20.0)), ("button", True)]

    events = next_mouse_events(state, hovering=False, trigger_down=False, position=None)
    assert events == [("pos", (-1.0, -1.0)), ("button", False)]


def test_overlay_transform_uses_configured_offset() -> None:
    openvr = pytest.importorskip("openvr")
    matrix = overlay_transform_matrix(
        OpenVrOverlayConfig(offset_x=1.0, offset_y=2.0, offset_z=3.0)
    )

    assert isinstance(matrix, openvr.HmdMatrix34_t)
    assert matrix.m[0][3] == 1.0
    assert matrix.m[1][3] == 2.0
    assert matrix.m[2][3] == 3.0


def test_overlay_config_defaults_left_anchor_right_pointer_large_overlay() -> None:
    config = OpenVrOverlayConfig()

    assert config.overlay_hand == "left"
    assert config.pointer_hand == "right"
    assert config.overlay_width_m == 0.50


def test_openvr_overlay_import_smoke() -> None:
    import osc_grimoire.openvr_overlay as openvr_overlay

    assert openvr_overlay.OVERLAY_KEY == "space.hiina.osc_grimoire.spellbook"


def test_runner_tolerates_missing_pose_array(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = OpenVrOverlayRunner(
        cast(DesktopVoiceUi, _FakeApp()), OpenVrOverlayConfig()
    )
    runner.openvr = _FakeOpenVr()
    runner.vr_system = _FakeSystem()
    runner.vr_overlay = _FakeOverlay()
    runner.overlay_handle = 123
    applied = []
    monkeypatch.setattr(
        runner,
        "_apply_mouse_events",
        lambda **kwargs: applied.append(kwargs),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: None)

    runner._inject_controller_input()

    assert applied == [{"hovering": False, "trigger_down": False, "position": None}]
    assert runner.app.controller.status == "Waiting for controller tracking..."


class _FakeController:
    status = ""


class _FakeApp:
    controller = _FakeController()


class _FakeOpenVr:
    k_unTrackedDeviceIndexInvalid = 999
    k_EButton_SteamVR_Trigger = 33
    TrackingUniverseStanding = 1
    TrackedControllerRole_LeftHand = 1
    TrackedControllerRole_RightHand = 2


class _FakeSystem:
    def getTrackedDeviceIndexForControllerRole(self, role: int) -> int:
        assert role == _FakeOpenVr.TrackedControllerRole_RightHand
        return 1

    def getControllerState(self, _device_index: int):
        return True, _FakeControllerState()

    def getDeviceToAbsoluteTrackingPose(self, _origin: int, _seconds: float, _poses):
        return None


class _FakeControllerState:
    ulButtonPressed = 0


class _FakeOverlay:
    pass
