from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from osc_grimoire.config import GestureRecognitionConfig, OpenVrOverlayConfig
from osc_grimoire.desktop_ui import DesktopVoiceUi
from osc_grimoire.openvr_overlay import (
    APP_KEY,
    OpenVrInputState,
    OpenVrOverlayRunner,
    OverlayMouseState,
    ensure_application_manifest,
    is_button_pressed,
    is_trigger_pressed,
    next_mouse_events,
    overlay_transform_matrix,
    stroke_points_to_pixels,
    trail_transform_matrix,
    uv_to_imgui,
)


def test_uv_to_imgui_preserves_overlay_y_axis() -> None:
    assert uv_to_imgui(0.25, 0.75, 1000, 760) == (250.0, 570.0)


def test_trigger_pressed_uses_openvr_button_mask() -> None:
    assert is_trigger_pressed(1 << 33)
    assert not is_trigger_pressed(1 << 32)


def test_button_pressed_uses_openvr_button_mask() -> None:
    assert is_button_pressed(1 << 2, 2)
    assert not is_button_pressed(1 << 1, 2)


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
    assert config.gesture_trail_width_m == 1.0


def test_stroke_points_to_pixels_matches_overlay_y_axis() -> None:
    points = [[0.0, 0.0], [0.25, -0.25]]

    pixels = stroke_points_to_pixels(
        cast(Any, points), texture_width=100, texture_height=100, width_m=1.0
    )

    assert pixels == [(50, 50), (75, 25)]


def test_trail_transform_uses_plane_axes_and_origin() -> None:
    openvr = pytest.importorskip("openvr")
    matrix = trail_transform_matrix(
        cast(Any, [1.0, 0.0, 0.0]),
        cast(Any, [0.0, 1.0, 0.0]),
        cast(Any, [2.0, 3.0, 4.0]),
    )

    assert isinstance(matrix, openvr.HmdMatrix34_t)
    assert matrix.m[0][0] == 1.0
    assert matrix.m[1][1] == 1.0
    assert matrix.m[2][2] == 1.0
    assert matrix.m[0][3] == 2.0
    assert matrix.m[1][3] == 3.0
    assert matrix.m[2][3] == 4.0


def test_openvr_overlay_import_smoke() -> None:
    import osc_grimoire.openvr_overlay as openvr_overlay

    assert openvr_overlay.OVERLAY_KEY == "space.hiina.osc_grimoire.spellbook"


def test_application_manifest_registers_action_manifest() -> None:
    path = ensure_application_manifest()

    payload = path.read_text(encoding="utf-8")

    assert APP_KEY in payload
    assert "action_manifest_path" in payload
    assert "actions.json" in payload


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
    monkeypatch.setattr(
        runner,
        "_input_state",
        lambda: OpenVrInputState(
            trigger_down=False,
            trigger_changed=False,
            grip_down=False,
            pose=_FakePose(_matrix((0, 0, 0))),
        ),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: None)

    runner._inject_controller_input()

    assert applied == [{"hovering": False, "trigger_down": False, "position": None}]
    assert runner.app.controller.status == "Waiting for controller tracking..."


def test_runner_routes_grip_stroke_to_controller() -> None:
    app = _FakeApp()
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    runner.vr_overlay = _FakeOverlay()
    runner.trail_overlay_handle = 456

    runner._update_gesture_capture(True, poses, poses[1])
    runner._update_gesture_capture(False, poses, poses[1])

    assert app.controller.gesture_count == 1
    assert runner.vr_overlay.shown == [456]
    assert runner.vr_overlay.hidden == [456]


def test_runner_ignores_grip_when_gesture_disabled() -> None:
    app = _FakeApp()
    app.controller.gesture_enabled = False
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    runner.vr_overlay = _FakeOverlay()
    runner.trail_overlay_handle = 456

    runner._update_gesture_capture(True, poses, poses[1])
    runner._cancel_gesture_capture()

    assert app.controller.gesture_count == 0
    assert runner.vr_overlay.shown == []
    assert runner.vr_overlay.hidden == []


def test_trigger_off_overlay_records_voice_not_mouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeApp()
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    runner.vr_system = _FakeSystem()
    runner.vr_overlay = _FakeOverlay()
    runner.overlay_handle = 123
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    applied = []
    monkeypatch.setattr(
        runner,
        "_input_state",
        lambda: OpenVrInputState(
            trigger_down=True,
            trigger_changed=True,
            grip_down=False,
            pose=poses[1],
        ),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: poses)
    monkeypatch.setattr(runner, "_compute_intersection", lambda _ray: None)
    monkeypatch.setattr(
        runner,
        "_apply_mouse_events",
        lambda **kwargs: applied.append(kwargs),
    )

    runner._inject_controller_input()

    assert app.voice_begin_count == 1
    assert app.voice_finish_count == 0
    assert runner.voice_trigger_down
    assert applied == [{"hovering": False, "trigger_down": False, "position": None}]


def test_trigger_off_overlay_ignores_voice_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeApp()
    app.controller.voice_enabled = False
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    runner.vr_system = _FakeSystem()
    runner.vr_overlay = _FakeOverlay()
    runner.overlay_handle = 123
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    monkeypatch.setattr(
        runner,
        "_input_state",
        lambda: OpenVrInputState(
            trigger_down=True,
            trigger_changed=True,
            grip_down=False,
            pose=poses[1],
        ),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: poses)
    monkeypatch.setattr(runner, "_compute_intersection", lambda _ray: None)
    monkeypatch.setattr(runner, "_apply_mouse_events", lambda **_kwargs: None)

    runner._inject_controller_input()

    assert app.voice_begin_count == 0
    assert not runner.voice_trigger_down


def test_trigger_changed_without_pressed_state_does_not_finish_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeApp()
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    runner.vr_system = _FakeSystem()
    runner.vr_overlay = _FakeOverlay()
    runner.overlay_handle = 123
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    monkeypatch.setattr(
        runner,
        "_input_state",
        lambda: OpenVrInputState(
            trigger_down=False,
            trigger_changed=True,
            grip_down=False,
            pose=poses[1],
        ),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: poses)
    monkeypatch.setattr(runner, "_compute_intersection", lambda _ray: None)
    monkeypatch.setattr(runner, "_apply_mouse_events", lambda **_kwargs: None)

    runner._inject_controller_input()

    assert app.voice_begin_count == 0
    assert app.voice_finish_count == 0
    assert not runner.voice_trigger_down


def test_trigger_off_overlay_finishes_voice_on_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeApp()
    runner = OpenVrOverlayRunner(cast(DesktopVoiceUi, app), OpenVrOverlayConfig())
    runner.openvr = _FakeOpenVr()
    runner.vr_system = _FakeSystem()
    runner.vr_overlay = _FakeOverlay()
    runner.overlay_handle = 123
    runner.trigger_down = True
    runner.voice_trigger_down = True
    poses = [_FakePose(_matrix((0, 0, 0))) for _ in range(3)]
    monkeypatch.setattr(
        runner,
        "_input_state",
        lambda: OpenVrInputState(
            trigger_down=False,
            trigger_changed=True,
            grip_down=False,
            pose=poses[1],
        ),
    )
    monkeypatch.setattr(runner, "_tracked_device_poses", lambda: poses)
    monkeypatch.setattr(runner, "_compute_intersection", lambda _ray: None)
    monkeypatch.setattr(runner, "_apply_mouse_events", lambda **_kwargs: None)

    runner._inject_controller_input()

    assert app.voice_begin_count == 0
    assert app.voice_finish_count == 1
    assert not runner.voice_trigger_down


class _FakeController:
    status = ""
    config = type("Config", (), {"gesture": GestureRecognitionConfig()})()

    def __init__(self) -> None:
        self.gesture_count = 0
        self.gesture_drawing: list[bool] = []
        self.voice_recording: list[bool] = []
        self.ui_enabled = True
        self.gesture_enabled = True
        self.voice_enabled = True

    def handle_gesture_stroke(self, _points) -> None:
        self.gesture_count += 1

    def set_gesture_drawing(self, drawing: bool) -> None:
        self.gesture_drawing.append(drawing)

    def set_voice_recording(self, recording: bool) -> None:
        self.voice_recording.append(recording)


class _FakeApp:
    def __init__(self) -> None:
        self.controller = _FakeController()
        self.voice_begin_count = 0
        self.voice_finish_count = 0

    def begin_overlay_voice_recording(self) -> None:
        self.voice_begin_count += 1

    def finish_overlay_voice_recording(self) -> None:
        self.voice_finish_count += 1


class _FakeOpenVr:
    k_unTrackedDeviceIndexInvalid = 999
    k_unTrackedDeviceIndex_Hmd = 0
    k_EButton_SteamVR_Trigger = 33
    k_EButton_Grip = 2
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
    def __init__(self) -> None:
        self.shown: list[int] = []
        self.hidden: list[int] = []

    def showOverlay(self, handle: int) -> None:
        self.shown.append(handle)

    def hideOverlay(self, handle: int) -> None:
        self.hidden.append(handle)

    def setOverlayTransformAbsolute(self, _handle: int, _origin: int, _matrix) -> None:
        pass


class _FakePose:
    bPoseIsValid = True

    def __init__(self, matrix) -> None:
        self.mDeviceToAbsoluteTracking = matrix


def _matrix(translation: tuple[float, float, float]):
    return SimpleNamespace(
        m=[
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
        ]
    )
