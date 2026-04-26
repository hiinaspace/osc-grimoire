from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import AppConfig, OpenVrOverlayConfig
from .desktop_controller import VoiceTrainingController
from .desktop_ui import DesktopVoiceUi
from .gesture_capture import GestureStrokeSampler
from .osc_input import OscInputService
from .osc_output import OscOutput
from .paths import default_data_dir

LOGGER = logging.getLogger(__name__)
OVERLAY_KEY = "space.hiina.osc_grimoire.spellbook"
OVERLAY_NAME = "OSC Grimoire Spellbook"
TRAIL_OVERLAY_KEY = "space.hiina.osc_grimoire.gesture_trail"
TRAIL_OVERLAY_NAME = "OSC Grimoire Gesture Trail"
APP_KEY = "space.hiina.osc_grimoire"


@dataclass(frozen=True)
class Ray:
    source: tuple[float, float, float]
    direction: tuple[float, float, float]


@dataclass
class OverlayMouseState:
    trigger_down: bool = False
    hovering: bool = False


MouseEvent = tuple[str, tuple[float, float]] | tuple[str, bool]


@dataclass(frozen=True)
class OpenVrActionHandles:
    action_set: int
    right_trigger: int
    right_grip: int
    right_pose: int
    right_source: int


@dataclass(frozen=True)
class OpenVrInputState:
    trigger_down: bool
    trigger_changed: bool
    grip_down: bool
    pose: Any | None


def action_manifest_path() -> Path:
    return Path(__file__).with_name("assets") / "actions.json"


def ensure_application_manifest() -> Path:
    manifest_dir = Path(tempfile.gettempdir()) / "osc-grimoire"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "app.vrmanifest"
    payload = {
        "source": "builtin",
        "applications": [
            {
                "app_key": APP_KEY,
                "launch_type": "binary",
                "binary_path_windows": sys.executable,
                "working_directory": str(Path.cwd()),
                "arguments": " ".join(sys.argv[1:]),
                "action_manifest_path": str(action_manifest_path()),
                "is_dashboard_overlay": False,
                "strings": {
                    "en_us": {
                        "name": "OSC Grimoire",
                        "description": "VR spellbook overlay for voice and gesture spell casting.",
                    }
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def uv_to_imgui(
    u: float, v: float, texture_width: int, texture_height: int
) -> tuple[float, float]:
    return (u * texture_width, v * texture_height)


def is_trigger_pressed(button_mask: int, trigger_button_id: int = 33) -> bool:
    return bool(button_mask & (1 << trigger_button_id))


def is_button_pressed(button_mask: int, button_id: int) -> bool:
    return bool(button_mask & (1 << button_id))


def next_mouse_events(
    state: OverlayMouseState,
    *,
    hovering: bool,
    trigger_down: bool,
    position: tuple[float, float] | None,
) -> list[MouseEvent]:
    events: list[MouseEvent] = []
    if hovering and position is not None:
        events.append(("pos", position))
    elif state.hovering:
        events.append(("pos", (-1.0, -1.0)))

    if trigger_down != state.trigger_down:
        events.append(("button", trigger_down))

    state.hovering = hovering
    state.trigger_down = trigger_down
    return events


def overlay_transform_matrix(config: OpenVrOverlayConfig) -> Any:
    import openvr

    matrix = openvr.HmdMatrix34_t()
    matrix.m[0][0] = 1.0
    matrix.m[0][1] = 0.0
    matrix.m[0][2] = 0.0
    matrix.m[0][3] = config.offset_x
    matrix.m[1][0] = 0.0
    matrix.m[1][1] = 1.0
    matrix.m[1][2] = 0.0
    matrix.m[1][3] = config.offset_y
    matrix.m[2][0] = 0.0
    matrix.m[2][1] = 0.0
    matrix.m[2][2] = 1.0
    matrix.m[2][3] = config.offset_z
    return matrix


def trail_transform_matrix(
    right: np.ndarray, up: np.ndarray, origin: np.ndarray
) -> Any:
    import openvr

    normal = np.cross(right, up)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 0.0:
        normal = normal / normal_norm
    matrix = openvr.HmdMatrix34_t()
    for row in range(3):
        matrix.m[row][0] = float(right[row])
        matrix.m[row][1] = float(up[row])
        matrix.m[row][2] = float(normal[row])
        matrix.m[row][3] = float(origin[row])
    return matrix


def stroke_points_to_pixels(
    points: np.ndarray, texture_width: int, texture_height: int, width_m: float
) -> list[tuple[int, int]]:
    if width_m <= 0.0:
        return []
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] == 0:
        return []
    scale_x = texture_width / width_m
    scale_y = texture_height / width_m
    center_x = texture_width * 0.5
    center_y = texture_height * 0.5
    pixels: list[tuple[int, int]] = []
    for point in array:
        x = int(round(center_x + float(point[0]) * scale_x))
        y = int(round(center_y + float(point[1]) * scale_y))
        pixels.append((x, y))
    return pixels


def ray_from_pose(pose: Any) -> Ray:
    matrix = pose.mDeviceToAbsoluteTracking.m
    source = (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))
    direction = (
        -float(matrix[0][2]),
        -float(matrix[1][2]),
        -float(matrix[2][2]),
    )
    norm = float(np.linalg.norm(np.asarray(direction, dtype=np.float32)))
    if norm <= 0.0:
        direction = (0.0, 0.0, -1.0)
    else:
        direction = (
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        )
    return Ray(source=source, direction=direction)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    config = AppConfig()
    osc_output = OscOutput(config.osc)
    osc_input = OscInputService(config.osc)
    osc_input.start()
    controller = VoiceTrainingController(
        data_dir, config=config, output=osc_output, osc_input=osc_input
    )
    controller.status = "Loading Whisper model..."
    controller.preload_backend()
    controller.status = "Starting OpenVR overlay..."
    app = DesktopVoiceUi(
        controller,
        overlay_mode=True,
        surface_size=(
            controller.config.openvr.texture_width,
            controller.config.openvr.texture_height,
        ),
    )
    runner = OpenVrOverlayRunner(app, controller.config.openvr)
    try:
        runner.run()
    except OpenVrOverlayError as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        app.shutdown()
    return 0


class OpenVrOverlayError(RuntimeError):
    pass


class OpenVrOverlayRunner:
    def __init__(self, app: DesktopVoiceUi, config: OpenVrOverlayConfig) -> None:
        self.app = app
        self.config = config
        self.renderer: HiddenGlfwImGuiRenderer | None = None
        self.openvr: Any = None
        self.vr_system: Any = None
        self.vr_overlay: Any = None
        self.vr_input: Any = None
        self.action_handles: OpenVrActionHandles | None = None
        self.overlay_handle: int | None = None
        self.trail_overlay_handle: int | None = None
        self.trail_texture: StrokeTrailTexture | None = None
        self.mouse_state = OverlayMouseState()
        self.gesture_sampler = GestureStrokeSampler(app.controller.config.gesture)
        self.grip_down = False
        self.trigger_down = False
        self.voice_trigger_down = False
        self._shutdown_openvr = False

    def run(self) -> None:
        self._init_openvr()
        self.renderer = HiddenGlfwImGuiRenderer(
            self.config.texture_width, self.config.texture_height
        )
        self.trail_texture = StrokeTrailTexture(
            self.config.gesture_trail_texture_size,
            self.config.gesture_trail_texture_size,
            self.config.gesture_trail_width_m,
        )
        try:
            assert self.overlay_handle is not None
            self.app.controller.status = "Ready."
            while not self.renderer.should_close():
                self.renderer.poll_events()
                self._update_overlay_transform()
                self._inject_controller_input()
                self.renderer.render_frame(self.app.draw)
                self._submit_overlay_texture()
                time.sleep(1.0 / 90.0)
        except KeyboardInterrupt:
            return
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.app.controller.set_voice_recording(False)
        self.app.controller.set_gesture_drawing(False)
        if self.vr_overlay is not None and self.overlay_handle is not None:
            try:
                self.vr_overlay.destroyOverlay(self.overlay_handle)
            except Exception:
                LOGGER.debug("Failed to destroy OpenVR overlay", exc_info=True)
        if self.vr_overlay is not None and self.trail_overlay_handle is not None:
            try:
                self.vr_overlay.destroyOverlay(self.trail_overlay_handle)
            except Exception:
                LOGGER.debug("Failed to destroy gesture trail overlay", exc_info=True)
        if self.trail_texture is not None:
            self.trail_texture.shutdown()
            self.trail_texture = None
        if self.renderer is not None:
            self.renderer.shutdown()
            self.renderer = None
        if self._shutdown_openvr and self.openvr is not None:
            self.openvr.shutdown()
            self._shutdown_openvr = False

    def _init_openvr(self) -> None:
        try:
            import openvr
        except ImportError as exc:
            raise OpenVrOverlayError(
                "OpenVR support is not installed. Run `uv sync` and try again."
            ) from exc

        self.openvr = openvr
        try:
            openvr.init(openvr.VRApplication_Overlay)
            self._shutdown_openvr = True
            self.vr_system = openvr.VRSystem()
            self.vr_overlay = openvr.VROverlay()
            self.vr_input = openvr.VRInput()
            self._register_application_manifest()
            self._configure_actions(action_manifest_path())
            self.overlay_handle = self.vr_overlay.createOverlay(
                OVERLAY_KEY, OVERLAY_NAME
            )
            self.trail_overlay_handle = self.vr_overlay.createOverlay(
                TRAIL_OVERLAY_KEY, TRAIL_OVERLAY_NAME
            )
            mouse_scale = openvr.HmdVector2_t()
            mouse_scale.v[0] = float(self.config.texture_width)
            mouse_scale.v[1] = float(self.config.texture_height)
            self.vr_overlay.setOverlayMouseScale(self.overlay_handle, mouse_scale)
            self.vr_overlay.setOverlayInputMethod(
                self.overlay_handle, openvr.VROverlayInputMethod_Mouse
            )
            self.vr_overlay.setOverlayWidthInMeters(
                self.overlay_handle, self.config.overlay_width_m
            )
            self.vr_overlay.setOverlayWidthInMeters(
                self.trail_overlay_handle, self.config.gesture_trail_width_m
            )
            self.vr_overlay.showOverlay(self.overlay_handle)
        except Exception as exc:
            if self._shutdown_openvr:
                openvr.shutdown()
                self._shutdown_openvr = False
            raise OpenVrOverlayError(
                "Could not start OpenVR overlay. Make sure SteamVR is running and "
                "updated, then retry `uv run osc-grimoire-overlay --data-dir ./data`."
            ) from exc

    def _register_application_manifest(self) -> None:
        assert self.openvr is not None
        applications = self.openvr.VRApplications()
        applications.addApplicationManifest(str(ensure_application_manifest()), True)
        applications.identifyApplication(os.getpid(), APP_KEY)

    def _update_overlay_transform(self) -> None:
        assert self.openvr is not None
        assert self.vr_system is not None
        assert self.vr_overlay is not None
        assert self.overlay_handle is not None
        device_index = self._overlay_device_index()
        if device_index == self.openvr.k_unTrackedDeviceIndexInvalid:
            self.app.controller.status = "Waiting for left controller..."
            return
        transform = overlay_transform_matrix(self.config)
        self.vr_overlay.setOverlayTransformTrackedDeviceRelative(
            self.overlay_handle, device_index, transform
        )

    def _inject_controller_input(self) -> None:
        assert self.openvr is not None
        assert self.vr_system is not None
        assert self.vr_overlay is not None
        assert self.overlay_handle is not None
        input_state = self._input_state()
        trigger_pressed = input_state.trigger_down and not self.trigger_down
        trigger_released = not input_state.trigger_down and self.trigger_down
        if input_state.pose is None:
            if trigger_released and self.voice_trigger_down:
                self.app.finish_overlay_voice_recording()
                self.voice_trigger_down = False
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            self.trigger_down = input_state.trigger_down
            return

        poses = self._tracked_device_poses()
        if poses is None:
            self.app.controller.status = "Waiting for controller tracking..."
            if trigger_released and self.voice_trigger_down:
                self.app.finish_overlay_voice_recording()
                self.voice_trigger_down = False
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            self.trigger_down = input_state.trigger_down
            return
        if not input_state.pose.bPoseIsValid:
            if trigger_released and self.voice_trigger_down:
                self.app.finish_overlay_voice_recording()
                self.voice_trigger_down = False
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            self._update_gesture_capture(False, poses, input_state.pose)
            self.trigger_down = input_state.trigger_down
            return
        self._update_gesture_capture(input_state.grip_down, poses, input_state.pose)
        ray = ray_from_pose(input_state.pose)
        intersection = self._compute_intersection(ray)
        if self.voice_trigger_down:
            self._update_overlay_voice_recording(trigger_pressed, trigger_released)
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            self.trigger_down = input_state.trigger_down
            return
        if intersection is None:
            self._update_overlay_voice_recording(trigger_pressed, trigger_released)
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            self.trigger_down = input_state.trigger_down
            return
        if trigger_released and self.voice_trigger_down:
            self.app.finish_overlay_voice_recording()
            self.voice_trigger_down = False
        self._apply_mouse_events(
            hovering=True,
            trigger_down=input_state.trigger_down,
            position=intersection,
        )
        self.trigger_down = input_state.trigger_down

    def _update_gesture_capture(
        self, grip_down: bool, poses: Any, pointer_pose: Any
    ) -> None:
        assert self.openvr is not None
        hmd_pose = poses[self.openvr.k_unTrackedDeviceIndex_Hmd]
        if grip_down and not self.grip_down:
            if not hmd_pose.bPoseIsValid or not pointer_pose.bPoseIsValid:
                self.app.controller.status = "Waiting for gesture tracking..."
                self.grip_down = grip_down
                return
            self.gesture_sampler.begin(hmd_pose.mDeviceToAbsoluteTracking)
            self.gesture_sampler.add_controller_pose(
                pointer_pose.mDeviceToAbsoluteTracking
            )
            self._show_gesture_trail()
            self._update_gesture_trail()
            self.app.controller.set_gesture_drawing(True)
            self.app.controller.status = "Recording gesture..."
        elif grip_down and self.gesture_sampler.active and pointer_pose.bPoseIsValid:
            self.gesture_sampler.add_controller_pose(
                pointer_pose.mDeviceToAbsoluteTracking
            )
            self._update_gesture_trail()
        elif not grip_down and self.grip_down:
            points = self.gesture_sampler.finish()
            self._hide_gesture_trail()
            self.app.controller.set_gesture_drawing(False)
            try:
                self.app.controller.handle_gesture_stroke(points)
            except Exception as exc:
                LOGGER.exception("Gesture action failed")
                self.app.controller.status = str(exc)
        self.grip_down = grip_down

    def _update_overlay_voice_recording(
        self, trigger_pressed: bool, trigger_released: bool
    ) -> None:
        if trigger_pressed:
            self.app.begin_overlay_voice_recording()
            self.voice_trigger_down = True
        elif trigger_released and self.voice_trigger_down:
            self.app.finish_overlay_voice_recording()
            self.voice_trigger_down = False

    def _configure_actions(self, manifest_path: Path) -> None:
        assert self.vr_input is not None
        self.vr_input.setActionManifestPath(str(manifest_path))
        self.action_handles = OpenVrActionHandles(
            action_set=self.vr_input.getActionSetHandle("/actions/main"),
            right_trigger=self.vr_input.getActionHandle(
                "/actions/main/in/right_trigger"
            ),
            right_grip=self.vr_input.getActionHandle("/actions/main/in/right_grip"),
            right_pose=self.vr_input.getActionHandle("/actions/main/in/right_pose"),
            right_source=self.vr_input.getInputSourceHandle("/user/hand/right"),
        )

    def _input_state(self) -> OpenVrInputState:
        self._update_action_state()
        digital_data = self._digital_action_data("right_trigger")
        return OpenVrInputState(
            trigger_down=bool(digital_data.bActive and digital_data.bState),
            trigger_changed=bool(digital_data.bActive and digital_data.bChanged),
            grip_down=self._digital_action_state("right_grip"),
            pose=self._right_pose_action(),
        )

    def _update_action_state(self) -> None:
        assert self.openvr is not None
        assert self.vr_input is not None
        assert self.action_handles is not None
        action_set = self.openvr.VRActiveActionSet_t()
        action_set.ulActionSet = self.action_handles.action_set
        action_set.ulRestrictedToDevice = self.openvr.k_ulInvalidInputValueHandle
        action_set.ulSecondaryActionSet = self.openvr.k_ulInvalidActionHandle
        action_set.nPriority = 0
        sets = (self.openvr.VRActiveActionSet_t * 1)()
        sets[0] = action_set
        self.vr_input.updateActionState(sets)

    def _digital_action_state(self, name: str) -> bool:
        data = self._digital_action_data(name)
        return bool(data.bActive and data.bState)

    def _digital_action_changed(self, name: str) -> bool:
        data = self._digital_action_data(name)
        return bool(data.bActive and data.bChanged)

    def _digital_action_data(self, name: str) -> Any:
        assert self.openvr is not None
        assert self.vr_input is not None
        assert self.action_handles is not None
        handle = getattr(self.action_handles, name)
        return self.vr_input.getDigitalActionData(
            handle, self.action_handles.right_source
        )

    def _right_pose_action(self) -> Any | None:
        assert self.openvr is not None
        assert self.vr_input is not None
        assert self.action_handles is not None
        pose_data = self.vr_input.getPoseActionDataForNextFrame(
            self.action_handles.right_pose,
            self.openvr.TrackingUniverseStanding,
            self.action_handles.right_source,
        )
        if pose_data.bActive and pose_data.pose.bPoseIsValid:
            return pose_data.pose
        return self._fallback_pointer_pose()

    def _show_gesture_trail(self) -> None:
        assert self.vr_overlay is not None
        assert self.trail_overlay_handle is not None
        if not self._update_gesture_trail_transform():
            return
        self.vr_overlay.showOverlay(self.trail_overlay_handle)

    def _hide_gesture_trail(self) -> None:
        if self.vr_overlay is None or self.trail_overlay_handle is None:
            return
        if self.trail_texture is not None:
            self.trail_texture.clear()
        self.vr_overlay.hideOverlay(self.trail_overlay_handle)

    def _update_gesture_trail(self) -> None:
        if self.trail_texture is None:
            return
        self._update_gesture_trail_transform()
        self.trail_texture.update(self.gesture_sampler.points)
        self._submit_trail_texture()

    def _update_gesture_trail_transform(self) -> bool:
        assert self.openvr is not None
        assert self.vr_overlay is not None
        assert self.trail_overlay_handle is not None
        right = self.gesture_sampler.right
        up = self.gesture_sampler.up
        origin = self.gesture_sampler.origin
        if right is None or up is None or origin is None:
            return False
        transform = trail_transform_matrix(right, up, origin)
        self.vr_overlay.setOverlayTransformAbsolute(
            self.trail_overlay_handle, self.openvr.TrackingUniverseStanding, transform
        )
        return True

    def _apply_mouse_events(
        self,
        *,
        hovering: bool,
        trigger_down: bool,
        position: tuple[float, float] | None,
    ) -> None:
        from imgui_bundle import imgui

        io = imgui.get_io()
        for event_type, value in next_mouse_events(
            self.mouse_state,
            hovering=hovering,
            trigger_down=trigger_down,
            position=position,
        ):
            if event_type == "pos":
                assert isinstance(value, tuple)
                x, y = value
                io.add_mouse_pos_event(float(x), float(y))
            elif event_type == "button":
                io.add_mouse_button_event(0, bool(value))

    def _compute_intersection(self, ray: Ray) -> tuple[float, float] | None:
        assert self.openvr is not None
        assert self.vr_overlay is not None
        assert self.overlay_handle is not None
        params = self.openvr.VROverlayIntersectionParams_t()
        for i, value in enumerate(ray.source):
            params.vSource.v[i] = value
        for i, value in enumerate(ray.direction):
            params.vDirection.v[i] = value
        params.eOrigin = self.openvr.TrackingUniverseStanding
        hit, results = self.vr_overlay.computeOverlayIntersection(
            self.overlay_handle, params
        )
        if not hit:
            return None
        return uv_to_imgui(
            float(results.vUVs.v[0]),
            float(results.vUVs.v[1]),
            self.config.texture_width,
            self.config.texture_height,
        )

    def _submit_overlay_texture(self) -> None:
        assert self.openvr is not None
        assert self.vr_overlay is not None
        assert self.overlay_handle is not None
        assert self.renderer is not None
        texture = self.openvr.Texture_t(
            ctypes.c_void_p(int(self.renderer.texture_id)),
            self.openvr.TextureType_OpenGL,
            self.openvr.ColorSpace_Auto,
        )
        self.vr_overlay.setOverlayTexture(self.overlay_handle, texture)

    def _submit_trail_texture(self) -> None:
        assert self.openvr is not None
        assert self.vr_overlay is not None
        assert self.trail_overlay_handle is not None
        assert self.trail_texture is not None
        texture = self.openvr.Texture_t(
            ctypes.c_void_p(int(self.trail_texture.texture_id)),
            self.openvr.TextureType_OpenGL,
            self.openvr.ColorSpace_Auto,
        )
        self.vr_overlay.setOverlayTexture(self.trail_overlay_handle, texture)

    def _overlay_device_index(self) -> int:
        assert self.openvr is not None
        assert self.vr_system is not None
        return self._device_index_for_hand(self.config.overlay_hand)

    def _pointer_device_index(self) -> int:
        return self._device_index_for_hand(self.config.pointer_hand)

    def _fallback_pointer_pose(self) -> Any | None:
        device_index = self._pointer_device_index()
        if device_index == self.openvr.k_unTrackedDeviceIndexInvalid:
            return None
        poses = self._tracked_device_poses()
        if poses is None:
            return None
        pose = poses[device_index]
        return pose if pose.bPoseIsValid else None

    def _device_index_for_hand(self, hand: str) -> int:
        assert self.openvr is not None
        assert self.vr_system is not None
        role = self.openvr.TrackedControllerRole_LeftHand
        if hand == "right":
            role = self.openvr.TrackedControllerRole_RightHand
        return int(self.vr_system.getTrackedDeviceIndexForControllerRole(role))

    def _tracked_device_poses(self) -> Any:
        assert self.openvr is not None
        assert self.vr_system is not None
        pose_array = (
            self.openvr.TrackedDevicePose_t * self.openvr.k_unMaxTrackedDeviceCount
        )()
        return self.vr_system.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding, 0.0, pose_array
        )


class HiddenGlfwImGuiRenderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.window: Any = None
        self.impl: Any = None
        self.framebuffer_id = 0
        self.texture_id = 0
        self.depth_buffer_id = 0
        self._init()

    def should_close(self) -> bool:
        import glfw

        return bool(glfw.window_should_close(self.window))

    def poll_events(self) -> None:
        import glfw

        glfw.poll_events()

    def render_frame(self, draw: Any) -> None:
        import OpenGL.GL as gl
        from imgui_bundle import imgui

        assert self.impl is not None
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer_id)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(0.055, 0.045, 0.065, 1.0)
        clear_flags = cast(Any, gl.GL_COLOR_BUFFER_BIT) | cast(
            Any, gl.GL_DEPTH_BUFFER_BIT
        )
        gl.glClear(clear_flags)
        self.impl.process_inputs()
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(float(self.width), float(self.height))
        io.display_framebuffer_scale = imgui.ImVec2(1.0, 1.0)
        imgui.new_frame()
        draw()
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def shutdown(self) -> None:
        import glfw
        import OpenGL.GL as gl

        if self.impl is not None:
            self.impl.shutdown()
            self.impl = None
        if self.framebuffer_id:
            gl.glDeleteFramebuffers(1, [self.framebuffer_id])
            self.framebuffer_id = 0
        if self.texture_id:
            gl.glDeleteTextures([self.texture_id])
            self.texture_id = 0
        if self.depth_buffer_id:
            gl.glDeleteRenderbuffers(1, [self.depth_buffer_id])
            self.depth_buffer_id = 0
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def _init(self) -> None:
        import glfw
        from imgui_bundle import imgui, implot
        from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

        if not glfw.init():
            raise OpenVrOverlayError("Could not initialize GLFW.")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.width, self.height, "OSC Grimoire Overlay", None, None
        )
        if self.window is None:
            glfw.terminate()
            raise OpenVrOverlayError("Could not create hidden OpenGL window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        imgui.create_context()
        implot.create_context()
        self.impl = GlfwRenderer(self.window, attach_callbacks=False)
        self._create_render_target()

    def _create_render_target(self) -> None:
        import OpenGL.GL as gl

        self.texture_id = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        self.depth_buffer_id = int(gl.glGenRenderbuffers(1))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth_buffer_id)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.width, self.height
        )

        self.framebuffer_id = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer_id)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.texture_id,
            0,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_STENCIL_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.depth_buffer_id,
        )
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise OpenVrOverlayError(f"OpenGL framebuffer is incomplete: {status}")


class StrokeTrailTexture:
    def __init__(self, width: int, height: int, width_m: float) -> None:
        self.width = width
        self.height = height
        self.width_m = width_m
        self.texture_id = 0
        self._pixels = np.zeros((height, width, 4), dtype=np.uint8)
        self._init_texture()

    def update(self, points: np.ndarray) -> None:
        self._pixels.fill(0)
        pixels = stroke_points_to_pixels(points, self.width, self.height, self.width_m)
        if len(pixels) == 1:
            _draw_point(self._pixels, pixels[0], radius=3)
        else:
            for start, end in zip(pixels[:-1], pixels[1:], strict=False):
                _draw_line(self._pixels, start, end, radius=2)
        self._upload()

    def clear(self) -> None:
        self._pixels.fill(0)
        self._upload()

    def shutdown(self) -> None:
        if not self.texture_id:
            return
        import OpenGL.GL as gl

        gl.glDeleteTextures([self.texture_id])
        self.texture_id = 0

    def _init_texture(self) -> None:
        import OpenGL.GL as gl

        self.texture_id = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self._pixels,
        )

    def _upload(self) -> None:
        import OpenGL.GL as gl

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self._pixels,
        )


def _draw_line(
    pixels: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, radius: int
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _draw_point(pixels, (x0, y0), radius=radius)
        if x0 == x1 and y0 == y1:
            return
        double_error = 2 * error
        if double_error >= dy:
            error += dy
            x0 += sx
        if double_error <= dx:
            error += dx
            y0 += sy


def _draw_point(pixels: np.ndarray, point: tuple[int, int], *, radius: int) -> None:
    x, y = point
    height, width = pixels.shape[:2]
    for py in range(max(0, y - radius), min(height, y + radius + 1)):
        for px in range(max(0, x - radius), min(width, x + radius + 1)):
            if (px - x) * (px - x) + (py - y) * (py - y) <= radius * radius:
                pixels[py, px] = (64, 255, 96, 230)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSC Grimoire OpenVR overlay")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
