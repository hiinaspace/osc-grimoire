from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import OpenVrOverlayConfig
from .desktop_controller import VoiceTrainingController
from .desktop_ui import DesktopVoiceUi
from .gesture_capture import GestureStrokeSampler
from .paths import default_data_dir

LOGGER = logging.getLogger(__name__)
OVERLAY_KEY = "space.hiina.osc_grimoire.spellbook"
OVERLAY_NAME = "OSC Grimoire Spellbook"


@dataclass(frozen=True)
class Ray:
    source: tuple[float, float, float]
    direction: tuple[float, float, float]


@dataclass
class OverlayMouseState:
    trigger_down: bool = False
    hovering: bool = False


MouseEvent = tuple[str, tuple[float, float]] | tuple[str, bool]


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
    controller = VoiceTrainingController(data_dir)
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
        self.overlay_handle: int | None = None
        self.mouse_state = OverlayMouseState()
        self.gesture_sampler = GestureStrokeSampler(app.controller.config.gesture)
        self.grip_down = False
        self._shutdown_openvr = False

    def run(self) -> None:
        self._init_openvr()
        self.renderer = HiddenGlfwImGuiRenderer(
            self.config.texture_width, self.config.texture_height
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
        if self.vr_overlay is not None and self.overlay_handle is not None:
            try:
                self.vr_overlay.destroyOverlay(self.overlay_handle)
            except Exception:
                LOGGER.debug("Failed to destroy OpenVR overlay", exc_info=True)
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
            self.overlay_handle = self.vr_overlay.createOverlay(
                OVERLAY_KEY, OVERLAY_NAME
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
            self.vr_overlay.showOverlay(self.overlay_handle)
        except Exception as exc:
            if self._shutdown_openvr:
                openvr.shutdown()
                self._shutdown_openvr = False
            raise OpenVrOverlayError(
                "Could not start OpenVR overlay. Make sure SteamVR is running and "
                "updated, then retry `uv run osc-grimoire-overlay --data-dir ./data`."
            ) from exc

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
        device_index = self._pointer_device_index()
        if device_index == self.openvr.k_unTrackedDeviceIndexInvalid:
            self._apply_mouse_events(hovering=False, trigger_down=False, position=None)
            return

        state_ok, controller_state = self.vr_system.getControllerState(device_index)
        trigger_down = bool(
            state_ok
            and is_trigger_pressed(
                int(controller_state.ulButtonPressed),
                int(self.openvr.k_EButton_SteamVR_Trigger),
            )
        )
        grip_down = bool(
            state_ok
            and is_button_pressed(
                int(controller_state.ulButtonPressed), int(self.openvr.k_EButton_Grip)
            )
        )
        poses = self._tracked_device_poses()
        if poses is None:
            self.app.controller.status = "Waiting for controller tracking..."
            self._apply_mouse_events(
                hovering=False, trigger_down=trigger_down, position=None
            )
            return
        pose = poses[device_index]
        if not pose.bPoseIsValid:
            self._apply_mouse_events(
                hovering=False, trigger_down=trigger_down, position=None
            )
            self._update_gesture_capture(False, poses, device_index)
            return
        self._update_gesture_capture(grip_down, poses, device_index)
        ray = ray_from_pose(pose)
        intersection = self._compute_intersection(ray)
        if intersection is None:
            self._apply_mouse_events(
                hovering=False, trigger_down=trigger_down, position=None
            )
            return
        self._apply_mouse_events(
            hovering=True,
            trigger_down=trigger_down,
            position=intersection,
        )

    def _update_gesture_capture(
        self, grip_down: bool, poses: Any, pointer_device_index: int
    ) -> None:
        assert self.openvr is not None
        hmd_pose = poses[self.openvr.k_unTrackedDeviceIndex_Hmd]
        pointer_pose = poses[pointer_device_index]
        if grip_down and not self.grip_down:
            if not hmd_pose.bPoseIsValid or not pointer_pose.bPoseIsValid:
                self.app.controller.status = "Waiting for gesture tracking..."
                self.grip_down = grip_down
                return
            self.gesture_sampler.begin(hmd_pose.mDeviceToAbsoluteTracking)
            self.gesture_sampler.add_controller_pose(
                pointer_pose.mDeviceToAbsoluteTracking
            )
            self.app.controller.status = "Recording gesture..."
        elif grip_down and self.gesture_sampler.active and pointer_pose.bPoseIsValid:
            self.gesture_sampler.add_controller_pose(
                pointer_pose.mDeviceToAbsoluteTracking
            )
        elif not grip_down and self.grip_down:
            points = self.gesture_sampler.finish()
            try:
                self.app.controller.handle_gesture_stroke(points)
            except Exception as exc:
                LOGGER.exception("Gesture action failed")
                self.app.controller.status = str(exc)
        self.grip_down = grip_down

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

    def _overlay_device_index(self) -> int:
        assert self.openvr is not None
        assert self.vr_system is not None
        return self._device_index_for_hand(self.config.overlay_hand)

    def _pointer_device_index(self) -> int:
        return self._device_index_for_hand(self.config.pointer_hand)

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
