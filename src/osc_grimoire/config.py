from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    input_device: int | str | None = None


@dataclass(frozen=True)
class VoiceRecognitionConfig:
    trim_top_db: float = 30.0
    # Garbage gate: accept only if best forced-incantation distance is low enough.
    voice_alias_distance_max: float = 7.0
    # Confusion gate: accept only if (second - best) / second >= margin_min.
    # Skipped when the spellbook has only one spell.
    relative_margin_min: float = 0.20


@dataclass(frozen=True)
class GestureRecognitionConfig:
    sample_spacing_m: float = 0.01
    min_points: int = 8
    point_count: int = 32
    score_min: float = 0.20
    margin_min: float = 0.03
    duplicate_distance: float = 0.0


@dataclass(frozen=True)
class StanceRecognitionConfig:
    start_hold_s: float = 0.5
    end_hold_s: float = 0.25
    active_timeout_s: float = 1.0
    start_position_tolerance_m: float = 0.20
    start_orientation_tolerance_rad: float = 0.75
    end_position_tolerance_m: float = 0.15
    end_orientation_tolerance_rad: float = 0.55
    record_chord_debounce_s: float = 0.20
    preview_gizmo_size_m: float = 0.10


@dataclass(frozen=True)
class OpenVrOverlayConfig:
    overlay_hand: str = "left"
    pointer_hand: str = "right"
    overlay_width_m: float = 0.50
    draw_gesture_trail_overlay: bool = True
    gesture_trail_width_m: float = 1.0
    gesture_trail_texture_size: int = 512
    texture_width: int = 1000
    texture_height: int = 760
    offset_x: float = 0.06
    offset_y: float = 0.06
    offset_z: float = -0.22


@dataclass(frozen=True)
class OscConfig:
    enabled: bool = True
    service_name: str = "OSC Grimoire"
    parameter_prefix: str = "OSCGrimoire"
    fallback_host: str = "127.0.0.1"
    fallback_port: int = 9000
    pulse_seconds: float = 0.15
    discovery_timeout_seconds: float = 0.5
    input_enabled: bool = True
    input_host: str = "127.0.0.1"
    input_osc_port: int = 0
    input_oscquery_port: int = 0
    input_log_limit: int = 80


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceRecognitionConfig = field(default_factory=VoiceRecognitionConfig)
    gesture: GestureRecognitionConfig = field(default_factory=GestureRecognitionConfig)
    stance: StanceRecognitionConfig = field(default_factory=StanceRecognitionConfig)
    openvr: OpenVrOverlayConfig = field(default_factory=OpenVrOverlayConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    hotkey: str = "space"
