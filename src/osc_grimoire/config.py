from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    input_device: int | str | None = None


@dataclass(frozen=True)
class VoiceRecognitionConfig:
    n_mfcc: int = 13
    trim_top_db: float = 30.0
    # Drop MFCC C0. It mostly carries broad energy/spectral-envelope level, and
    # in this tiny-vocabulary setup it made short negatives like "yes" look too
    # much like valid one-word spells.
    drop_mfcc_c0: bool = True
    # Normalize each clip's cepstral features independently. This makes DTW
    # compare the shape/dynamics of the utterance more than recording loudness
    # or mic distance.
    cepstral_normalize: bool = True
    # Garbage gate: accept only if best DTW <= ratio_max * intra_class_median(best_spell).
    intra_class_ratio_max: float = 2.5
    # Confusion gate: accept only if (second - best) / second >= margin_min.
    # Skipped when the spellbook has only one spell.
    relative_margin_min: float = 0.20
    # If a spell has no intra_class_median yet (untrained / single sample),
    # fall back to this absolute distance.
    untrained_distance_fallback: float = 600.0


@dataclass(frozen=True)
class GestureRecognitionConfig:
    sample_spacing_m: float = 0.01
    min_points: int = 8
    point_count: int = 32
    score_min: float = 0.20
    margin_min: float = 0.03
    duplicate_distance: float = 0.0


@dataclass(frozen=True)
class OpenVrOverlayConfig:
    overlay_hand: str = "left"
    pointer_hand: str = "right"
    overlay_width_m: float = 0.50
    gesture_trail_width_m: float = 1.0
    gesture_trail_texture_size: int = 512
    texture_width: int = 1000
    texture_height: int = 760
    offset_x: float = 0.06
    offset_y: float = 0.06
    offset_z: float = -0.22


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceRecognitionConfig = field(default_factory=VoiceRecognitionConfig)
    gesture: GestureRecognitionConfig = field(default_factory=GestureRecognitionConfig)
    openvr: OpenVrOverlayConfig = field(default_factory=OpenVrOverlayConfig)
    hotkey: str = "space"
    default_samples_per_spell: int = 5
