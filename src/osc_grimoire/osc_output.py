from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import OscConfig
from .spellbook import (
    OscAction,
    OscPause,
    OscStep,
    Spell,
    format_osc_sequence,
)

LOGGER = logging.getLogger(__name__)
AVATAR_PARAMETER_PREFIX = "/avatar/parameters/"


@dataclass(frozen=True)
class OscTarget:
    host: str
    port: int
    source: str


@dataclass(frozen=True)
class ScheduledOscAction:
    deadline: float
    action: OscAction
    owner_key: str | None


def safe_spell_parameter_suffix(name: str, fallback_id: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    suffix = "".join(word[:1].upper() + word[1:] for word in words)
    if suffix:
        return suffix
    fallback = re.sub(r"[^A-Za-z0-9]", "", fallback_id)
    return fallback[:1].upper() + fallback[1:] if fallback else "Unknown"


def avatar_parameter_path(parameter_name: str) -> str:
    clean = parameter_name.strip("/")
    if clean.startswith("avatar/parameters/"):
        return f"/{clean}"
    return f"{AVATAR_PARAMETER_PREFIX}{clean}"


def spell_osc_parameter_name(spell: Spell, config: OscConfig) -> str:
    suffix = safe_spell_parameter_suffix(spell.name, spell.id)
    return f"{config.parameter_prefix}Spell{suffix}"


def spell_osc_signal_summary(spell: Spell, config: OscConfig) -> str:
    return format_osc_sequence(spell_osc_sequence(spell, config)) or "(nothing)"


def spell_osc_sequence(spell: Spell, config: OscConfig) -> tuple[OscStep, ...]:
    if spell.osc_sequence is not None:
        return spell.osc_sequence
    parameter = spell_osc_parameter_name(spell, config)
    return (
        OscAction(parameter, True),
        OscPause(config.pulse_seconds),
        OscAction(parameter, False),
    )


def fizzle_osc_parameter_name(config: OscConfig) -> str:
    return f"{config.parameter_prefix}Fizzle"


def stance_start_osc_parameter_name(config: OscConfig) -> str:
    return f"{config.parameter_prefix}StanceStart"


def discover_osc_target(config: OscConfig) -> OscTarget:
    try:
        from pythonoscquery.osc_query_browser import OSCQueryBrowser, OSCQueryClient
    except ImportError:
        LOGGER.warning("python-oscquery is unavailable; using OSC fallback target.")
        return _fallback_target(config)

    browser = None
    try:
        browser = OSCQueryBrowser()
        if config.discovery_timeout_seconds > 0.0:
            time.sleep(config.discovery_timeout_seconds)
        services = list(browser.get_discovered_oscquery())
        ranked_services = sorted(
            services,
            key=lambda service: (
                0 if "vrchat" in str(getattr(service, "name", "")).casefold() else 1,
                str(getattr(service, "name", "")),
            ),
        )
        target = select_osc_target_from_services(
            ranked_services, lambda service: OSCQueryClient(service).get_host_info()
        )
        if target is not None:
            return target
    except Exception:
        LOGGER.debug("OSCQuery discovery failed", exc_info=True)
    finally:
        if browser is not None:
            _close_browser(browser)
    return _fallback_target(config)


def select_osc_target_from_services(
    services: list[Any], host_info_for: Callable[[Any], Any | None]
) -> OscTarget | None:
    ranked_services = sorted(
        services,
        key=lambda service: (
            0 if "vrchat" in str(getattr(service, "name", "")).casefold() else 1,
            str(getattr(service, "name", "")),
        ),
    )
    for service in ranked_services:
        try:
            host_info = host_info_for(service)
        except Exception:
            LOGGER.debug("Failed to query OSCQuery service", exc_info=True)
            continue
        if host_info is None:
            continue
        name = str(getattr(host_info, "name", ""))
        transport = str(getattr(host_info, "osc_transport", "UDP") or "UDP")
        host = getattr(host_info, "osc_ip", None)
        port = getattr(host_info, "osc_port", None)
        if transport.upper() != "UDP" or not host or port is None:
            continue
        source = f"OSCQuery {name or getattr(service, 'name', 'unknown')}"
        return OscTarget(str(host), int(port), source)
    return None


def _close_browser(browser: Any) -> None:
    try:
        browser.browser.cancel()
    except Exception:
        LOGGER.debug("Failed to cancel OSCQuery browser", exc_info=True)
    try:
        browser.zc.close()
    except Exception:
        LOGGER.debug("Failed to close OSCQuery zeroconf", exc_info=True)


def _fallback_target(config: OscConfig) -> OscTarget:
    return OscTarget(config.fallback_host, config.fallback_port, "fallback")


class OscOutput:
    def __init__(
        self,
        config: OscConfig,
        *,
        client: Any | None = None,
        target: OscTarget | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.time_fn = time_fn
        self.target = target or discover_osc_target(config)
        self.client = client or self._create_client(self.target)
        self._scheduled_actions: list[ScheduledOscAction] = []

    @property
    def status_text(self) -> str:
        return (
            f"OSC target: {self.target.host}:{self.target.port} ({self.target.source})"
        )

    def send_bool(self, path: str, value: bool) -> None:
        if not self.config.enabled:
            return
        self.client.send_message(avatar_parameter_path(path), bool(value))

    def pulse_bool(self, path: str) -> None:
        self.pulse_sequence(
            (
                OscAction(path, True),
                OscPause(self.config.pulse_seconds),
                OscAction(path, False),
            ),
            owner_key=f"pulse:{avatar_parameter_path(path)}",
        )

    def pulse_sequence(
        self,
        sequence: tuple[OscStep, ...],
        *,
        owner_key: str | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if owner_key is not None:
            self._scheduled_actions = [
                scheduled
                for scheduled in self._scheduled_actions
                if scheduled.owner_key != owner_key
            ]
        cumulative_delay = 0.0
        now = self.time_fn()
        for step in sequence:
            if isinstance(step, OscPause):
                cumulative_delay += step.duration_s
            elif cumulative_delay <= 0.0:
                self.send_action(step)
            else:
                self._scheduled_actions.append(
                    ScheduledOscAction(now + cumulative_delay, step, owner_key)
                )

    def send_action(self, action: OscAction) -> None:
        self.client.send_message(avatar_parameter_path(action.parameter), action.value)

    def tick(self, now: float | None = None) -> None:
        if not self.config.enabled:
            return
        current = self.time_fn() if now is None else now
        pending: list[ScheduledOscAction] = []
        for scheduled in self._scheduled_actions:
            if current + 1e-9 >= scheduled.deadline:
                self.send_action(scheduled.action)
            else:
                pending.append(scheduled)
        self._scheduled_actions = pending

    def set_voice_recording(self, recording: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}VoiceRecording", recording)

    def set_gesture_drawing(self, drawing: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}GestureDrawing", drawing)

    def set_stance_casting(self, casting: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}StanceCasting", casting)

    def set_ui_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}UIEnabled", enabled)

    def set_voice_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}VoiceEnabled", enabled)

    def set_gesture_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}GestureEnabled", enabled)

    def set_stance_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}StanceEnabled", enabled)

    def set_enable_toggles(
        self,
        *,
        ui_enabled: bool,
        gesture_enabled: bool,
        voice_enabled: bool,
        stance_enabled: bool,
    ) -> None:
        self.set_ui_enabled(ui_enabled)
        self.set_gesture_enabled(gesture_enabled)
        self.set_voice_enabled(voice_enabled)
        self.set_stance_enabled(stance_enabled)

    def pulse_stance_start(self) -> None:
        self.pulse_bool(stance_start_osc_parameter_name(self.config))

    def pulse_spell(self, spell: Spell) -> None:
        self.pulse_sequence(
            spell_osc_sequence(spell, self.config),
            owner_key=f"spell:{spell.id}",
        )

    def pulse_fizzle(self) -> None:
        self.pulse_bool(fizzle_osc_parameter_name(self.config))

    @staticmethod
    def _create_client(target: OscTarget) -> Any:
        from pythonosc.udp_client import SimpleUDPClient

        return SimpleUDPClient(target.host, target.port)
