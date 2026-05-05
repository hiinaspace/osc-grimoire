from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import OscConfig
from .spellbook import OscAction, Spell, format_osc_actions

LOGGER = logging.getLogger(__name__)
AVATAR_PARAMETER_PREFIX = "/avatar/parameters/"


@dataclass(frozen=True)
class OscTarget:
    host: str
    port: int
    source: str


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


def avatar_parameter_name(parameter_name: str) -> str:
    clean = parameter_name.strip("/")
    if clean.startswith("avatar/parameters/"):
        return clean.rsplit("/", 1)[-1]
    return clean


def spell_osc_parameter_name(spell: Spell, config: OscConfig) -> str:
    if spell.osc_address and spell.osc_address.strip():
        return avatar_parameter_name(spell.osc_address)
    suffix = safe_spell_parameter_suffix(spell.name, spell.id)
    return f"{config.parameter_prefix}Spell{suffix}"


def spell_osc_signal_summary(spell: Spell, config: OscConfig) -> str:
    on_cast, after_cast = spell_osc_actions(spell, config)
    on_text = format_osc_actions(on_cast) or "(nothing)"
    after_text = format_osc_actions(after_cast)
    if after_text:
        return f"{on_text} -> {after_text}"
    return f"{on_text}"


def spell_osc_actions(
    spell: Spell, config: OscConfig
) -> tuple[tuple[OscAction, ...], tuple[OscAction, ...]]:
    if spell.osc_on_cast is None and spell.osc_after_cast is None:
        parameter = spell_osc_parameter_name(spell, config)
        return (OscAction(parameter, True),), (OscAction(parameter, False),)
    return spell.osc_on_cast or (), spell.osc_after_cast or ()


def fizzle_osc_parameter_name(config: OscConfig) -> str:
    return f"{config.parameter_prefix}Fizzle"


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
        self._scheduled_actions: list[tuple[float, OscAction]] = []

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
        self.pulse_actions(
            (OscAction(path, True),),
            (OscAction(path, False),),
        )

    def pulse_actions(
        self,
        on_cast: tuple[OscAction, ...],
        after_cast: tuple[OscAction, ...] = (),
    ) -> None:
        if not self.config.enabled:
            return
        for action in on_cast:
            self.send_action(action)
        if after_cast:
            after_paths = {
                avatar_parameter_path(action.parameter) for action in after_cast
            }
            self._scheduled_actions = [
                (deadline, action)
                for deadline, action in self._scheduled_actions
                if avatar_parameter_path(action.parameter) not in after_paths
            ]
            deadline = self.time_fn() + self.config.pulse_seconds
            self._scheduled_actions.extend((deadline, action) for action in after_cast)

    def send_action(self, action: OscAction) -> None:
        self.client.send_message(avatar_parameter_path(action.parameter), action.value)

    def tick(self, now: float | None = None) -> None:
        if not self.config.enabled:
            return
        current = self.time_fn() if now is None else now
        pending: list[tuple[float, OscAction]] = []
        for deadline, action in self._scheduled_actions:
            if current >= deadline:
                self.send_action(action)
            else:
                pending.append((deadline, action))
        self._scheduled_actions = pending

    def set_voice_recording(self, recording: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}VoiceRecording", recording)

    def set_gesture_drawing(self, drawing: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}GestureDrawing", drawing)

    def set_ui_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}UIEnabled", enabled)

    def set_voice_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}VoiceEnabled", enabled)

    def set_gesture_enabled(self, enabled: bool) -> None:
        self.send_bool(f"{self.config.parameter_prefix}GestureEnabled", enabled)

    def set_enable_toggles(
        self, *, ui_enabled: bool, gesture_enabled: bool, voice_enabled: bool
    ) -> None:
        self.set_ui_enabled(ui_enabled)
        self.set_gesture_enabled(gesture_enabled)
        self.set_voice_enabled(voice_enabled)

    def pulse_spell(self, spell: Spell) -> None:
        on_cast, after_cast = spell_osc_actions(spell, self.config)
        self.pulse_actions(on_cast, after_cast)

    def pulse_fizzle(self) -> None:
        self.pulse_bool(fizzle_osc_parameter_name(self.config))

    @staticmethod
    def _create_client(target: OscTarget) -> Any:
        from pythonosc.udp_client import SimpleUDPClient

        return SimpleUDPClient(target.host, target.port)
