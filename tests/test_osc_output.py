from __future__ import annotations

from types import SimpleNamespace

from osc_grimoire.config import OscConfig
from osc_grimoire.osc_output import (
    OscOutput,
    OscTarget,
    avatar_parameter_path,
    safe_spell_parameter_suffix,
    select_osc_target_from_services,
    spell_osc_parameter_name,
)
from osc_grimoire.spellbook import Spell


def test_safe_spell_parameter_suffix_cleans_display_name() -> None:
    assert (
        safe_spell_parameter_suffix("alohomora maxima", "spell-1") == "AlohomoraMaxima"
    )
    assert safe_spell_parameter_suffix("lumos!", "spell-1") == "Lumos"
    assert safe_spell_parameter_suffix("!!!", "spell-abc") == "Spellabc"


def test_avatar_parameter_path_accepts_names_or_paths() -> None:
    assert (
        avatar_parameter_path("OSCGrimoireFizzle")
        == "/avatar/parameters/OSCGrimoireFizzle"
    )
    assert (
        avatar_parameter_path("/avatar/parameters/OSCGrimoireFizzle")
        == "/avatar/parameters/OSCGrimoireFizzle"
    )


def test_spell_osc_parameter_uses_override_or_default() -> None:
    config = OscConfig(parameter_prefix="OSCGrimoire")

    assert (
        spell_osc_parameter_name(Spell(id="spell-1", name="Lumos!"), config)
        == "OSCGrimoireSpellLumos"
    )
    assert (
        spell_osc_parameter_name(
            Spell(id="spell-1", name="Lumos!", osc_address="CustomFire"),
            config,
        )
        == "CustomFire"
    )
    assert (
        spell_osc_parameter_name(
            Spell(
                id="spell-1",
                name="Lumos!",
                osc_address="/avatar/parameters/CustomFire",
            ),
            config,
        )
        == "CustomFire"
    )


def test_select_osc_target_prefers_vrchat_udp_service() -> None:
    services = [
        SimpleNamespace(name="Other._oscjson._tcp.local."),
        SimpleNamespace(name="VRChat-Client._oscjson._tcp.local."),
    ]
    host_infos = {
        id(services[0]): SimpleNamespace(
            name="Other", osc_ip="127.0.0.2", osc_port=9100, osc_transport="UDP"
        ),
        id(services[1]): SimpleNamespace(
            name="VRChat", osc_ip="127.0.0.1", osc_port=9000, osc_transport="UDP"
        ),
    }

    target = select_osc_target_from_services(
        services, lambda service: host_infos[id(service)]
    )

    assert target == OscTarget("127.0.0.1", 9000, "OSCQuery VRChat")


def test_select_osc_target_ignores_non_udp_services() -> None:
    services = [SimpleNamespace(name="VRChat._oscjson._tcp.local.")]
    host_infos = {
        id(services[0]): SimpleNamespace(
            name="VRChat", osc_ip="127.0.0.1", osc_port=9000, osc_transport="TCP"
        )
    }

    assert (
        select_osc_target_from_services(
            services, lambda service: host_infos[id(service)]
        )
        is None
    )


def test_osc_output_sends_recording_pulses_and_resets() -> None:
    client = _FakeOscClient()
    clock = _Clock()
    output = OscOutput(
        OscConfig(pulse_seconds=0.15),
        client=client,
        target=OscTarget("127.0.0.1", 9000, "test"),
        time_fn=clock.now,
    )

    output.set_voice_recording(True)
    output.set_gesture_drawing(True)
    output.pulse_spell(Spell(id="spell-1", name="Lumos!", osc_address="CustomFire"))
    output.pulse_fizzle()
    clock.value = 0.20
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/OSCGrimoireVoiceRecording", True),
        ("/avatar/parameters/OSCGrimoireGestureDrawing", True),
        ("/avatar/parameters/CustomFire", True),
        ("/avatar/parameters/OSCGrimoireFizzle", True),
        ("/avatar/parameters/CustomFire", False),
        ("/avatar/parameters/OSCGrimoireFizzle", False),
    ]


class _Clock:
    value = 0.0

    def now(self) -> float:
        return self.value


class _FakeOscClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def send_message(self, path: str, value: bool) -> None:
        self.messages.append((path, value))
