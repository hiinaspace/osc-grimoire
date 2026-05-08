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
    spell_osc_signal_summary,
)
from osc_grimoire.spellbook import OscAction, OscPause, Spell


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


def test_spell_osc_parameter_uses_default_name() -> None:
    config = OscConfig(parameter_prefix="OSCGrimoire")

    assert (
        spell_osc_parameter_name(Spell(id="spell-1", name="Lumos!"), config)
        == "OSCGrimoireSpellLumos"
    )


def test_spell_osc_signal_summary_describes_default_or_custom_sequence() -> None:
    config = OscConfig(parameter_prefix="OSCGrimoire")

    assert (
        spell_osc_signal_summary(Spell(id="spell-1", name="Lumos"), config)
        == "OSCGrimoireSpellLumos=true, (Pause 150ms), OSCGrimoireSpellLumos=false"
    )
    assert (
        spell_osc_signal_summary(
            Spell(
                id="spell-1",
                name="Lumos",
                osc_sequence=(
                    OscAction("Spell", 4),
                    OscAction("MagicPrepared", True),
                ),
            ),
            config,
        )
        == "Spell=4, MagicPrepared=true"
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
    output.set_stance_casting(True)
    output.set_enable_toggles(
        ui_enabled=False,
        gesture_enabled=True,
        voice_enabled=False,
        stance_enabled=True,
    )
    output.pulse_stance_start()
    output.pulse_spell(
        Spell(
            id="spell-1",
            name="Lumos!",
            osc_sequence=(
                OscAction("CustomFire", True),
                OscPause(0.15),
                OscAction("CustomFire", False),
            ),
        )
    )
    output.pulse_fizzle()
    clock.value = 0.20
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/OSCGrimoireVoiceRecording", True),
        ("/avatar/parameters/OSCGrimoireGestureDrawing", True),
        ("/avatar/parameters/OSCGrimoireStanceCasting", True),
        ("/avatar/parameters/OSCGrimoireUIEnabled", False),
        ("/avatar/parameters/OSCGrimoireGestureEnabled", True),
        ("/avatar/parameters/OSCGrimoireVoiceEnabled", False),
        ("/avatar/parameters/OSCGrimoireStanceEnabled", True),
        ("/avatar/parameters/OSCGrimoireStanceStart", True),
        ("/avatar/parameters/CustomFire", True),
        ("/avatar/parameters/OSCGrimoireFizzle", True),
        ("/avatar/parameters/OSCGrimoireStanceStart", False),
        ("/avatar/parameters/CustomFire", False),
        ("/avatar/parameters/OSCGrimoireFizzle", False),
    ]


def test_osc_output_recasting_same_spell_cancels_pending_sequence() -> None:
    client = _FakeOscClient()
    clock = _Clock()
    output = OscOutput(
        OscConfig(pulse_seconds=0.15),
        client=client,
        target=OscTarget("127.0.0.1", 9000, "test"),
        time_fn=clock.now,
    )
    spell = Spell(
        id="spell-1",
        name="Lumos",
        osc_sequence=(
            OscAction("CustomFire", True),
            OscPause(0.15),
            OscAction("CustomFire", False),
        ),
    )

    output.pulse_spell(spell)
    clock.value = 0.10
    output.pulse_spell(spell)
    clock.value = 0.16
    output.tick()
    clock.value = 0.26
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/CustomFire", True),
        ("/avatar/parameters/CustomFire", True),
        ("/avatar/parameters/CustomFire", False),
    ]


def test_osc_output_can_send_custom_spell_actions_without_end_actions() -> None:
    client = _FakeOscClient()
    clock = _Clock()
    output = OscOutput(
        OscConfig(pulse_seconds=0.15),
        client=client,
        target=OscTarget("127.0.0.1", 9000, "test"),
        time_fn=clock.now,
    )

    output.pulse_spell(
        Spell(
            id="spell-1",
            name="Lumos",
            osc_sequence=(
                OscAction("Spell", 7),
                OscAction("MagicPrepared", True),
            ),
        )
    )
    clock.value = 0.20
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/Spell", 7),
        ("/avatar/parameters/MagicPrepared", True),
    ]


def test_osc_output_schedules_sequence_pauses_cumulatively() -> None:
    client = _FakeOscClient()
    clock = _Clock()
    output = OscOutput(
        OscConfig(pulse_seconds=0.15),
        client=client,
        target=OscTarget("127.0.0.1", 9000, "test"),
        time_fn=clock.now,
    )
    spell = Spell(
        id="spell-1",
        name="Lumos",
        osc_sequence=(
            OscAction("A", 1),
            OscPause(0.10),
            OscAction("B", 2),
            OscPause(0.20),
            OscAction("C", 3),
        ),
    )

    output.pulse_spell(spell)
    clock.value = 0.10
    output.tick()
    clock.value = 0.30
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/A", 1),
        ("/avatar/parameters/B", 2),
        ("/avatar/parameters/C", 3),
    ]


def test_osc_output_keeps_different_spell_sequences_independent() -> None:
    client = _FakeOscClient()
    clock = _Clock()
    output = OscOutput(
        OscConfig(pulse_seconds=0.15),
        client=client,
        target=OscTarget("127.0.0.1", 9000, "test"),
        time_fn=clock.now,
    )
    first = Spell(
        id="spell-1",
        name="Lumos",
        osc_sequence=(
            OscAction("First", True),
            OscPause(0.15),
            OscAction("First", False),
        ),
    )
    second = Spell(
        id="spell-2",
        name="Nox",
        osc_sequence=(
            OscAction("Second", True),
            OscPause(0.15),
            OscAction("Second", False),
        ),
    )

    output.pulse_spell(first)
    clock.value = 0.05
    output.pulse_spell(second)
    clock.value = 0.16
    output.tick()
    clock.value = 0.21
    output.tick()

    assert client.messages == [
        ("/avatar/parameters/First", True),
        ("/avatar/parameters/Second", True),
        ("/avatar/parameters/First", False),
        ("/avatar/parameters/Second", False),
    ]


class _Clock:
    value = 0.0

    def now(self) -> float:
        return self.value


class _FakeOscClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []

    def send_message(self, path: str, value: object) -> None:
        self.messages.append((path, value))
