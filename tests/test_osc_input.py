from __future__ import annotations

import pytest

from osc_grimoire.config import OscConfig
from osc_grimoire.osc_input import (
    OscInputService,
    advertised_input_paths,
    find_oscquery_node,
    format_recent_osc_messages,
    oscquery_host_info,
    oscquery_tree,
)


def test_advertised_input_paths_match_vrchat_send_roots() -> None:
    assert advertised_input_paths() == (
        "/avatar/parameters/OSCGrimoireUIEnabled",
        "/avatar/parameters/OSCGrimoireGestureEnabled",
        "/avatar/parameters/OSCGrimoireVoiceEnabled",
    )


def test_oscquery_tree_matches_vrcadvert_shape() -> None:
    tree = oscquery_tree()

    avatar = find_oscquery_node(tree, "/avatar/parameters/OSCGrimoireUIEnabled")

    assert avatar is not None
    assert avatar["ACCESS"] == 2
    assert avatar["TYPE"] == "T"
    assert find_oscquery_node(tree, "/tracking/vrsystem/head/pose") is None


def test_oscquery_host_info_advertises_udp_target() -> None:
    host_info = oscquery_host_info(OscConfig(), 4567)

    assert host_info["OSC_IP"] == "127.0.0.1"
    assert host_info["OSC_PORT"] == 4567
    assert host_info["OSC_TRANSPORT"] == "UDP"


def test_osc_input_service_records_relevant_parameters_only() -> None:
    service = OscInputService(
        OscConfig(input_log_limit=2),
        time_fn=lambda: 123.0,
    )

    service._handle_message("/avatar/parameters/Foo", True)
    service._handle_message("/avatar/parameters/OSCGrimoireUIEnabled", False)
    service._handle_message("/avatar/parameters/OSCGrimoireGestureEnabled", 0)
    service._handle_message("/avatar/parameters/OSCGrimoireVoiceEnabled", 1.0)
    service._handle_message("/tracking/vrsystem/head", 1.0, 2.0, 3.0)

    messages = service.recent_messages()
    assert [message.address for message in messages] == [
        "/avatar/parameters/OSCGrimoireGestureEnabled",
        "/avatar/parameters/OSCGrimoireVoiceEnabled",
    ]
    assert not service.ui_enabled
    assert not service.gesture_enabled
    assert service.voice_enabled
    assert "OSCGrimoireVoiceEnabled" in format_recent_osc_messages(messages)


def test_osc_input_avatar_change_invokes_resync_callback() -> None:
    callback_calls: list[bool] = []
    service = OscInputService(
        OscConfig(),
        on_avatar_change=lambda: callback_calls.append(True),
    )

    service._handle_message("/avatar/change", "avatar-id")

    assert callback_calls == [True]
    assert service.recent_messages() == ()


def test_osc_input_set_enabled_state_mirrors_outgoing_state() -> None:
    service = OscInputService(OscConfig())
    service._handle_message("/avatar/parameters/OSCGrimoireVoiceEnabled", False)

    service.set_enabled_state(ui_enabled=False, voice_enabled=True)

    assert not service.ui_enabled
    assert service.gesture_enabled
    assert service.voice_enabled


def test_osc_input_disabled_does_not_start() -> None:
    service = OscInputService(OscConfig(input_enabled=False))

    service.start()

    assert service.status_text == "OSC input: disabled"
    assert service.ports is None


def test_osc_input_start_failure_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    service = OscInputService(OscConfig())

    def fail_start() -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(service, "_start", fail_start)

    service.start()

    assert service.status_text == "OSC input: unavailable"
    assert service.ports is None
