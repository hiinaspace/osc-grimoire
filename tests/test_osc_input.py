from __future__ import annotations

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
        "/avatar/parameters",
        "/tracking/vrsystem/head/pose",
        "/tracking/vrsystem/leftwrist/pose",
        "/tracking/vrsystem/rightwrist/pose",
    )


def test_oscquery_tree_matches_vrcadvert_shape() -> None:
    tree = oscquery_tree()

    avatar = find_oscquery_node(tree, "/avatar/parameters")
    head = find_oscquery_node(tree, "/tracking/vrsystem/head/pose")

    assert avatar is not None
    assert avatar["ACCESS"] == 2
    assert avatar["TYPE"] == "b"
    assert head is not None
    assert head["ACCESS"] == 2
    assert head["TYPE"] == "ffffff"


def test_oscquery_host_info_advertises_udp_target() -> None:
    host_info = oscquery_host_info(OscConfig(), 4567)

    assert host_info["OSC_IP"] == "127.0.0.1"
    assert host_info["OSC_PORT"] == 4567
    assert host_info["OSC_TRANSPORT"] == "UDP"


def test_osc_input_service_records_all_messages() -> None:
    service = OscInputService(
        OscConfig(input_log_limit=2),
        time_fn=lambda: 123.0,
    )

    service._handle_message("/avatar/parameters/Foo", True)
    service._handle_message("/tracking/vrsystem/head", 1.0, 2.0, 3.0)
    service._handle_message("/avatar/change", "avatar-id")

    messages = service.recent_messages()
    assert [message.address for message in messages] == [
        "/tracking/vrsystem/head",
        "/avatar/change",
    ]
    assert messages[-1].values == ("avatar-id",)
    assert "/avatar/change" in format_recent_osc_messages(messages)


def test_osc_input_disabled_does_not_start() -> None:
    service = OscInputService(OscConfig(input_enabled=False))

    service.start()

    assert service.status_text == "OSC input: disabled"
    assert service.ports is None
