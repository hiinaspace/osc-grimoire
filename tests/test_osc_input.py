from __future__ import annotations

from osc_grimoire.config import OscConfig
from osc_grimoire.osc_input import (
    OscInputService,
    advertised_input_paths,
    format_recent_osc_messages,
)


def test_advertised_input_paths_match_vrchat_send_roots() -> None:
    assert advertised_input_paths() == ("/avatar", "/tracking/vrsystem")


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
