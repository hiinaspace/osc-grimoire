from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from .config import OscConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReceivedOscMessage:
    timestamp: float
    address: str
    values: tuple[Any, ...]

    def format(self) -> str:
        value_text = ", ".join(repr(value) for value in self.values)
        return f"{self.timestamp:10.3f} {self.address} [{value_text}]"


@dataclass(frozen=True)
class OscInputPorts:
    osc_port: int
    oscquery_port: int


class OscInputService:
    def __init__(
        self,
        config: OscConfig,
        *,
        time_fn=time.monotonic,
    ) -> None:
        self.config = config
        self.time_fn = time_fn
        self._messages: deque[ReceivedOscMessage] = deque(
            maxlen=max(1, config.input_log_limit)
        )
        self._lock = threading.Lock()
        self._osc_server: ThreadingOSCUDPServer | None = None
        self._osc_thread: threading.Thread | None = None
        self._oscquery_service: Any | None = None
        self._ports: OscInputPorts | None = None
        self.status_text = "OSC input: stopped"

    @property
    def ports(self) -> OscInputPorts | None:
        return self._ports

    def start(self) -> None:
        if not self.config.enabled or not self.config.input_enabled:
            self.status_text = "OSC input: disabled"
            return
        if self._osc_server is not None:
            return

        ports = _resolve_input_ports(self.config)
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._handle_message, needs_reply_address=False)
        server = ThreadingOSCUDPServer(
            (self.config.input_host, ports.osc_port), dispatcher
        )
        self._osc_server = server
        actual_osc_port = int(server.server_address[1])
        self._ports = OscInputPorts(actual_osc_port, ports.oscquery_port)
        self._osc_thread = threading.Thread(
            target=server.serve_forever,
            name="osc-grimoire-osc-input",
            daemon=True,
        )
        self._osc_thread.start()
        self._start_oscquery(actual_osc_port, ports.oscquery_port)
        self.status_text = (
            f"OSC input: {self.config.input_host}:{actual_osc_port}, "
            f"OSCQuery :{ports.oscquery_port}"
        )
        LOGGER.info("%s", self.status_text)

    def stop(self) -> None:
        if self._oscquery_service is not None:
            self._oscquery_service.stop()
            self._oscquery_service = None
        if self._osc_server is not None:
            self._osc_server.shutdown()
            self._osc_server.server_close()
            self._osc_server = None
        if self._osc_thread is not None:
            self._osc_thread.join(timeout=1.0)
            self._osc_thread = None
        self.status_text = "OSC input: stopped"

    def recent_messages(self) -> tuple[ReceivedOscMessage, ...]:
        with self._lock:
            return tuple(self._messages)

    def _handle_message(self, address: str, *values: Any) -> None:
        message = ReceivedOscMessage(
            timestamp=self.time_fn(),
            address=address,
            values=tuple(values),
        )
        with self._lock:
            self._messages.append(message)
        LOGGER.info("OSC input %s", message.format())

    def _start_oscquery(self, osc_port: int, oscquery_port: int) -> None:
        try:
            from pythonoscquery.osc_query_service import (
                OSCAccess,
                OSCAddressSpace,
                OSCPathNode,
                OSCQueryService,
            )
        except ImportError:
            LOGGER.warning(
                "python-oscquery is unavailable; OSC input is not advertised."
            )
            return

        address_space = OSCAddressSpace()
        for path in advertised_input_paths():
            address_space.add_node(
                OSCPathNode(
                    path,
                    access=OSCAccess.NO_VALUE,
                    description="OSC Grimoire receive endpoint",
                )
            )
        service = OSCQueryService(
            address_space,
            self.config.service_name,
            oscquery_port,
            osc_port,
            self.config.input_host,
        )
        service.start()
        self._oscquery_service = service


def advertised_input_paths() -> tuple[str, ...]:
    return ("/avatar", "/tracking/vrsystem")


def _resolve_input_ports(config: OscConfig) -> OscInputPorts:
    osc_port = config.input_osc_port or _free_udp_port(config.input_host)
    oscquery_port = config.input_oscquery_port or _free_tcp_port(config.input_host)
    return OscInputPorts(osc_port, oscquery_port)


def _free_udp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def format_recent_osc_messages(messages: Sequence[ReceivedOscMessage]) -> str:
    if not messages:
        return "(none)"
    return "\n".join(message.format() for message in messages)
