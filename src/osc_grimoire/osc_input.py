from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from zeroconf import ServiceInfo, Zeroconf

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
        self._oscquery_http_server: _OscQueryHttpServer | None = None
        self._oscquery_http_thread: threading.Thread | None = None
        self._zeroconf: Zeroconf | None = None
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
        if self._zeroconf is not None:
            self._zeroconf.unregister_all_services()
            self._zeroconf.close()
            self._zeroconf = None
        if self._oscquery_http_server is not None:
            self._oscquery_http_server.shutdown()
            self._oscquery_http_server.server_close()
            self._oscquery_http_server = None
        if self._oscquery_http_thread is not None:
            self._oscquery_http_thread.join(timeout=1.0)
            self._oscquery_http_thread = None
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
        host_info = oscquery_host_info(self.config, osc_port)
        tree = oscquery_tree()
        server = _OscQueryHttpServer(
            (self.config.input_host, oscquery_port),
            _OscQueryHttpHandler,
            host_info=host_info,
            tree=tree,
        )
        self._oscquery_http_server = server
        self._oscquery_http_thread = threading.Thread(
            target=server.serve_forever,
            name="osc-grimoire-oscquery-input",
            daemon=True,
        )
        self._oscquery_http_thread.start()
        self._zeroconf = Zeroconf()
        self._zeroconf.register_service(
            _service_info(
                "_oscjson._tcp.local.",
                f"{self.config.service_name}._oscjson._tcp.local.",
                self.config.service_name,
                oscquery_port,
                self.config.input_host,
            ),
            allow_name_change=True,
        )
        self._zeroconf.register_service(
            _service_info(
                "_osc._udp.local.",
                f"{self.config.service_name}._osc._udp.local.",
                self.config.service_name,
                osc_port,
                self.config.input_host,
            ),
            allow_name_change=True,
        )


class _OscQueryHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        host_info: dict[str, Any],
        tree: dict[str, Any],
    ) -> None:
        super().__init__(server_address, handler_class)
        self.host_info = host_info
        self.tree = tree


class _OscQueryHttpHandler(BaseHTTPRequestHandler):
    server: _OscQueryHttpServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        queries = parse_qs(parsed.query, keep_blank_values=True)
        if "HOST_INFO" in queries:
            self._respond_json(self.server.host_info)
            return

        node = find_oscquery_node(self.server.tree, parsed.path or "/")
        if node is None:
            self.send_error(404, "OSC Path not found")
            return
        if queries:
            attribute = next(iter(queries)).upper()
            if attribute not in node:
                self.send_response(204)
                self.end_headers()
                return
            self._respond_json({attribute: node[attribute]})
            return
        self._respond_json(node)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("OSCQuery HTTP " + format, *args)

    def _respond_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def advertised_input_paths() -> tuple[str, ...]:
    return (
        "/avatar/parameters",
        "/tracking/vrsystem/head/pose",
        "/tracking/vrsystem/leftwrist/pose",
        "/tracking/vrsystem/rightwrist/pose",
    )


def oscquery_host_info(config: OscConfig, osc_port: int) -> dict[str, Any]:
    return {
        "NAME": config.service_name,
        "OSC_IP": config.input_host,
        "OSC_PORT": osc_port,
        "OSC_TRANSPORT": "UDP",
        "EXTENSIONS": {
            "ACCESS": True,
            "CLIPMODE": False,
            "RANGE": False,
            "TYPE": True,
            "VALUE": True,
        },
    }


def oscquery_tree() -> dict[str, Any]:
    return {
        "FULL_PATH": "/",
        "ACCESS": 0,
        "DESCRIPTION": "root node",
        "CONTENTS": {
            "avatar": {
                "FULL_PATH": "/avatar",
                "ACCESS": 0,
                "CONTENTS": {
                    "parameters": {
                        "FULL_PATH": "/avatar/parameters",
                        "ACCESS": 2,
                        "TYPE": "b",
                        "DESCRIPTION": "Receive VRChat avatar parameters",
                    }
                },
            },
            "tracking": {
                "FULL_PATH": "/tracking",
                "ACCESS": 0,
                "CONTENTS": {
                    "vrsystem": {
                        "FULL_PATH": "/tracking/vrsystem",
                        "ACCESS": 0,
                        "CONTENTS": {
                            "head": _pose_container(
                                "/tracking/vrsystem/head",
                                "/tracking/vrsystem/head/pose",
                            ),
                            "leftwrist": _pose_container(
                                "/tracking/vrsystem/leftwrist",
                                "/tracking/vrsystem/leftwrist/pose",
                            ),
                            "rightwrist": _pose_container(
                                "/tracking/vrsystem/rightwrist",
                                "/tracking/vrsystem/rightwrist/pose",
                            ),
                        },
                    }
                },
            },
        },
    }


def _pose_container(container_path: str, pose_path: str) -> dict[str, Any]:
    return {
        "FULL_PATH": container_path,
        "ACCESS": 0,
        "CONTENTS": {
            "pose": {
                "FULL_PATH": pose_path,
                "ACCESS": 2,
                "TYPE": "ffffff",
                "DESCRIPTION": "Receive VRChat tracking pose",
            }
        },
    }


def find_oscquery_node(tree: dict[str, Any], path: str) -> dict[str, Any] | None:
    if path == tree.get("FULL_PATH"):
        return tree
    contents = tree.get("CONTENTS")
    if not isinstance(contents, dict):
        return None
    for child in contents.values():
        found = find_oscquery_node(child, path)
        if found is not None:
            return found
    return None


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


def _service_info(
    service_type: str, service_name: str, server_name: str, port: int, host: str
) -> ServiceInfo:
    return ServiceInfo(
        service_type,
        service_name,
        port=port,
        properties={"txtvers": "1"},
        server=f"{server_name}.local.",
        parsed_addresses=[host],
    )


def format_recent_osc_messages(messages: Sequence[ReceivedOscMessage]) -> str:
    if not messages:
        return "(none)"
    return "\n".join(message.format() for message in messages)
