"""Test configuration.

The socket guard is the mechanism that makes "the offline suite costs nothing
and is deterministic" a fact rather than an intention. Without it, a stray
client construction in a future change would quietly start hitting the live
API, and the suite would keep passing while becoming non-deterministic and
billable.
"""

from __future__ import annotations

import socket

import pytest

_real_socket = socket.socket


class BlockedNetwork(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Fail any test that opens a socket, unless it is marked `live`."""
    if request.node.get_closest_marker("live"):
        return

    def blocked(*args, **kwargs):
        raise BlockedNetwork(
            "this test tried to open a network connection. The offline suite "
            "must replay cassettes. If the call is genuinely meant to hit the "
            "API, mark the test with @pytest.mark.live."
        )

    monkeypatch.setattr(socket, "socket", blocked)
