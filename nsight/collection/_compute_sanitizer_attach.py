# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait for a self-injected sanitizer backend and exec its attach frontend."""

import os
import sys
import time
from pathlib import Path
from typing import NoReturn

_TCP_LISTEN_STATE = "0A"
_POLL_INTERVAL_SECONDS = 0.01


def _parent_is_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_is_listening(port: int) -> bool:
    """Return whether Linux procfs reports a TCP listener on ``port``.

    Linux exposes IPv4 and IPv6 socket tables in ``/proc/net/tcp`` and
    ``/proc/net/tcp6``. After one header line, each row describes a socket. The
    second whitespace-separated field is its local address in the form
    ``HEX_ADDRESS:HEX_PORT`` and the fourth field is its TCP state. State
    ``0A`` means ``TCP_LISTEN``.

    Only the local port and state matter here: the sanitizer backend may bind
    either address family and any local interface. ``port`` is therefore
    formatted as the four-digit hexadecimal value used by procfs and compared
    with the portion following the final colon in the local-address field.

    This function deliberately reads procfs instead of probing the port with
    ``connect()``. The backend accepts a single analyzer connection, so a probe
    connection could consume that slot before the real ``compute-sanitizer
    --mode attach`` frontend reaches it.
    """
    expected_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4:
                continue
            local_address = fields[1]
            state = fields[3]
            if (
                state == _TCP_LISTEN_STATE
                and local_address.rsplit(":", 1)[-1] == expected_port
            ):
                return True
    return False


def _exec_frontend(command: list[str]) -> NoReturn:
    os.execv(command[0], command)
    raise AssertionError("os.execv unexpectedly returned")


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: _compute_sanitizer_attach.py PARENT_PID PORT COMMAND...",
            file=sys.stderr,
        )
        return 2

    parent_pid = int(sys.argv[1])
    port = int(sys.argv[2])
    command = sys.argv[3:]

    while _parent_is_alive(parent_pid):
        if _port_is_listening(port):
            _exec_frontend(command)
        time.sleep(_POLL_INTERVAL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
