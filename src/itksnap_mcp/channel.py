"""Client for ITK-SNAP's live ``--agent-listen`` command channel.

ITK-SNAP, launched with ``--agent-listen <socket>``, serves newline-delimited
JSON-RPC over a Unix domain socket on its GUI thread (see ``itksnap`` commit
d9f2329f and follow-ups). This is the *live* half of the human-in-the-loop
workflow: an external agent drives / reads the running GUI the human sees.

Commands (as of sprint W3):
  ping                                   -> {"result": "pong"}
  get_cursor / set_cursor {x,y,z}        -> voxel crosshair
  set_actor {actor: "agent"|"human"}     -> tag the next committed edit
  apply_box {x0,y0,z0,x1,y1,z1,label}    -> paint a labeled box (committed edit)
  apply_seg_file {path,label}            -> apply a proposed mask NIfTI (committed edit)
  get_audit                              -> the last edit's structured audit record

Stdlib only (``socket`` + ``json``) so it has no import cost and can run in any
environment; SimpleITK/numpy are only needed by callers that build masks.
"""
from __future__ import annotations

import json
import socket
from typing import Any


class SnapChannelError(RuntimeError):
    """Raised when the channel returns ``ok: false`` or the socket errors."""


class SnapChannel:
    """Thin request/response client over ITK-SNAP's agent-listen socket."""

    def __init__(self, sock_path: str = "/tmp/snap-agent.sock", timeout: float = 30.0):
        self.sock_path = sock_path
        self.timeout = timeout

    def call(self, cmd: str, args: dict | None = None, *, raise_on_error: bool = True) -> dict[str, Any]:
        """Send one command and return the parsed JSON response."""
        req = json.dumps({"id": 1, "cmd": cmd, "args": args or {}}) + "\n"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.sock_path)
            s.sendall(req.encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        if not buf:
            raise SnapChannelError(f"no response from {self.sock_path} for cmd={cmd!r}")
        resp = json.loads(buf.decode().strip())
        if raise_on_error and not resp.get("ok", False):
            raise SnapChannelError(f"{cmd} failed: {resp.get('error', resp)}")
        return resp

    # --- convenience wrappers -------------------------------------------------
    def ping(self) -> str:
        return self.call("ping")["result"]

    def set_actor(self, actor: str) -> None:
        self.call("set_actor", {"actor": actor})

    def apply_box(self, x0, y0, z0, x1, y1, z1, label) -> dict[str, Any]:
        return self.call("apply_box", {"x0": x0, "y0": y0, "z0": z0,
                                       "x1": x1, "y1": y1, "z1": z1, "label": label})["result"]

    def apply_seg_file(self, path: str, label: int) -> dict[str, Any]:
        return self.call("apply_seg_file", {"path": path, "label": int(label)})["result"]

    def get_audit(self) -> dict | None:
        return self.call("get_audit")["result"]

    def set_cursor(self, x: int, y: int, z: int) -> None:
        self.call("set_cursor", {"x": x, "y": y, "z": z})
