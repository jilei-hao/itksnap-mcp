#!/usr/bin/env python
"""Send one JSON-RPC command to a running ITK-SNAP `--agent-listen` socket.

    python demo/agent_send.py /tmp/snap-agent.sock ping
    python demo/agent_send.py /tmp/snap-agent.sock get_cursor
    python demo/agent_send.py /tmp/snap-agent.sock set_cursor 30 40 10

Handy for manually driving the live GUI channel (Gate-2 prototype). Stdlib only.
"""
import json
import socket
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    sock_path, cmd = sys.argv[1], sys.argv[2]
    args = {}
    if cmd == "set_cursor":
        args = {"x": int(sys.argv[3]), "y": int(sys.argv[4]), "z": int(sys.argv[5])}
    req = {"id": 1, "cmd": cmd, "args": args}

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    print(buf.decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
