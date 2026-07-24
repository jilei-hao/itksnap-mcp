"""Configuration for itksnap-mcp: where the ITK-SNAP binaries live, where
workspaces are kept, and how to reach the model server / live command socket.

Everything is resolved from the environment with sensible defaults so the MCP
server can run with zero configuration for the headless workspace flow, and be
pointed at real binaries when the user wants the live GUI.

Environment variables
---------------------
  ITKSNAP_WT_BIN        path to the ``itksnap-wt`` workspace CLI (headless
                        workspace create/edit). Falls back to ``itksnap-wt`` on PATH.
  ITKSNAP_BIN           path to the ``ITK-SNAP`` GUI binary (used by
                        ``open_in_itksnap`` to show a workspace to the human).
                        Falls back to ``ITK-SNAP`` / ``itksnap`` on PATH.
  ITKSNAP_LAUNCH_PREFIX optional command prefix for launching the GUI, e.g.
                        ``"xvfb-run -a"`` on a headless box. Space-split.
  ITKSNAP_WORKSPACE_DIR directory where workspaces + their segmentations live.
                        Default: ``<tmp>/itksnap-mcp/workspaces``.
  ITKSNAP_DLS_URL       base URL of the itksnap-dls model server.
                        Default: ``http://localhost:8911``.
  ITKSNAP_AGENT_SOCK    Unix socket a *live* ITK-SNAP listens on (optional).
                        Default: ``/tmp/snap-agent.sock``.
"""
from __future__ import annotations

import os
import shutil
import shlex
import tempfile
from dataclasses import dataclass, field


def _resolve_bin(env_name: str, *names: str) -> str | None:
    """Resolve a binary: explicit env override first, else the first name on PATH."""
    override = os.environ.get(env_name)
    if override:
        return override
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return None


@dataclass
class Config:
    wt_bin: str | None = None            # itksnap-wt (headless workspace engine)
    gui_bin: str | None = None           # ITK-SNAP GUI (optional live view)
    launch_prefix: list[str] = field(default_factory=list)
    workspace_dir: str = ""
    dls_url: str = "http://localhost:8911"
    agent_sock: str = "/tmp/snap-agent.sock"

    def require_wt(self) -> str:
        if not self.wt_bin:
            raise RuntimeError(
                "itksnap-wt not found. Set ITKSNAP_WT_BIN to the workspace-tool "
                "binary (e.g. <build>/Utilities/Workspace/itksnap-wt) or put it on PATH."
            )
        return self.wt_bin

    def require_gui(self) -> str:
        if not self.gui_bin:
            raise RuntimeError(
                "ITK-SNAP GUI binary not found. Set ITKSNAP_BIN to the ITK-SNAP "
                "binary (e.g. <build>/ITK-SNAP) or put it on PATH."
            )
        return self.gui_bin


def load_config() -> Config:
    workspace_dir = os.environ.get(
        "ITKSNAP_WORKSPACE_DIR",
        os.path.join(tempfile.gettempdir(), "itksnap-mcp", "workspaces"),
    )
    prefix = os.environ.get("ITKSNAP_LAUNCH_PREFIX", "")
    return Config(
        wt_bin=_resolve_bin("ITKSNAP_WT_BIN", "itksnap-wt"),
        gui_bin=_resolve_bin("ITKSNAP_BIN", "ITK-SNAP", "itksnap"),
        launch_prefix=shlex.split(prefix) if prefix else [],
        workspace_dir=workspace_dir,
        dls_url=os.environ.get("ITKSNAP_DLS_URL", "http://localhost:8911"),
        agent_sock=os.environ.get("ITKSNAP_AGENT_SOCK", "/tmp/snap-agent.sock"),
    )
