"""MCP server exposing the human-in-the-loop segmentation workflow as agent tools.

The P2 flow ("model proposes, human disposes") as callable tools, built around a
**headless ITK-SNAP workspace** as the source of truth -- a live GUI is optional:

  list_models()                 -> available segmentation models
  create_workspace(ct_path)     -> make an .itksnap workspace (main image +
                                   empty segmentation) that all work applies into
  propose(ct_path, ...)         -> run automatic segmentation (DLS/TotalSegmentator),
                                   return the present labels + voxel counts
  apply(label_id, ...)          -> apply one proposed structure into the workspace
                                   segmentation as a committed edit; return the
                                   audit record. No running ITK-SNAP required.
  apply_file(path, ...)         -> apply a mask NIfTI already on disk
  open_in_itksnap(...)          -> launch the ITK-SNAP GUI on the workspace (with a
                                   live --agent-listen socket) so the human can view/correct
  read_audit()                  -> the most recent committed edit's audit record
                                   (live GUI if attached, else the workspace log)
  set_actor(actor)              -> tag the next committed edit (agent | human)

``propose`` talks to the DLS server over HTTP (``dls_client``). ``apply`` edits the
workspace segmentation headlessly via ``itksnap-wt`` + SimpleITK (``workspace``).
The live ``--agent-listen`` socket (``channel.SnapChannel``) is used only when a GUI
is actually attached (read_audit / set_actor for the human correction beat).

See projects/agentic-api/docs/sprint_caimi.md.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from .config import Config, load_config
from .dls_client import DLSClient, AutomaticResult, load_nifti_for_upload
from .channel import SnapChannel, SnapChannelError
from .workspace import Workspace, WorkspaceError


# --- reusable pipeline steps (no MCP dependency) -----------------------------

def propose_segmentation(client: DLSClient, ct_path: str, model_id: str = "TotalSegmentator",
                         fast: bool = True, task: str | None = None):
    """Run automatic segmentation on ``ct_path``. Returns (result, source_sitk_image)."""
    arr_zyx, size_xyz, source = load_nifti_for_upload(ct_path)
    session = client.start_session(model_id)
    try:
        client.upload_image(session, arr_zyx, size_xyz)
        result = client.run_automatic(session, size_xyz, task=task, fast=fast)
    finally:
        client.end_session(session)
    return result, source


def proposal_summary(result: AutomaticResult) -> dict[str, Any]:
    """Present labels with names and voxel counts (what the agent gates on)."""
    import numpy as np
    counts = {int(i): int(np.count_nonzero(result.labels == i)) for i in result.present_label_ids}
    return {
        "shape_zyx": list(result.labels.shape),
        "present_labels": [
            {"id": i, "name": result.label_map.get(i, str(i)), "voxels": counts[i]}
            for i in result.present_label_ids
        ],
    }


def write_label_mask(result: AutomaticResult, source, label_id: int, out_path: str) -> str:
    """Extract a binary mask for ``label_id`` and write it as a NIfTI, restoring the
    source image geometry (the DLS scalar path runs on identity geometry, so the
    proposed labels come back on the uploaded grid -- copy the CT geometry back so
    the mask aligns with the workspace image)."""
    import numpy as np
    import SimpleITK as sitk
    mask = (result.labels == int(label_id)).astype(np.uint8)  # [z, y, x]
    img = sitk.GetImageFromArray(mask)
    if source is not None:
        img.CopyInformation(source)
    sitk.WriteImage(img, out_path)
    return out_path


def default_workspace_path(cfg: Config, ct_path: str, name: str | None = None) -> str:
    """Choose a workspace path under the configured workspace dir."""
    if name:
        stem = name[:-8] if name.endswith(".itksnap") else name
    else:
        stem = os.path.splitext(os.path.splitext(os.path.basename(ct_path))[0])[0]
    return os.path.join(cfg.workspace_dir, stem + ".itksnap")


# --- MCP server --------------------------------------------------------------

def build_server(cfg: Config | None = None):
    """Construct the MCP server. Imports ``mcp`` lazily so the package is importable
    without the optional dependency installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the MCP extra:  pip install 'itksnap-mcp[mcp]'") from e

    cfg = cfg or load_config()
    mcp = FastMCP("itksnap-mcp")
    client = DLSClient(cfg.dls_url)
    # session state: last proposal + the "current" workspace so tools can omit the path
    state: dict[str, Any] = {"result": None, "source": None, "workspace": None}

    def _current_ws(workspace: str | None) -> Workspace:
        path = workspace or state.get("workspace")
        if not path:
            raise RuntimeError("no workspace: call create_workspace(ct_path) first")
        return Workspace(path, cfg.require_wt())

    def _live_channel() -> SnapChannel | None:
        """Return a channel if a live ITK-SNAP is actually reachable, else None."""
        ch = SnapChannel(cfg.agent_sock)
        try:
            ch.ping()
            return ch
        except (SnapChannelError, OSError):
            return None

    @mcp.tool()
    def list_models() -> list[dict]:
        """List available segmentation models and their capabilities."""
        return client.list_models()

    @mcp.tool()
    def create_workspace(ct_path: str, name: str | None = None) -> dict:
        """Create the base ITK-SNAP workspace for a CT: the main image plus an empty
        segmentation layer that ``apply`` edits. Becomes the 'current' workspace.
        Returns the workspace + segmentation paths. No running ITK-SNAP required."""
        path = default_workspace_path(cfg, ct_path, name)
        ws = Workspace.create(ct_path, path, cfg.require_wt())
        state["workspace"] = ws.path
        return {"workspace": ws.path, "segmentation": ws.seg_path(), "source": os.path.abspath(ct_path)}

    @mcp.tool()
    def propose(ct_path: str, model_id: str = "TotalSegmentator",
                fast: bool = True, task: str | None = None) -> dict:
        """Run automatic segmentation on a CT and return the present structures
        (id, anatomy name, voxel count). The result is cached for ``apply``."""
        result, source = propose_segmentation(client, ct_path, model_id, fast, task)
        state["result"] = result
        state["source"] = source
        return proposal_summary(result)

    @mcp.tool()
    def apply(label_id: int, itksnap_label: int = 1, actor: str = "agent",
              workspace: str | None = None) -> dict:
        """Apply one proposed structure (``label_id`` from the last ``propose``) into
        the workspace segmentation as a committed edit under ``itksnap_label``, tagged
        ``actor``. Returns the structured audit record. Headless -- no running ITK-SNAP."""
        if state.get("result") is None:
            raise RuntimeError("call propose() before apply()")
        ws = _current_ws(workspace)
        out = os.path.join(tempfile.gettempdir(), f"itksnap_mcp_proposal_{label_id}.nii.gz")
        write_label_mask(state["result"], state.get("source"), label_id, out)
        return ws.apply_mask(out, itksnap_label, actor=actor)

    @mcp.tool()
    def apply_file(path: str, itksnap_label: int = 1, actor: str = "agent",
                   workspace: str | None = None) -> dict:
        """Apply a segmentation mask NIfTI already on disk (e.g. a cached proposal) into
        the workspace segmentation as a committed edit tagged ``actor``; return the audit
        record. Use this to apply a pre-computed proposal without re-running the model."""
        ws = _current_ws(workspace)
        return ws.apply_mask(path, itksnap_label, actor=actor)

    @mcp.tool()
    def open_in_itksnap(workspace: str | None = None, live: bool = True) -> dict:
        """Launch the ITK-SNAP GUI on the workspace so the human can view/correct the
        agent's proposal. With ``live=True`` also opens the ``--agent-listen`` socket so
        read_audit/set_actor reflect the human's live corrections. Returns the child PID."""
        ws = _current_ws(workspace)
        sock = cfg.agent_sock if live else None
        pid = ws.open_gui(cfg.require_gui(), agent_listen=sock, launch_prefix=cfg.launch_prefix)
        return {"pid": pid, "workspace": ws.path, "agent_listen": sock}

    @mcp.tool()
    def read_audit(workspace: str | None = None) -> dict | None:
        """Return the most recent committed edit's structured audit record: from a live
        ITK-SNAP if one is attached (the human-correction beat), else from the workspace
        audit log (the headless agent applies)."""
        ch = _live_channel()
        if ch is not None:
            live = ch.get_audit()
            if live is not None:
                return live
        return _current_ws(workspace).last_audit()

    @mcp.tool()
    def set_actor(actor: str, workspace: str | None = None) -> None:
        """Declare who is responsible for the next committed edit (agent | human). Arms
        the live ITK-SNAP if attached, and the workspace for the next headless apply."""
        ch = _live_channel()
        if ch is not None:
            ch.set_actor(actor)
        try:
            _current_ws(workspace).set_actor(actor)
        except RuntimeError:
            pass  # no workspace yet; the live arming (if any) still took effect

    return mcp


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
