"""MCP server exposing the human-in-the-loop segmentation workflow as agent tools.

The P2 flow ("model proposes, human disposes") as callable tools:

  list_models()                 -> available segmentation models
  propose(ct_path, ...)         -> run automatic segmentation (DLS/TotalSegmentator),
                                   return the present labels + voxel counts
  apply(label_id, ...)          -> apply one proposed structure into the running
                                   ITK-SNAP as a committed edit; return the audit record
  read_audit()                  -> the most recent committed edit's audit record
  set_actor(actor)              -> tag the next committed edit (agent | human)

``propose`` talks to the DLS server over HTTP (``dls_client``); ``apply`` / ``read_audit``
/ ``set_actor`` drive the running ITK-SNAP over its ``--agent-listen`` socket
(``channel.SnapChannel``). The module-level helpers below are reused by
``demo/run_p2.py`` for a scripted end-to-end run.

See projects/agentic-api/docs/sprint_caimi.md (W3).
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from .dls_client import DLSClient, AutomaticResult, load_nifti_for_upload
from .channel import SnapChannel


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
    the mask aligns with the volume loaded in ITK-SNAP)."""
    import numpy as np
    import SimpleITK as sitk
    mask = (result.labels == int(label_id)).astype(np.uint8)  # [z, y, x]
    img = sitk.GetImageFromArray(mask)
    if source is not None:
        img.CopyInformation(source)
    sitk.WriteImage(img, out_path)
    return out_path


# --- MCP server --------------------------------------------------------------

def build_server(base_url: str = "http://localhost:8911",
                 sock_path: str = "/tmp/snap-agent.sock"):
    """Construct the MCP server. Imports ``mcp`` lazily so the package is importable
    without the optional dependency installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the MCP extra:  pip install 'itksnap-mcp[mcp]'") from e

    mcp = FastMCP("itksnap-mcp")
    client = DLSClient(base_url)
    state: dict[str, Any] = {}  # last proposal: {"result", "source"}

    @mcp.tool()
    def list_models() -> list[dict]:
        """List available segmentation models and their capabilities."""
        return client.list_models()

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
    def apply(label_id: int, itksnap_label: int = 1, actor: str = "agent") -> dict:
        """Apply one proposed structure (``label_id`` from the last ``propose``) into
        the running ITK-SNAP as a committed edit under ``itksnap_label``, tagged with
        ``actor``. Returns the structured audit record for the applied proposal."""
        if "result" not in state:
            raise RuntimeError("call propose() before apply()")
        out = os.path.join(tempfile.gettempdir(), f"itksnap_mcp_proposal_{label_id}.nii.gz")
        write_label_mask(state["result"], state.get("source"), label_id, out)
        ch = SnapChannel(sock_path)
        ch.set_actor(actor)
        return ch.apply_seg_file(out, itksnap_label)

    @mcp.tool()
    def apply_file(path: str, itksnap_label: int = 1, actor: str = "agent") -> dict:
        """Apply a segmentation mask NIfTI already on disk (e.g. a cached proposal) into the
        running ITK-SNAP as a committed edit tagged ``actor``; return the audit record. Use
        this to apply a pre-computed proposal without re-running the model."""
        ch = SnapChannel(sock_path)
        ch.set_actor(actor)
        return ch.apply_seg_file(path, itksnap_label)

    @mcp.tool()
    def read_audit() -> dict | None:
        """Return the most recent committed edit's structured audit record (or null)."""
        return SnapChannel(sock_path).get_audit()

    @mcp.tool()
    def set_actor(actor: str) -> None:
        """Declare who is responsible for the next committed edit (agent | human)."""
        SnapChannel(sock_path).set_actor(actor)

    return mcp


if __name__ == "__main__":  # pragma: no cover
    import os
    build_server(
        base_url=os.environ.get("ITKSNAP_DLS_URL", "http://localhost:8911"),
        sock_path=os.environ.get("ITKSNAP_AGENT_SOCK", "/tmp/snap-agent.sock"),
    ).run()
