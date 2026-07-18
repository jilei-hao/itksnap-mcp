"""MCP server exposing the human-in-the-loop segmentation workflow as agent tools.

WIP SKELETON (built out in sprint W3). One tool namespace, two backends:

  headless.*  -> Layer-1 / DLS via HTTP (no GUI process)
      - headless.list_models()
      - headless.propose(case)         : run automatic segmentation, return summary + present labels
      - headless.confidence(case)      : run the gate, return auto-accept vs route-to-human
      - headless.commit(case)          : return the structured audit record (needs itksnap C++ audit)

  live.*      -> requires an attached, running ITK-SNAP GUI (STRETCH, gated on the live-channel spike)
      - live.request_human(case, slice): focus the running GUI on the uncertain slice with the proposal loaded
      - live.await_commit(case)        : block until SegmentationChangeEvent, return the audited diff

The DLS calls are already implemented in ``dls_client``; the audit record and the live
command channel are net-new (see projects/agentic-api/docs/sprint_caimi.md).
"""
from __future__ import annotations

from .dls_client import DLSClient
from . import confidence


def build_server(base_url: str = "http://localhost:8911"):
    """Construct the MCP server. Imports ``mcp`` lazily so the package is importable
    without the optional dependency installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the MCP extra:  pip install 'itksnap-mcp[mcp]'") from e

    mcp = FastMCP("itksnap-mcp")
    client = DLSClient(base_url)

    @mcp.tool()
    def headless_list_models() -> list[dict]:
        """List available segmentation models and their capabilities."""
        return client.list_models()

    # TODO(W3): headless_propose / headless_confidence / headless_commit,
    #           then live_request_human / live_await_commit once the C++ audit
    #           record and live command channel land. See sprint_caimi.md.
    return mcp


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
