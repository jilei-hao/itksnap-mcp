#!/usr/bin/env python3
"""Scripted P2 end-to-end: create workspace -> propose (DLS) -> apply (headless) -> open.

This is the "model proposes, human disposes" backbone as a runnable script (the
same steps the MCP tools expose). The workspace is the base for all work; the
``apply`` step edits the workspace segmentation headlessly -- no running ITK-SNAP
required. With ``--open`` the script then launches ITK-SNAP on the workspace so the
human can correct the proposal; that correction is the human-tagged audit record.

Prereqs:
  1. itksnap-wt + ITK-SNAP from an ITK-SNAP build, exported for the config:
       export ITKSNAP_WT_BIN=<build>/Utilities/Workspace/itksnap-wt
       export ITKSNAP_BIN=<build>/ITK-SNAP
  2. DLS server (only for ``propose``):
       conda activate base && cd itksnap-dls && python -m itksnap_dls --port 8911 --device cuda

Usage:
  python demo/run_p2.py --ct <body_ct.nii.gz> [--label <id>] [--itksnap-label 1]
                        [--url http://localhost:8911] [--open]
"""
from __future__ import annotations

import argparse
import json

from itksnap_mcp.config import load_config
from itksnap_mcp.dls_client import DLSClient
from itksnap_mcp.server import (
    propose_segmentation, proposal_summary, write_label_mask, default_workspace_path,
)
from itksnap_mcp.workspace import Workspace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="body CT (NIfTI/NRRD) to segment")
    ap.add_argument("--label", type=int, default=None,
                    help="proposed label id to apply (default: largest structure)")
    ap.add_argument("--itksnap-label", type=int, default=1)
    ap.add_argument("--url", default="http://localhost:8911")
    ap.add_argument("--model", default="TotalSegmentator")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--name", default=None, help="workspace name (default: from CT filename)")
    ap.add_argument("--open", action="store_true",
                    help="launch ITK-SNAP on the workspace for the human correction beat")
    args = ap.parse_args()

    cfg = load_config()
    client = DLSClient(args.url)

    ws_path = default_workspace_path(cfg, args.ct, args.name)
    print(f"[create_workspace] {ws_path}")
    ws = Workspace.create(args.ct, ws_path, cfg.require_wt())
    print(f"[create_workspace] segmentation: {ws.seg_path()}")

    print(f"[propose] {args.model} on {args.ct} ...")
    result, source = propose_segmentation(client, args.ct, args.model, args.fast)
    summary = proposal_summary(result)
    print("[propose] ->", json.dumps(summary, indent=2))
    if not summary["present_labels"]:
        print("[propose] no labels produced; aborting")
        return 1

    label = args.label
    if label is None:
        label = max(summary["present_labels"], key=lambda d: d["voxels"])["id"]
    name = next((d["name"] for d in summary["present_labels"] if d["id"] == label), str(label))
    print(f"[apply] proposed label {label} ({name}) -> workspace label {args.itksnap_label}")

    out = write_label_mask(result, source, label, ws.seg_path() + ".proposal.nii.gz")
    record = ws.apply_mask(out, args.itksnap_label, actor="agent")
    print("[apply] ->", json.dumps(record, indent=2))
    print("[read_audit] ->", json.dumps(ws.last_audit(), indent=2))

    if args.open:
        pid = ws.open_gui(cfg.require_gui(), agent_listen=cfg.agent_sock,
                          launch_prefix=cfg.launch_prefix)
        print(f"\n[open_in_itksnap] launched ITK-SNAP (pid {pid}) on the workspace, "
              f"listening on {cfg.agent_sock}.")
        print("Correct the proposal with the paintbrush, then read_audit again for the "
              "human-tagged correction record.")
    else:
        print("\nRe-run with --open to launch ITK-SNAP on the workspace for the human correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
