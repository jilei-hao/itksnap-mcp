#!/usr/bin/env python3
"""Scripted P2 end-to-end: propose (DLS) -> apply into live ITK-SNAP -> read audit.

This is the "model proposes, human disposes" backbone as a runnable script (the
same steps the MCP tools expose). The human-correction beat happens in the GUI
between ``apply`` and a second ``read_audit``.

Prereqs:
  1. DLS server:  conda activate base && cd itksnap-dls && \
                    python -m itksnap_dls --port 8911 --device cuda
  2. ITK-SNAP:    ITK-SNAP -g <ct> --agent-listen /tmp/snap-agent.sock
                    (wrap in ``xvfb-run -a`` / a headless Xvfb if no display)

Usage:
  python demo/run_p2.py --ct <body_ct.nii.gz> [--label <id>] [--itksnap-label 1]
                        [--url http://localhost:8911] [--sock /tmp/snap-agent.sock]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from itksnap_mcp.channel import SnapChannel
from itksnap_mcp.dls_client import DLSClient
from itksnap_mcp.server import propose_segmentation, proposal_summary, write_label_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="body CT (NIfTI/NRRD) already loaded in ITK-SNAP")
    ap.add_argument("--label", type=int, default=None,
                    help="proposed label id to apply (default: largest structure)")
    ap.add_argument("--itksnap-label", type=int, default=1)
    ap.add_argument("--url", default="http://localhost:8911")
    ap.add_argument("--sock", default="/tmp/snap-agent.sock")
    ap.add_argument("--model", default="TotalSegmentator")
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()

    client = DLSClient(args.url)

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
    print(f"[apply] proposed label {label} ({name}) -> ITK-SNAP label {args.itksnap_label}")

    out = os.path.join(tempfile.gettempdir(), f"p2_proposal_{label}.nii.gz")
    write_label_mask(result, source, label, out)

    ch = SnapChannel(args.sock)
    ch.set_actor("agent")
    applied = ch.apply_seg_file(out, args.itksnap_label)
    print("[apply] ->", json.dumps(applied, indent=2))
    print("[read_audit] ->", json.dumps(ch.get_audit(), indent=2))
    print("\nNow the human corrects the proposal in ITK-SNAP; call read_audit again "
          "for the human-tagged correction record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
