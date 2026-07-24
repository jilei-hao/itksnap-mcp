"""Headless ITK-SNAP workspace as the base for the human-in-the-loop flow.

An ``.itksnap`` workspace is the durable source of truth: it holds the main
image and a segmentation layer on disk. The agent applies model proposals into
the workspace's segmentation **headlessly** -- no running GUI required -- and
gets back the same structured audit record the live GUI produces. A live
ITK-SNAP is then an optional *view/correct* surface opened on that same
workspace (see ``open_gui``), not a prerequisite for applying a proposal.

Design decisions (see projects/agentic-api):
  * The workspace file is written by the canonical ``itksnap-wt`` CLI, so it is
    always a valid ITK-SNAP workspace the human can open. This module only
    *edits pixels* (SimpleITK) and *records provenance* (Python); it never
    hand-writes the ``.itksnap`` registry.
  * The audit record is computed directly from the before/after label arrays
    (this module knows both), yielding the identical schema the GUI derives
    from its undo delta -- changed_voxels, tight bbox, before/after counts.
  * A sidecar ``<workspace>.mcp.json`` records MCP-owned metadata: the source
    image, the segmentation path, the armed actor, and the append-only audit
    log -- making the whole flow resumable with no live process.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any


class WorkspaceError(RuntimeError):
    """Raised when an itksnap-wt call fails or a proposal cannot be applied."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_empty_seg(ref_path: str, out_path: str) -> None:
    """Write an all-zero label image on the same grid/geometry as ``ref_path``."""
    import numpy as np
    import SimpleITK as sitk
    ref = sitk.ReadImage(ref_path)
    zeros = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    seg = sitk.GetImageFromArray(zeros)
    seg.CopyInformation(ref)
    sitk.WriteImage(seg, out_path)


def _audit_record(before, after, actor: str, op: str) -> dict[str, Any]:
    """Reconstruct the audit record from before/after label arrays (both [z,y,x]).

    Counts and bbox are over the *changed* voxels only, matching the GUI's
    delta-derived record. bbox min/max are in voxel [x, y, z] order.
    """
    import numpy as np
    changed = before != after
    n = int(np.count_nonzero(changed))
    if n > 0:
        zz, yy, xx = np.nonzero(changed)
        bbox = {
            "valid": True,
            "min": [int(xx.min()), int(yy.min()), int(zz.min())],
            "max": [int(xx.max()), int(yy.max()), int(zz.max())],
        }
        bvals, bcnts = np.unique(before[changed], return_counts=True)
        avals, acnts = np.unique(after[changed], return_counts=True)
        before_counts = {str(int(v)): int(c) for v, c in zip(bvals, bcnts)}
        after_counts = {str(int(v)): int(c) for v, c in zip(avals, acnts)}
    else:
        bbox = {"valid": False, "min": [0, 0, 0], "max": [0, 0, 0]}
        before_counts, after_counts = {}, {}
    return {
        "op": op,
        "timestamp": _now_iso(),
        "actor": actor,
        "changed_voxels": n,
        "bbox": bbox,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "time_point": 0,
    }


class Workspace:
    """An ITK-SNAP workspace edited headlessly via ``itksnap-wt`` + SimpleITK."""

    def __init__(self, path: str, wt_bin: str):
        self.path = os.path.abspath(path)
        self.wt_bin = wt_bin
        self.dir = os.path.dirname(self.path)
        base = os.path.basename(self.path)
        self.stem = base[:-8] if base.endswith(".itksnap") else os.path.splitext(base)[0]

    # --- itksnap-wt plumbing --------------------------------------------------
    def _run_wt(self, *args: str) -> str:
        proc = subprocess.run(
            [self.wt_bin, *args], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise WorkspaceError(
                f"itksnap-wt {' '.join(args)} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    # --- sidecar state --------------------------------------------------------
    @property
    def sidecar(self) -> str:
        return self.path + ".mcp.json"

    def _default_seg_path(self) -> str:
        return os.path.join(self.dir, self.stem + ".seg.nii.gz")

    def load_state(self) -> dict[str, Any]:
        if os.path.exists(self.sidecar):
            with open(self.sidecar) as f:
                return json.load(f)
        return {"source": None, "seg": self._default_seg_path(), "actor": "human", "audit": []}

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp = self.sidecar + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.sidecar)

    def seg_path(self) -> str:
        return self.load_state().get("seg") or self._default_seg_path()

    # --- lifecycle ------------------------------------------------------------
    @classmethod
    def create(cls, ct_path: str, path: str, wt_bin: str, *, overwrite: bool = True) -> "Workspace":
        """Create a workspace from a CT: main image + an empty segmentation layer."""
        ct_path = os.path.abspath(ct_path)
        if not os.path.exists(ct_path):
            raise WorkspaceError(f"CT not found: {ct_path}")
        ws = cls(path, wt_bin)
        os.makedirs(ws.dir, exist_ok=True)
        if os.path.exists(ws.path) and not overwrite:
            raise WorkspaceError(f"workspace already exists: {ws.path}")
        # 1) main image
        ws._run_wt("-laa", ct_path, "-o", ws.path)
        # 2) empty segmentation on the same grid, added as the segmentation layer
        seg = ws._default_seg_path()
        _write_empty_seg(ct_path, seg)
        ws._run_wt("-i", ws.path, "-las", seg, "-o", ws.path)
        ws._save_state({"source": ct_path, "seg": seg, "actor": "human", "audit": []})
        return ws

    # --- editing --------------------------------------------------------------
    def set_actor(self, actor: str) -> None:
        """Arm the actor for the next headless apply (persisted in the sidecar)."""
        st = self.load_state()
        st["actor"] = actor
        self._save_state(st)

    def apply_mask(self, mask_path: str, itksnap_label: int = 1,
                   actor: str = "agent", op: str = "Agent apply (proposal)") -> dict[str, Any]:
        """Paint ``itksnap_label`` where ``mask_path`` is nonzero into the workspace
        segmentation, write it back, and return + log the audit record."""
        import numpy as np
        import SimpleITK as sitk

        seg_path = self.seg_path()
        if not os.path.exists(seg_path):
            raise WorkspaceError(
                f"segmentation image missing: {seg_path} (create the workspace first)"
            )
        seg_img = sitk.ReadImage(seg_path)
        before = sitk.GetArrayFromImage(seg_img)  # [z, y, x]

        mask_img = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_img)   # [z, y, x]
        if mask.shape != before.shape:
            raise WorkspaceError(
                f"proposal grid {mask.shape} != workspace segmentation grid "
                f"{before.shape}; the proposal must share the workspace image grid"
            )

        after = before.copy()
        after[mask != 0] = int(itksnap_label)
        record = _audit_record(before, after, actor, op)

        out = sitk.GetImageFromArray(after.astype(before.dtype))
        out.CopyInformation(seg_img)
        sitk.WriteImage(out, seg_path)

        st = self.load_state()
        st["audit"].append(record)
        st["actor"] = "human"   # consume-on-commit: the tag auto-resets to human
        self._save_state(st)
        return record

    # --- provenance -----------------------------------------------------------
    def last_audit(self) -> dict[str, Any] | None:
        log = self.load_state().get("audit", [])
        return log[-1] if log else None

    def audit_log(self) -> list[dict[str, Any]]:
        return self.load_state().get("audit", [])

    # --- optional live view ---------------------------------------------------
    def open_gui(self, gui_bin: str, agent_listen: str | None = None,
                 launch_prefix: list[str] | None = None) -> int:
        """Launch the ITK-SNAP GUI on this workspace (optionally with a live
        ``--agent-listen`` socket). Returns the child PID. Non-blocking."""
        cmd = list(launch_prefix or []) + [gui_bin, "-w", self.path]
        if agent_listen:
            cmd += ["--agent-listen", agent_listen]
        proc = subprocess.Popen(
            cmd, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return proc.pid
