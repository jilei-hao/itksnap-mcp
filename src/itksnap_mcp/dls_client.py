"""Thin HTTP client for the itksnap-dls segmentation server.

Wire format mirrors ``itksnap_dls/common/image_utils.py`` and
``modules/segmentation/router.py`` on the ``feature/agentic-api`` branch:

- upload_raw : multipart ``file`` = gzip(float32 raw, numpy [z,y,x] C-order),
  ``metadata`` = JSON ``{"dimensions": [x, y, z], "components_per_pixel": 1}``.
- run_automatic : returns ``{"status", "result": base64(gzip(int16)), "dtype",
  "labels": {id: name}}``; the client reshapes ``result`` to the uploaded [z,y,x].

Pure HTTP + gzip + base64 + numpy — **no ITK-SNAP / ITK dependency**, so the client
ships independently of any compiled binary (SimpleITK is used only to load NIfTI files
in the convenience helpers).
"""
from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass
class AutomaticResult:
    """Decoded multi-label automatic-segmentation result."""
    labels: np.ndarray          # int16 volume in [z, y, x] order
    label_map: dict[int, str]   # {label_id: anatomy name}
    dtype: str

    @property
    def present_label_ids(self) -> list[int]:
        return sorted(int(v) for v in np.unique(self.labels) if v != 0)


class DLSClient:
    """Minimal client over the itksnap-dls REST API."""

    def __init__(self, base_url: str = "http://localhost:8911", timeout: float = 600.0,
                 verify: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify = verify

    # --- plumbing -------------------------------------------------------------
    def _get(self, path: str, **params) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/{path.lstrip('/')}", params=params or None,
                         timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        return r.json()

    # --- server / models ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return self._get("status")

    def list_models(self) -> list[dict[str, Any]]:
        return self._get("v2/models").get("models", [])

    def model_labels(self, model_id: str, task: str | None = None) -> dict[int, str]:
        out = self._get(f"v2/models/{model_id}/labels", **({"task": task} if task else {}))
        return {int(k): v for k, v in out.get("labels", {}).items()}

    # --- session lifecycle ----------------------------------------------------
    def start_session(self, model_id: str) -> str:
        out = self._get(f"v2/start_session/{model_id}")
        if "session_id" not in out:
            raise RuntimeError(f"start_session failed: {out}")
        return out["session_id"]

    def end_session(self, session_id: str) -> dict[str, Any]:
        return self._get(f"v2/end_session/{session_id}")

    # --- upload ---------------------------------------------------------------
    def upload_image(self, session_id: str, array_zyx: np.ndarray,
                     size_xyz: tuple[int, int, int]) -> dict[str, Any]:
        """Upload a scalar volume. ``array_zyx`` is float32 in numpy [z,y,x] order;
        ``size_xyz`` is the ITK/[x,y,z] size (== ``sitk.Image.GetSize()``)."""
        arr = np.ascontiguousarray(array_zyx, dtype=np.float32)
        blob = gzip.compress(arr.tobytes())
        metadata = json.dumps({"dimensions": list(size_xyz), "components_per_pixel": 1})
        r = requests.post(
            f"{self.base_url}/v2/upload_raw/{session_id}",
            files={"file": ("image.raw.gz", blob, "application/octet-stream")},
            data={"metadata": metadata},
            timeout=self.timeout, verify=self.verify,
        )
        r.raise_for_status()
        return r.json()

    # --- automatic (prompt-free) inference ------------------------------------
    def run_automatic(self, session_id: str, size_xyz: tuple[int, int, int],
                      task: str | None = None, fast: bool = True) -> AutomaticResult:
        params: dict[str, Any] = {"fast": str(fast).lower()}
        if task:
            params["task"] = task
        out = self._get(f"v2/run_automatic/{session_id}", **params)
        if out.get("status") != "success":
            raise RuntimeError(f"run_automatic failed: {out}")
        raw = gzip.decompress(base64.b64decode(out["result"]))
        dtype = np.dtype(out.get("dtype", "int16"))
        x, y, z = size_xyz
        labels = np.frombuffer(raw, dtype=dtype).reshape((z, y, x))
        label_map = {int(k): v for k, v in out.get("labels", {}).items()}
        return AutomaticResult(labels=labels, label_map=label_map, dtype=str(dtype))


def load_nifti_for_upload(path: str):
    """Load a NIfTI/NRRD via SimpleITK -> (array_zyx float32, size_xyz, sitk.Image).

    Returns the source image too so callers can copy geometry back onto results.
    NOTE: the current DLS scalar upload path drops spacing/origin/direction, so
    automatic segmentation runs on identity geometry until the server is fixed.
    """
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
    size_xyz = tuple(int(s) for s in img.GetSize())        # [x, y, z]
    return arr, size_xyz, img
