#!/usr/bin/env python
"""Gate-1 smoke test: end-to-end automatic segmentation through the DLS server.

    python demo/smoke_totalseg.py --ct body_ct.nii.gz --url http://localhost:8911 --out seg.nii.gz

Exercises: /status -> /v2/start_session/TotalSegmentator -> /v2/upload_raw ->
/v2/run_automatic?fast=true, then decodes the multi-label result, checks the shape
matches the uploaded volume, prints the present anatomy labels, and (optionally)
writes the segmentation as NIfTI (geometry copied from the input).

Exit 0 = Gate 1 PASS.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/demo/", 1)[0] + "/src")
from itksnap_mcp.dls_client import DLSClient, load_nifti_for_upload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="body CT (NIfTI, HU)")
    ap.add_argument("--url", default="http://localhost:8911")
    ap.add_argument("--task", default=None, help="TotalSegmentator task (default: server DEFAULT_TASK)")
    ap.add_argument("--out", default=None, help="write the segmentation here (NIfTI)")
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()

    client = DLSClient(args.url)
    print("status:", client.status())
    print("models:", [m.get("id", m.get("ID")) for m in client.list_models()])

    arr_zyx, size_xyz, src = load_nifti_for_upload(args.ct)
    print(f"input: size(xyz)={size_xyz} dtype={arr_zyx.dtype} range=[{arr_zyx.min():.0f},{arr_zyx.max():.0f}]")

    sid = client.start_session("TotalSegmentator")
    print("session:", sid)
    print("upload:", client.upload_image(sid, arr_zyx, size_xyz))

    print("running automatic segmentation (fast=%s)..." % args.fast)
    res = client.run_automatic(sid, size_xyz, task=args.task, fast=args.fast)
    client.end_session(sid)

    x, y, z = size_xyz
    assert res.labels.shape == (z, y, x), f"shape mismatch: {res.labels.shape} != {(z, y, x)}"
    present = res.present_label_ids
    print(f"result: shape={res.labels.shape} dtype={res.dtype} n_labels={len(present)}")
    for lid in present[:15]:
        print(f"  label {lid:>3}: {res.label_map.get(lid, '?')}")
    if len(present) > 15:
        print(f"  ... (+{len(present) - 15} more)")

    if args.out:
        import SimpleITK as sitk
        seg = sitk.GetImageFromArray(res.labels.astype(np.int16))
        seg.CopyInformation(src)  # geometry from the input CT
        sitk.WriteImage(seg, args.out)
        print("wrote:", args.out)

    if not present:
        print("GATE 1 FAIL: no labels produced")
        return 1
    print("GATE 1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
