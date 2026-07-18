# itksnap-mcp

**Model proposes, human disposes.** An agent-callable interface to
[ITK-SNAP](https://itksnap.org)-based segmentation: an external agent (via
[MCP](https://modelcontextprotocol.io)) runs automatic segmentation, and when a case needs human
judgment, routes it to an expert who corrects it — with the correction returned as a **structured,
audited record**.

This repo is the **Python glue**: a thin client for the ITK-SNAP deep-learning segmentation (DLS)
server, a confidence gate, and an MCP server exposing the workflow as agent tools. The C++ pieces
(voxel edit + audit record, live command channel) live in
[`itksnap`](https://github.com/jilei-hao/itksnap); the model server lives in
[`itksnap-dls`](https://github.com/jilei-hao/itksnap-dls).

> **Status: prototype / work-in-progress.** Built for the SIIM-CAIMI26 AI Builder Showcase.
> This is the demo artifact reviewers can browse; it is also the intended pip-installable
> agent-facing surface for the ITK-SNAP OS4LS "composable human-in-the-loop" effort.

## What's here

```
src/itksnap_mcp/
  dls_client.py   # thin HTTP client for the itksnap-dls server (status/models/start/upload/run_automatic)
  confidence.py   # confidence gate: decide auto-accept vs route-to-human   (WIP)
  server.py       # MCP server exposing headless.* (and later live.*) tools  (WIP skeleton)
demo/
  smoke_totalseg.py       # end-to-end TotalSegmentator smoke test via the DLS server (Gate 1)
  manifest.example.yaml   # per-case demo manifest (never hardcode filenames)
```

## Quickstart (dev)

```bash
pip install -e '.[dev]'

# 1) Run the DLS server (from an itksnap-dls checkout on feature/agentic-api, GPU box):
python -m itksnap_dls --port 8911 --device cuda

# 2) Smoke-test automatic segmentation end-to-end:
python demo/smoke_totalseg.py --ct /path/to/body_ct.nii.gz --url http://localhost:8911 --out /tmp/seg.nii.gz
```

## Architecture (three repos)

| Piece | Repo | Role |
|---|---|---|
| Voxel edits + **audit record**, (stretch) live command channel | `itksnap` (`sprint/caimi`) | C++ Logic tier + GUI |
| Model server (nnInteractive, SAM2, **TotalSegmentator** automatic) | `itksnap-dls` (`feature/agentic-api`) | FastAPI + PyTorch |
| **This repo** — thin DLS client, confidence gate, MCP server, demo | `itksnap-mcp` | Python glue / agent surface |

## Known limitations (prototype)

- The DLS `upload_raw` scalar path currently ships pixels only (no spacing/origin/direction), so
  automatic-segmentation geometry uses identity spacing until the server threads geometry through.
  Fine for pipeline/shape checks; matters for anatomically faithful output.
- MCP `server.py` and `confidence.py` are skeletons; the audited-correction round-trip depends on the
  C++ audit record (in `itksnap`).

## License

**TBD** — MIT (matches `itksnap-dls`, pure HTTP glue) vs GPL-3.0 (matches the ITK-SNAP project).
Decide before publishing the CAIMI demo link. See `projects/agentic-api/docs/sprint_caimi.md` §7.
