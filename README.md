# itksnap-mcp

**Model proposes, human disposes.** An agent-callable interface to
[ITK-SNAP](https://itksnap.org)-based segmentation: an external agent (via
[MCP](https://modelcontextprotocol.io)) runs automatic segmentation, applies a proposed structure
into a **live ITK-SNAP** session, and when a case needs human judgment the expert corrects it —
with the correction returned as a **structured, audited record** the agent can consume.

This repo is the **Python glue** that makes ITK-SNAP callable: a thin client for the ITK-SNAP
deep-learning segmentation (DLS) server, a socket client for ITK-SNAP's live command channel, and an
MCP server exposing the whole workflow as agent tools. The C++ pieces (voxel edit + audit record +
`--agent-listen` command channel) live in [`itksnap`](https://github.com/jilei-hao/itksnap); the
model server lives in [`itksnap-dls`](https://github.com/jilei-hao/itksnap-dls).

> **Status: working prototype** built for the SIIM-CAIMI26 AI Builder Showcase. The full
> propose → apply → audit backbone is verified live end-to-end (see below). This is also the intended
> pip-installable, agent-facing surface for the ITK-SNAP "composable human-in-the-loop" effort.

See **[`docs/DESIGN.md`](docs/DESIGN.md)** for the architecture and the *why*, with the
[architecture](docs/architecture.svg) and [end-to-end flow](docs/flow-chart.svg) figures.

---

## What it does

An agent orchestrates an automatic model **and** a human expert as two callable steps in one
pipeline, and every change comes back as machine-readable provenance:

```json
{
  "op": "Agent apply (proposal)",
  "timestamp": "2026-07-19T02:26:20Z",
  "actor": "agent",
  "changed_voxels": 1169665,
  "bbox": { "valid": true, "min": [84, 2, 0], "max": [247, 189, 180] },
  "before_counts": { "0": 1169665 },
  "after_counts":  { "1": 1169665 }
}
```

`actor` distinguishes an agent-applied proposal from a human correction — so a downstream pipeline
knows *who* made each change and can feed corrections back into model fine-tuning and QA.

## MCP tools

| Tool | What it does |
|---|---|
| `list_models()` | list available segmentation models |
| `propose(ct_path, model_id="TotalSegmentator", fast=True)` | run automatic segmentation → present labels + voxel counts |
| `apply(label_id, itksnap_label=1, actor="agent")` | apply one proposed structure into the running ITK-SNAP; return the audit record |
| `read_audit()` | the most recent committed edit's audit record |
| `set_actor(actor)` | tag who is responsible for the next committed edit (`agent` \| `human`) |

## Run the full demo (3 commands)

Prereqs: a GPU box with the DLS server dependencies, an ITK-SNAP build with `--agent-listen`, and a
3-D body CT (`ct.nii.gz`).

```bash
pip install -e '.[dev]'

# 1) Model server (from an itksnap-dls checkout on feature/agentic-api):
python -m itksnap_dls --port 8911 --device cuda

# 2) ITK-SNAP with the live command channel (add `xvfb-run -a` if headless):
ITK-SNAP -g ct.nii.gz --agent-listen /tmp/snap-agent.sock

# 3) Drive the whole flow — propose → apply → read the audit record:
python demo/run_p2.py --ct ct.nii.gz --sock /tmp/snap-agent.sock --url http://localhost:8911
```

`run_p2.py` runs TotalSegmentator on the CT, applies the largest proposed structure (or `--label N`)
into the live ITK-SNAP tagged `actor: agent`, and prints the audit record. Then correct the result in
the GUI with the paintbrush and call `read_audit` again — the correction comes back tagged
`actor: human`.

## What's here

```
src/itksnap_mcp/
  dls_client.py   # thin HTTP client for the itksnap-dls server (status/models/start/upload/run_automatic)
  channel.py      # SnapChannel: client for ITK-SNAP's --agent-listen Unix socket (JSON-RPC)
  server.py       # MCP server: propose / apply / read_audit / set_actor / list_models
  confidence.py   # confidence gate: decide auto-accept vs route-to-human   (WIP)
demo/
  run_p2.py               # scripted end-to-end driver (propose → apply → read_audit)
  agent_send.py           # send one raw command to the live socket (debugging)
  smoke_totalseg.py       # DLS-only automatic-segmentation smoke test
  manifest.example.yaml   # per-case demo manifest (copy to manifest.yaml, gitignored)
docs/
  DESIGN.md · architecture.svg · flow-chart.svg
```

## Architecture (three repos)

| Piece | Repo | Role |
|---|---|---|
| Voxel edits + **audit record** + `--agent-listen` command channel | [`itksnap`](https://github.com/jilei-hao/itksnap) (`sprint/caimi`) | C++ Logic tier + GUI |
| Model server (TotalSegmentator automatic; nnInteractive, SAM2) | [`itksnap-dls`](https://github.com/jilei-hao/itksnap-dls) (`feature/agentic-api`) | FastAPI + PyTorch |
| **This repo** — DLS client, socket client, MCP server, demo | `itksnap-mcp` | Python glue / agent surface |

## Known limitations (prototype)

- The DLS `upload_raw` scalar path currently ships pixels only (no spacing/origin/direction), so the
  proposal comes back on identity geometry; the agent restores the source CT's geometry before
  applying (`server.write_label_mask`). Proposal and image must share the same voxel grid.
- `apply` applies one structure under one label; a full multi-label apply is a straightforward
  extension. `confidence.py` (auto-accept vs route-to-human gating) is still a placeholder.
- The audit `actor` tag is armed one commit ahead (`set_actor`) and consumed by the next commit; arm
  it immediately before a committing operation.

## License

**MIT** — see [`LICENSE`](LICENSE). This repo is pure HTTP/socket glue and contains no ITK-SNAP
(GPL) source; the GUI and model server keep their own licenses in their respective repos.
