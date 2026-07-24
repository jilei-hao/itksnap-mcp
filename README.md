# itksnap-mcp

**Model proposes, human disposes.** An agent-callable interface to
[ITK-SNAP](https://itksnap.org)-based segmentation: an external agent (via
[MCP](https://modelcontextprotocol.io)) runs automatic segmentation, applies a proposed structure
into an **ITK-SNAP workspace**, and when a case needs human judgment the expert corrects it —
with the correction returned as a **structured, audited record** the agent can consume.

The **workspace is the base for all work** — a durable `.itksnap` file the agent creates and applies
into **headlessly**, with no running GUI required. A **live ITK-SNAP is a choice, not a requirement**:
the agent can *optionally* launch ITK-SNAP on that same workspace (with a live command socket) so the
human can view and correct the proposal.

This repo is the **Python glue** that makes ITK-SNAP callable: a thin client for the ITK-SNAP
deep-learning segmentation (DLS) server, a headless workspace engine (drives the `itksnap-wt` CLI +
SimpleITK), a socket client for ITK-SNAP's live command channel, and an MCP server exposing the whole
workflow as agent tools. The C++ pieces (voxel edit + audit record + `--agent-listen` command channel)
live in [`itksnap`](https://github.com/jilei-hao/itksnap); the model server lives in
[`itksnap-dls`](https://github.com/jilei-hao/itksnap-dls).

> **Status: working prototype** built for the SIIM-CAIMI26 AI Builder Showcase. The full
> propose → apply → audit backbone is verified live end-to-end (see below). This is also the intended
> pip-installable, agent-facing surface for the ITK-SNAP "composable human-in-the-loop" effort.

The architecture and the *why* (with the architecture + end-to-end-flow figures) are written up in the
project's design docs, maintained alongside the ITK-SNAP agentic-API sprint.

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
| `create_workspace(ct_path, name=None)` | create the base `.itksnap` workspace (main image + empty segmentation) that all work applies into |
| `propose(ct_path, model_id="TotalSegmentator", fast=True)` | run automatic segmentation → present labels + voxel counts |
| `apply(label_id, itksnap_label=1, actor="agent")` | apply one proposed structure into the **workspace segmentation** (headless); return the audit record |
| `apply_file(path, itksnap_label=1, actor="agent")` | apply a mask NIfTI already on disk into the workspace |
| `open_in_itksnap(live=True)` | launch the ITK-SNAP GUI on the workspace (optional) so the human can view/correct; `live` opens the `--agent-listen` socket |
| `read_audit()` | the most recent committed edit's audit record (live GUI if attached, else the workspace log) |
| `set_actor(actor)` | tag who is responsible for the next committed edit (`agent` \| `human`) |

## Configuration

The MCP server reads its setup from the environment (all optional; the headless flow needs only
`itksnap-wt`):

| Variable | Purpose | Default |
|---|---|---|
| `ITKSNAP_WT_BIN` | path to the `itksnap-wt` workspace CLI (headless engine) | `itksnap-wt` on `PATH` |
| `ITKSNAP_BIN` | path to the `ITK-SNAP` GUI (for `open_in_itksnap`) | `ITK-SNAP`/`itksnap` on `PATH` |
| `ITKSNAP_LAUNCH_PREFIX` | prefix for launching the GUI, e.g. `xvfb-run -a` on a headless box | *(none)* |
| `ITKSNAP_WORKSPACE_DIR` | where workspaces + their segmentations live | `<tmp>/itksnap-mcp/workspaces` |
| `ITKSNAP_DLS_URL` | itksnap-dls model server base URL | `http://localhost:8911` |
| `ITKSNAP_AGENT_SOCK` | socket a *live* ITK-SNAP listens on | `/tmp/snap-agent.sock` |

## Run the full demo

Prereqs: `itksnap-wt` + `ITK-SNAP` from an ITK-SNAP build, a GPU box with the DLS server dependencies
(only for `propose`), and a 3-D body CT (`ct.nii.gz`).

```bash
pip install -e '.[dev]'
export ITKSNAP_WT_BIN=/path/to/build/Utilities/Workspace/itksnap-wt
export ITKSNAP_BIN=/path/to/build/ITK-SNAP

# 1) Model server (from an itksnap-dls checkout on feature/agentic-api):
python -m itksnap_dls --port 8911 --device cuda

# 2) Drive the whole flow — create workspace → propose → apply (headless) → open for the human:
python demo/run_p2.py --ct ct.nii.gz --url http://localhost:8911 --open
```

`run_p2.py` creates the workspace, runs TotalSegmentator on the CT, applies the largest proposed
structure (or `--label N`) into the **workspace segmentation** tagged `actor: agent`, prints the audit
record, and (with `--open`) launches ITK-SNAP on the workspace. Correct the result in the GUI with the
paintbrush and call `read_audit` again — the correction comes back tagged `actor: human`. No running
ITK-SNAP is needed for the `apply` step itself.

## What's here

```
src/itksnap_mcp/
  config.py       # resolve binaries (itksnap-wt / ITK-SNAP), workspace dir, DLS url, socket from env
  workspace.py    # headless workspace engine: itksnap-wt create/edit + SimpleITK apply + audit log
  dls_client.py   # thin HTTP client for the itksnap-dls server (status/models/start/upload/run_automatic)
  channel.py      # SnapChannel: client for ITK-SNAP's --agent-listen Unix socket (JSON-RPC)
  server.py       # MCP server: create_workspace / propose / apply / open_in_itksnap / read_audit / ...
  confidence.py   # confidence gate: decide auto-accept vs route-to-human   (WIP)
demo/
  run_p2.py               # scripted end-to-end driver (create_workspace → propose → apply → open)
  agent_send.py           # send one raw command to the live socket (debugging)
  smoke_totalseg.py       # DLS-only automatic-segmentation smoke test
  manifest.example.yaml   # per-case demo manifest (copy to manifest.yaml, gitignored)
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
- Headless `apply` and a live GUI edit the same segmentation on disk. Because the running GUI holds the
  segmentation in memory, apply *before* `open_in_itksnap` (the intended order); a headless apply made
  while the GUI is open is not reflected until the workspace is reloaded.

## License

**MIT** — see [`LICENSE`](LICENSE). This repo is pure HTTP/socket glue and contains no ITK-SNAP
(GPL) source; the GUI and model server keep their own licenses in their respective repos.
