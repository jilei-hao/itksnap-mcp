# ITK-SNAP Agentic API — Design

> **One sentence:** we turned an expert's manual segmentation correction — normally a
> side effect trapped inside a GUI — into a **callable, resumable, audited pipeline step**
> that an automated agent can invoke and get a machine-readable answer back from.

This document explains *what* the system is and *why* it is built the way it is. For the *how*
(files, functions, the exact algorithms), read the source: this repo's `src/itksnap_mcp/` for the
agent glue, and [`itksnap`](https://github.com/jilei-hao/itksnap) for the C++ audit engine and the
`--agent-listen` command channel. Two figures accompany this doc:

- [`architecture.svg`](./architecture.svg) — the components and how they connect.
- [`flow-chart.svg`](./flow-chart.svg) — the end-to-end control/data flow of one run.

---

## 1. The problem, in plain terms

Automatic segmentation models (e.g. TotalSegmentator) are good but imperfect. Real clinical
and research pipelines still need a human expert to *verify and fix* the model's output. But
today that human judgement is **locked inside an interactive GUI**: a person opens the image,
paints a correction, saves a file, and the "why" and "what changed" evaporate. An automated
pipeline or an AI agent has no clean way to **call a human as a step** and get a structured
answer back.

Our thesis — *"model proposes, human disposes"* — is to make the human expert a **first-class,
callable checkpoint**. An agent runs a model, and when it wants human judgement it invokes
ITK-SNAP as a tool; the human's correction comes back not as an opaque file but as a
**structured audit record**:

```json
{
  "op": "Agent apply (proposal)",
  "timestamp": "2026-07-19T02:26:20Z",
  "actor": "agent",
  "changed_voxels": 1169665,
  "bbox": { "valid": true, "min": [84, 2, 0], "max": [247, 189, 180] },
  "before_counts": { "0": 1169665 },
  "after_counts":  { "1": 1169665 },
  "time_point": 0
}
```

That record is the "return value" of the edit: *who* did it (agent vs human), *when*, *how many
voxels changed*, *where* (bounding box), and *from which label to which label*. It makes an edit
**attributable, reproducible, and consumable** by the next stage of an automated workflow.

---

## 2. The big picture — three tiers

The system spans three repositories, each a tier with a single job. See
[`architecture.svg`](./architecture.svg).

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  AGENT / ORCHESTRATION                                               │
  │  itksnap-mcp  (public Python: MCP tools + demo driver)              │
  │    propose · apply · read_audit · set_actor · list_models          │
  └───────────────┬───────────────────────────────┬────────────────────┘
                  │ HTTP (propose)                 │ Unix socket (apply / audit)
                  ▼                                 ▼
  ┌───────────────────────────┐     ┌──────────────────────────────────┐
  │  MODEL SERVER             │     │  ITK-SNAP  (C++ GUI + Logic)      │
  │  itksnap-dls (FastAPI)    │     │  --agent-listen JSON-RPC channel  │
  │  TotalSegmentator, etc.   │     │  + the audit-record engine        │
  └───────────────────────────┘     └──────────────────────────────────┘
```

**Why three tiers instead of one?** Each boundary is a deliberate seam:

- **The model is swappable.** `itksnap-dls` speaks a plain HTTP wire format, so TotalSegmentator
  can be replaced by any other model without touching the agent or the GUI.
- **The agent glue is shippable on its own.** `itksnap-mcp` is pure Python (no compiled ITK
  dependency), so it can be `pip install`-ed and updated independently of the heavyweight,
  natively-installed GUI. This mirrors the `greedy_python` distribution pattern and is the
  artifact whose downloads we can actually measure.
- **The GUI stays where the human is.** The live human correction happens in the *real*
  ITK-SNAP process the human sees — not a headless clone — because the whole point is capturing
  genuine expert judgement.

---

## 3. The two ways the agent talks to ITK-SNAP

There are exactly two channels, and keeping them separate is a core design choice:

| Channel | Transport | Purpose | Example call |
|---|---|---|---|
| **Propose** | HTTP → `itksnap-dls` | Run a model, get a label volume | `propose(ct_path)` |
| **Drive / read** | Unix domain socket → ITK-SNAP `--agent-listen` | Apply into the live GUI, read the audit | `apply(label_id)`, `read_audit()` |

The **socket** channel is the novel part. ITK-SNAP is launched with
`--agent-listen /tmp/snap-agent.sock`; it opens a local server that speaks **newline-delimited
JSON-RPC** on the GUI thread. An external process sends one JSON line, ITK-SNAP acts on the
*running* GUI and replies with one JSON line. Commands today:

```
ping                                     → "pong"
set_cursor {x,y,z} / get_cursor          → move/read the crosshair
set_actor {actor: "agent"|"human"}       → tag who is responsible for the next edit
apply_box {x0..z1, label}                → paint a labeled box (a committed edit)
apply_seg_file {path, label}             → apply a proposed mask NIfTI (a committed edit)
get_audit                                → return the last edit's audit record
```

Running commands *on the GUI thread* is not a limitation — it is what makes this **safe**. Every
edit goes through the same code path a mouse click would, so there is no second, racy write path
into the segmentation.

---

## 4. The core idea: the audit record is *reconstructed*, not *instrumented*

The most important design decision is how the audit record is produced. The naive approach would
be to sprinkle "record what changed" bookkeeping through every one of ITK-SNAP's ~11 editing
operations (paintbrush, polygon, 3D spray, threshold, interpolation, auto-seg apply, …). That is
invasive and fragile.

Instead we exploit a fact that was already true: **ITK-SNAP already records every edit as a
compressed "delta"** for its undo/redo system. A delta stores, per voxel, the *difference*
`new − old` in run-length-encoded form. So after an edit commits, we can **reconstruct** exactly
what happened by walking that delta against the *current* (post-edit) image:

```
for each changed voxel:
    new_label = current_image[voxel]        # the image already holds the new state
    old_label = new_label − delta[voxel]    # subtract the recorded difference
```

From that single pass we get everything: the count of changed voxels, the tight bounding box, and
the before/after label histograms. This is the *same* arithmetic ITK-SNAP's Undo already uses to
roll an edit back, so we know it is correct.

**A tiny worked example.** Suppose a 2×2 patch was blank (label `0`) and the agent painted it to
label `3`. The delta stores `+3` for those 4 voxels. After the edit the image reads `3` there, so
`old = 3 − 3 = 0`. The reconstruction yields: `changed_voxels = 4`, `before = {0: 4}`,
`after = {3: 4}`, `bbox = the 2×2 patch`. Exactly right, and we never touched the paint code.

**Why this is safe (the one precondition):** within a *single* commit ITK-SNAP always paints with
one constant label, and re-touching an already-painted voxel records a *zero* difference. So no
voxel is ever assigned two different non-zero deltas in one commit, which is precisely the
condition that makes `old = new − delta` exact. Every real editing path satisfies this; the
implementation doc spells out the one contrived case that would not, and the unit test that guards
it.

**The single capture point.** Because reconstruction only needs "a delta was just committed",
capture lives in the *one* function every edit funnels through — the commit sink. One place to get
right, not eleven.

---

## 5. Who did it? — the actor model

An audit record must say **agent** or **human**. We model this as a one-shot "arm" flag:

- Before an agent-driven edit, the agent calls `set_actor agent`.
- The *next* committed edit consumes that flag and is tagged `agent`; the flag then **auto-resets
  to `human`**.
- So a subsequent edit that nobody armed — e.g. the human picking up the paintbrush — is tagged
  `human` automatically.

This "consume-on-commit" rule was chosen after an adversarial review found that a naive
"reset on every call" could mislabel a human edit as agent (or let a throwaway internal commit
steal the agent tag). Consume-on-commit ties the tag precisely to the commit it belongs to. The
one honest caveat: the agent must arm the flag *immediately before* an operation it knows will
commit — documented, and true for the demo's flow.

---

## 6. Geometry: making the proposal line up with the image

The model server currently transports raw voxels and drops the image's spatial metadata (spacing,
origin, orientation), so a proposed mask comes back on an "identity" grid. Before the agent hands a
proposal to ITK-SNAP, it **restores the original CT's geometry onto the mask** (copying spacing/
origin/orientation from the source image). Because ITK-SNAP has the *same* CT loaded, the mask then
lines up voxel-for-voxel. This is a small, explicit step in the agent glue rather than a silent
assumption.

---

## 7. What one end-to-end run looks like

See [`flow-chart.svg`](./flow-chart.svg) for the diagram. In words, the demo we ran live on a GPU:

1. **propose** — the agent sends a body CT to `itksnap-dls`; TotalSegmentator returns a multi-label
   volume (48 structures: heart, aorta, lungs, vertebrae, …).
2. **gate** *(planned)* — a confidence check decides *auto-accept* vs *route-to-human*.
3. **apply** — the agent extracts one structure (e.g. the left upper lung lobe, 1,169,665 voxels),
   restores geometry, arms `actor = agent`, and sends `apply_seg_file` over the socket. ITK-SNAP
   applies it through its normal edit/commit path and the audit engine captures the record.
4. **read_audit** — the agent reads back the structured record (shown in §1).
5. **human disposes** — the expert corrects the proposal in the live GUI (paintbrush). That edit
   commits and is auto-tagged `human`; the agent calls `read_audit` again and receives the
   correction as a structured diff.

The result: an agent orchestrated an automatic model **and** a human expert as two callable steps
in one pipeline, with every change captured as reusable, attributable provenance.

---

## 8. Design principles, summarized

1. **Integrate, don't reinvent.** The audit record rides on the existing undo delta; the socket
   channel reuses the existing GUI-thread editing primitives.
2. **One chokepoint.** All edits commit through a single function, so provenance is captured in one
   place regardless of which tool made the edit.
3. **Reconstruct over instrument.** Derive the record from data that already exists, keeping the
   change tiny and the blast radius near zero.
4. **Make the human a real code path.** The correction happens in the live GUI, not a simulation.
5. **Toolkit-independent core.** The audit record and its serializer have no GUI dependency, so
   they are unit-testable in isolation and reusable by a future headless API.
6. **Explicit seams.** Model ↔ agent ↔ GUI are separate transports so each can evolve or be
   swapped independently.
