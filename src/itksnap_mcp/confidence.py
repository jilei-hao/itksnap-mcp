"""Confidence gate: decide whether an automatic segmentation is trustworthy enough
to auto-accept, or should be routed to a human for review.

WIP — the real gate (plan §5.1) compares mask stability across perturbed inputs
(e.g. two seeds / two fast-mode resolutions). This module currently provides a
transparent placeholder heuristic over a single multi-label result so the end-to-end
plumbing works; swap in the perturbation-based gate during W3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateDecision:
    route_to_human: bool
    reason: str
    per_label: dict[int, float]  # label_id -> instability score in [0, 1]


def surface_to_volume_instability(labels: np.ndarray, label_ids=None) -> dict[int, float]:
    """Crude per-label instability proxy: surface-voxel fraction (surface/volume).

    High surface-to-volume flags fragmented/thin/noisy masks that most often need a
    human look. Placeholder for the perturbation-based signal.
    """
    ids = label_ids if label_ids is not None else [i for i in np.unique(labels) if i != 0]
    scores: dict[int, float] = {}
    for lid in ids:
        mask = labels == lid
        vol = int(mask.sum())
        if vol == 0:
            continue
        # 6-neighbour surface: a voxel is "surface" if any face neighbour differs
        surf = np.zeros_like(mask)
        for ax in range(mask.ndim):
            surf |= mask ^ np.roll(mask, 1, axis=ax)
            surf |= mask ^ np.roll(mask, -1, axis=ax)
        surf &= mask
        scores[int(lid)] = float(surf.sum()) / float(vol)
    return scores


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap between two boolean masks (1.0 if both are empty)."""
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * int((a & b).sum()) / denom


def agreement_gate(labels_a: np.ndarray, labels_b: np.ndarray,
                   threshold: float = 0.9, label_ids=None) -> GateDecision:
    """Perturbation-based gate: compare two automatic runs of the same case (e.g. two
    seeds, or fast vs full resolution) label by label. A structure whose two masks
    disagree — per-label Dice below ``threshold`` — is unstable and routed to a human.

    This is the signal the design calls for (mask instability across perturbed inputs);
    ``evaluate`` above is the single-result fallback when only one run is available.
    ``per_label`` here holds the Dice *agreement* (higher = more stable).
    """
    ids = label_ids if label_ids is not None else sorted(
        {int(i) for i in np.unique(labels_a) if i != 0} |
        {int(i) for i in np.unique(labels_b) if i != 0})
    per = {lid: dice(labels_a == lid, labels_b == lid) for lid in ids}
    disagree = {lid: d for lid, d in per.items() if d < threshold}
    if disagree:
        worst = min(disagree, key=disagree.get)
        return GateDecision(
            route_to_human=True,
            reason=f"label {worst} unstable (Dice={disagree[worst]:.2f} < {threshold})",
            per_label=per,
        )
    return GateDecision(route_to_human=False, reason="all labels agree across runs",
                        per_label=per)


def evaluate(labels: np.ndarray, threshold: float = 0.6, label_ids=None) -> GateDecision:
    scores = surface_to_volume_instability(labels, label_ids)
    flagged = {lid: s for lid, s in scores.items() if s >= threshold}
    if flagged:
        worst = max(flagged, key=flagged.get)
        return GateDecision(
            route_to_human=True,
            reason=f"label {worst} unstable (surface/volume={flagged[worst]:.2f} >= {threshold})",
            per_label=scores,
        )
    return GateDecision(route_to_human=False, reason="all labels within stability threshold",
                        per_label=scores)
