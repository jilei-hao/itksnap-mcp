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
