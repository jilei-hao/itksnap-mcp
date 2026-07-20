"""Unit tests for the confidence gate primitives (pure numpy, no GPU/server)."""
import numpy as np

from itksnap_mcp.confidence import dice, agreement_gate, evaluate


def _box(shape, lo, hi, label=1):
    a = np.zeros(shape, dtype=np.int16)
    sl = tuple(slice(lo[i], hi[i]) for i in range(len(shape)))
    a[sl] = label
    return a


def test_dice_identical_and_disjoint():
    a = _box((4, 4, 4), (1, 1, 1), (3, 3, 3))
    assert dice(a == 1, a == 1) == 1.0
    b = _box((4, 4, 4), (0, 0, 0), (1, 1, 1))
    c = _box((4, 4, 4), (3, 3, 3), (4, 4, 4))
    assert dice(b == 1, c == 1) == 0.0


def test_agreement_gate_agree():
    a = _box((6, 6, 6), (1, 1, 1), (5, 5, 5))
    d = agreement_gate(a, a, threshold=0.9)
    assert d.route_to_human is False
    assert d.per_label[1] == 1.0


def test_agreement_gate_disagree_routes_to_human():
    a = _box((6, 6, 6), (1, 1, 1), (5, 5, 5))       # 4x4x4 = 64
    b = _box((6, 6, 6), (1, 1, 1), (3, 3, 3))       # 2x2x2 = 8  -> low Dice
    d = agreement_gate(a, b, threshold=0.9)
    assert d.route_to_human is True
    assert d.per_label[1] < 0.9


def test_single_result_evaluate_runs():
    a = _box((6, 6, 6), (1, 1, 1), (5, 5, 5))
    d = evaluate(a, threshold=0.6)
    assert 1 in d.per_label
