"""Fuzzy confidence inference."""

from typing import Iterable

import config


def confidence_membership(confidence: float) -> float:
    """Map confidence to fuzzy membership in [0, 1].

    Rule:
      confidence < 0.4  -> 0
      0.4..0.7          -> linear interpolation
      confidence > 0.7  -> 1
    """
    if confidence <= config.LOW_CONF:
        return 0.0
    if confidence >= config.HIGH_CONF:
        return 1.0
    return (confidence - config.LOW_CONF) / (config.HIGH_CONF - config.LOW_CONF)


def fuzzy_scores(confidences: Iterable[float]) -> list[float]:
    return [confidence_membership(float(c)) for c in confidences]


def fuzzy_filter(confidences: Iterable[float], membership_threshold: float = config.FUZZY_MEMBERSHIP_THRESHOLD):
    """Return fuzzy memberships and a boolean keep mask."""
    scores = fuzzy_scores(confidences)
    mask = [score >= membership_threshold for score in scores]
    return scores, mask
