"""Rule-side derivation for human-target v2 myopia scores."""

from __future__ import annotations


ACTION_MYOPIA_THRESHOLD = 0.6
SCORABLE_SCOPES = {"substantive", "uncertain"}


def _score(value) -> float:
    try:
        f = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if f != f:
        return 0.0
    return max(0.0, min(1.0, f))


def _inverse_task_score(value) -> float:
    if value is None:
        return 0.0
    return 1.0 - _score(value)


def derive_action_myopia_score(action: dict) -> float:
    """Derive action-level myopia score from v2 scalar fields and penalty."""
    if action.get("risk_scope") not in SCORABLE_SCOPES:
        return 0.0

    rv = action.get("manual_risk_vector") or {}
    wa = action.get("wrong_abstraction") or {}
    score = max(
        _inverse_task_score(rv.get("task_advancement")),
        _score(rv.get("debt_density")),
        _score(rv.get("fragility_delta")),
        _score(rv.get("regression_surface")),
        _score(rv.get("observability_loss")),
        _score(wa.get("severity")),
    )
    return round(score, 3)


def derive_is_myopic(action: dict) -> bool:
    """Return the v2 binary myopic flag derived from action_myopia_score."""
    return derive_action_myopia_score(action) >= ACTION_MYOPIA_THRESHOLD


def derive_trajectory_myopia_score(actions: list[dict], trajectory_penalties: dict) -> float:
    """Derive trajectory-level myopia from action scores and trajectory penalties."""
    action_max = max((derive_action_myopia_score(a) for a in actions), default=0.0)
    broad = trajectory_penalties.get("broad_rewrite") or {}
    artifact = trajectory_penalties.get("artifact_residue") or {}
    score = max(action_max, _score(broad.get("severity")), _score(artifact.get("severity")))
    return round(score, 3)


def min_confidence(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return min(present)
