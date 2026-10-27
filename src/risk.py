"""
SolarGuard AI — Risk Prioritisation
====================================
Rule-based review-priority assignment using config/risk_rules.json.

All priorities, suggested actions, and review intervals are provisional
demonstration rules. They must be verified by qualified personnel before
any operational use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RULES_PATH = Path(__file__).parent.parent / "config" / "risk_rules.json"
_PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]


def _load_rules() -> dict[str, Any]:
    if not _RULES_PATH.exists():
        raise FileNotFoundError(
            f"Risk rules file not found: {_RULES_PATH}. "
            "Ensure config/risk_rules.json exists."
        )
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_rule(class_name: str) -> dict[str, Any] | None:
    """Return the risk rule for a given class name, or None if unknown."""
    rules = _load_rules()
    return rules.get("class_rules", {}).get(class_name)


def priority_rank(priority: str) -> int:
    """Lower rank = higher urgency (Critical=0, High=1, …)."""
    try:
        return _PRIORITY_ORDER.index(priority)
    except ValueError:
        return len(_PRIORITY_ORDER)


def assess_inspection(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assess a list of detections and return a structured risk report.

    Parameters
    ----------
    detections:
        List of dicts with at least ``class_name`` and ``confidence`` keys.

    Returns
    -------
    dict with keys:
        overall_priority, class_results, reasons, suggested_actions,
        review_interval_days, disclaimer, no_detection
    """
    rules = _load_rules()
    class_rules: dict[str, Any] = rules.get("class_rules", {})
    disclaimer: str = rules.get("disclaimer", "")

    if not detections:
        return {
            "overall_priority": "None",
            "class_results": [],
            "reasons": [],
            "suggested_actions": [],
            "review_interval_days": None,
            "disclaimer": disclaimer,
            "no_detection": True,
            "no_detection_message": rules.get(
                "no_detection_message",
                "No supported fault detected in this image.",
            ),
            "no_detection_note": rules.get("no_detection_note", ""),
        }

    class_results: list[dict] = []
    seen_classes: set[str] = set()

    for det in detections:
        cls = det.get("class_name", "Unknown")
        conf = float(det.get("confidence", 0.0))
        if cls in seen_classes:
            continue
        seen_classes.add(cls)

        rule = class_rules.get(cls)
        if rule is None:
            rule = {
                "priority": "Medium",
                "reason": f"Class '{cls}' has no configured risk rule; defaulting to Medium.",
                "suggested_action": "Review with domain expert.",
                "review_interval_days": 30,
                "severity_note": "No configured rule.",
            }

        class_results.append(
            {
                "class_name": cls,
                "max_confidence": conf,
                "priority": rule["priority"],
                "reason": rule["reason"],
                "suggested_action": rule["suggested_action"],
                "review_interval_days": rule["review_interval_days"],
                "severity_note": rule.get("severity_note", ""),
            }
        )

    # Determine the highest-urgency priority across all detected classes.
    if class_results:
        class_results.sort(key=lambda r: priority_rank(r["priority"]))
        overall_priority = class_results[0]["priority"]
        min_interval = min(
            (r["review_interval_days"] for r in class_results if r["review_interval_days"]),
            default=30,
        )
    else:
        overall_priority = "None"
        min_interval = None

    return {
        "overall_priority": overall_priority,
        "class_results": class_results,
        "reasons": [r["reason"] for r in class_results],
        "suggested_actions": list(
            dict.fromkeys(r["suggested_action"] for r in class_results)
        ),
        "review_interval_days": min_interval,
        "disclaimer": disclaimer,
        "no_detection": False,
        "no_detection_message": "",
        "no_detection_note": "",
    }


def priority_color(priority: str) -> str:
    """Return a Streamlit-compatible color string for a priority level."""
    return {
        "Critical": "#FF4B4B",
        "High": "#FF8C00",
        "Medium": "#FFC107",
        "Low": "#4CAF50",
        "None": "#9E9E9E",
    }.get(priority, "#9E9E9E")


def priority_badge(priority: str) -> str:
    """Return an HTML badge string for a priority level."""
    color = priority_color(priority)
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 8px;border-radius:4px;font-size:0.8em;">'
        f"{priority}</span>"
    )
