"""Small human-calibrated scope model for encoder judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from encoder_judge.evidence import ActionEvidence, build_evidence_table


SCORABLE_SCOPES = {"substantive", "uncertain"}


class ScopeCalibrator:
    """A shallow, generic scope classifier trained from human-target v2 gold."""

    def __init__(self) -> None:
        self.vectorizer = DictVectorizer()
        self.classifier = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)
        self.is_fit = False

    def fit(self, template_paths: list[Path], repo_cache_root: Path | None = None) -> None:
        rows: list[dict[str, Any]] = []
        labels: list[str] = []
        for template_path in template_paths:
            target_path = template_path.with_name(
                template_path.name.replace(".target.template.json", ".target.json")
            )
            if not target_path.exists():
                continue
            template = json.loads(template_path.read_text())
            target = json.loads(target_path.read_text())
            gold_by_id = {a.get("action_id"): a for a in target.get("action_level", [])}
            base_by_id = load_base_prelabel(template_path)
            for ev in build_evidence_table(template, template_path, repo_cache_root=repo_cache_root):
                gold = gold_by_id.get(ev.action_id)
                if not gold or not gold.get("risk_scope"):
                    continue
                rows.append(features_for(ev, base_by_id.get(ev.action_id)))
                labels.append(str(gold["risk_scope"]))
        if not rows:
            raise ValueError("no calibration rows found")
        matrix = self.vectorizer.fit_transform(rows)
        self.classifier.fit(matrix, labels)
        self.is_fit = True

    def predict_scope(self, ev: ActionEvidence, base_action: dict[str, Any] | None = None) -> str:
        if not self.is_fit:
            raise RuntimeError("ScopeCalibrator is not fit")
        matrix = self.vectorizer.transform([features_for(ev, base_action)])
        return str(self.classifier.predict(matrix)[0])


def calibration_templates(set_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for set_dir in set_dirs:
        paths.extend(sorted(set_dir.glob("*/*.target.template.json")))
    return paths


def load_base_prelabel(template_path: Path) -> dict[str, dict[str, Any]]:
    path = template_path.with_name(template_path.name.replace(".target.template.json", ".pre_label.json"))
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(a.get("action_id")): a for a in data.get("action_level", [])}


def features_for(ev: ActionEvidence, base_action: dict[str, Any] | None = None) -> dict[str, Any]:
    base_action = base_action or {}
    rv = base_action.get("manual_risk_vector") or {}
    final = ev.final_diff_contribution or {}
    semantic = str(final.get("semantic_survival") or "")
    source_paths = final.get("source_paths") or []
    artifact_paths = final.get("artifact_paths") or []
    hints = ev.static_evidence.get("wrong_abstraction_hints") or {}
    return {
        "kind": ev.action_kind,
        "observation_effect": ev.observation_effect,
        "patch_status": ev.patch_survival.status,
        "patch_confidence_bin": round(ev.patch_survival.confidence, 1),
        "final_contribution": bool(ev.final_diff_contribution.get("present")),
        "final_independent": bool(final.get("independent")),
        "semantic_survival": semantic,
        "exact_text_survived": bool(final.get("exact_text_survived")),
        "has_symbol_hits": bool(final.get("symbol_hits")),
        "has_source_paths": bool(source_paths),
        "has_artifact_final_paths": bool(artifact_paths),
        "n_paths": min(5, len(ev.target_files)),
        "has_artifact_path": bool(ev.artifact_signals.get("introduced_artifact_paths")),
        "has_static_artifact": bool((ev.static_evidence.get("artifact_scan") or {}).get("introduced")),
        "static_available": bool(ev.static_evidence.get("available")),
        "wa_hint_conf_bin": round(float(hints.get("confidence") or 0.0), 1),
        "has_duplicate_payload": bool(hints.get("duplicate_same_payload_actions")),
        "later_same_file_count": min(10, len(ev.patch_survival.later_same_file_actions)),
        "same_file_edit_count": min(10, int(ev.trajectory_signals.get("same_file_edit_count") or 0)),
        "base_scope": base_action.get("risk_scope") or "",
        "base_is_scorable": (base_action.get("risk_scope") in SCORABLE_SCOPES),
        "base_myopia_bin": round(float(base_action.get("action_myopia_score") or 0.0), 1),
        "base_task_bin": round(float(rv.get("task_advancement") or 0.0), 1),
        "base_regression_bin": round(float(rv.get("regression_surface") or 0.0), 1),
    }
