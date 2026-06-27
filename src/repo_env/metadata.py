"""Resolve a trajectory's upstream repo URL and base_commit.

The human-target template carries `trajectory_meta.sample_id` + `model`, but
not the base_commit. The actual repo + commit live in the raw `.traj` file's
`replay_config` JSON string. We follow `trajectory_meta.source_traj` (or
reconstruct the path from sample_id + model) and parse it.

Example replay_config snippet:
  {"env": {"deployment": {"image": "...sweap-images/flipt-io.flipt:flipt-io__flipt-c154...289"},
           "repo": {"repo_name": "app", "base_commit": "e432032cf2d11...daa", ...}}}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


# image path looks like ".../sweap-images/<owner>.<repo>:<owner>__<repo>-<sha>"
_IMAGE_OWNER_REPO_RE = re.compile(
    r"sweap-images/([^.]+)\.([^:/\s]+):"
)


@dataclass(frozen=True)
class TrajectoryEnv:
    """All metadata needed to recreate the agent's starting environment."""

    instance_id: str             # e.g. "instance_flipt-io__flipt-c154dd1a..."
    model: str                   # e.g. "claude-4sonnet-10132025"
    owner: str                   # e.g. "flipt-io"
    repo: str                    # e.g. "flipt"
    base_commit: str             # 40-char hex from replay_config
    repo_url: str                # "https://github.com/<owner>/<repo>.git"
    container_workdir: str       # in-container path the agent saw, e.g. "/app"
    raw_traj_path: Path          # path to the raw .traj


def _resolve_raw_traj_path(template_path: Path, model: str, sample_id: str,
                            data_root: Path | None = None) -> Path:
    """Find the raw .traj file for a given trajectory template.

    Layout: `<data_root>/<model>/traj/<sample_id>/<sample_id>.traj`.

    `data_root` defaults to the project's `data/` dir, inferred from template_path
    which lives at `<data_root>/human-target[-2]/<set>/<instance_id>/<model>.target.template.json`.
    """
    if data_root is None:
        # template_path: .../data/human-target[-2]/<set>/<instance>/<model>.target.template.json
        p = template_path.resolve()
        # Walk up to find "data" directory.
        for ancestor in p.parents:
            if ancestor.name in {"human-target", "human-target-2"}:
                data_root = ancestor.parent
                break
        else:
            raise ValueError(
                f"Could not infer data_root from template path: {template_path}"
            )
    candidate = data_root / model / "traj" / sample_id / f"{sample_id}.traj"
    if not candidate.exists():
        raise FileNotFoundError(f"Raw .traj not found at {candidate}")
    return candidate


def _parse_owner_repo_from_image(image: str) -> tuple[str, str]:
    m = _IMAGE_OWNER_REPO_RE.search(image)
    if not m:
        raise ValueError(f"Could not parse owner/repo from image: {image!r}")
    return m.group(1), m.group(2)


def _parse_owner_repo_from_instance_id(instance_id: str) -> tuple[str, str]:
    """Parse `instance_<owner>__<repo>-<sha>...` pattern."""
    body = instance_id
    if body.startswith("instance_"):
        body = body[len("instance_"):]
    if "__" not in body:
        raise ValueError(f"Unexpected instance_id format: {instance_id!r}")
    owner, rest = body.split("__", 1)
    # rest is like `<repo>-<sha>[-v<variant>]`. The repo can contain dashes.
    # Strip trailing `-v...` variant tag if present.
    rest = re.sub(r"-v(?:[a-f0-9]+|nan)$", "", rest)
    # Strip the trailing `-<40-hex>` base-instance commit.
    rest = re.sub(r"-[a-f0-9]{40}$", "", rest)
    if not rest:
        raise ValueError(f"Could not split repo from instance_id: {instance_id!r}")
    return owner, rest


def _parse_replay_config(traj_blob: dict) -> dict:
    rc = traj_blob.get("replay_config")
    if isinstance(rc, str):
        return json.loads(rc)
    if isinstance(rc, dict):
        return rc
    raise ValueError("traj has no replay_config")


def resolve_trajectory_env(template_path: Path,
                            data_root: Path | None = None) -> TrajectoryEnv:
    """Build a TrajectoryEnv from a target.template.json path."""
    template_path = Path(template_path)
    with open(template_path) as f:
        template = json.load(f)
    meta = template.get("trajectory_meta") or {}
    model = meta.get("model")
    sample_id = meta.get("sample_id")
    if not model or not sample_id:
        raise ValueError(
            f"Template missing trajectory_meta.model or .sample_id: {template_path}"
        )

    raw_traj = _resolve_raw_traj_path(template_path, model, sample_id, data_root)
    with open(raw_traj) as f:
        traj_blob = json.load(f)
    rc = _parse_replay_config(traj_blob)
    env_cfg = rc.get("env") or {}
    repo_cfg = env_cfg.get("repo") or {}
    base_commit = repo_cfg.get("base_commit")
    if not base_commit:
        raise ValueError(f"No base_commit in replay_config for {sample_id}")
    container_workdir = "/" + (repo_cfg.get("repo_name") or "app")

    image = (env_cfg.get("deployment") or {}).get("image", "")
    try:
        owner, repo = _parse_owner_repo_from_instance_id(sample_id)
    except ValueError:
        owner, repo = _parse_owner_repo_from_image(image)

    return TrajectoryEnv(
        instance_id=sample_id,
        model=model,
        owner=owner,
        repo=repo,
        base_commit=base_commit,
        repo_url=f"https://github.com/{owner}/{repo}.git",
        container_workdir=container_workdir,
        raw_traj_path=raw_traj,
    )
