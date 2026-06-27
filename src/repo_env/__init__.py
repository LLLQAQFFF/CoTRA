"""Trajectory replay infrastructure (Phase 2.4b).

Given a SWE-Bench Pro trajectory, this module can:
  - Resolve the upstream repo URL + base_commit (metadata.resolve_trajectory_env)
  - Clone / checkout that revision into a sandboxed workdir
  - Apply normalized actions sequentially (action_applier)
  - Produce per-step CodebaseSnapshot for downstream static analyzers
"""
