# Baselines

CoTRA is compared against four baselines under one shared output schema.

| ID | Name | Where |
|----|------|-------|
| B1 | Evidence + rules | this repo: `scripts/run_baselines.sh` (tag `rules-only-nollm`) |
| B2 | ProcCtrlBench | **external package** (see below) |
| B3 | Evidence + LLM (no fixed scope gate) | this repo: `scripts/run_baselines.sh` (tag `abl-nogate`) |
| B4 | End-to-end LLM judge | this repo: `scripts/run_e2e.sh` (tag `e2e-judge`) |

## B2: ProcCtrlBench

ProcCtrlBench is a separate process-control benchmark/package and is not
vendored here. To reproduce B2:

1. Obtain the `procctrlbench` package (its own repository).
2. Run its detectors over the same trajectories to produce per-action /
   per-trajectory predictions.
3. Map its outputs onto the CoTRA schema (`risk_scope` + the derived scores)
   using its `mapping`/`metrics` utilities, then evaluate with
   `scripts/eval_predictions.py` exactly like the other methods.

We evaluate B2 from its **uncalibrated** outputs, so that it does not borrow
CoTRA's learned scope mapping. The published B2 scope agreement is
Cohen's kappa = 0.431.
