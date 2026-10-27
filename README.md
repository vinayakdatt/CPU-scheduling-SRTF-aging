# CPU Scheduling Simulator: SRTF with Aging

Simulator and experiments for the paper *Reducing SRTF Starvation with Aging: A Comparative Study of CPU Scheduling Algorithms under Synthetic and Heavy-Tailed Workloads* (Operating Systems course project, NMIMS Chandigarh, 2026-27).

Authors: Vinayak Datt, Raghav Singla

## What it does

- Simulates a single CPU (no I/O, zero context-switch cost) under FCFS, SJF, SRTF, Priority (non-preemptive and preemptive), Round Robin (q = 4) and **SRTF-Aging**.
- SRTF-Aging picks the process with the smallest `remaining - alpha * waited`. `alpha = 0` is exactly SRTF.
- Reports average waiting / turnaround / response time, p95 / p99 / max waiting time, context switches, throughput, CPU utilization and Jain's fairness index (on slowdown).

## Requirements

Python 3.9+ and matplotlib:

```
pip install -r requirements.txt
```

## Usage

```
python scheduler_sim.py --selftest   # correctness checks (should end with "All self-tests passed.")
python scheduler_sim.py --all        # full pipeline, regenerates everything in results/
```

`--all` runs, in order: the alpha sweep on tuning seeds 1-10 (selection rule fixed in advance), the held-out comparison of all algorithms on unseen seeds 101-130, the paired SRTF-Aging vs SRTF analysis, and a single-seed demo with a Gantt chart. Individual steps: `--sweep`, `--heldout`, `--paired`. All randomness is seeded, so the numbers are reproducible.

## Output (`results/`)

| File | Contents |
|---|---|
| `sweep.csv`, `aging_tradeoff.png` | Alpha sweep (paper Table 1) |
| `heldout.csv`, `heldout_*.png` | All algorithms on 30 unseen seeds (Tables 2 and 3) |
| `paired.csv` | Paired per-seed differences with bootstrap 95% CIs (Table 4) |
| `results.csv`, `gantt_textbook.png`, `rr_quantum_sensitivity.png` | Single-seed demo |
| `chosen_alpha.txt` | Alpha chosen by the pre-declared rule (0.1) |

## Limitations

- Workloads are **synthetic** (a mixed workload and a heavy-tailed lognormal one). The `--trace` option loads a CSV with columns `arrival,burst[,priority]`, but it has **not been tested on real trace data**.
- Single CPU, no I/O, no context-switch cost.
- Only one linear aging form with a 1-unit re-evaluation tick was tested.

See the paper for the full discussion of limitations.
