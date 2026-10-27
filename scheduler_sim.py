#!/usr/bin/env python3
"""
CPU Scheduling Simulator  (v2: adds SRTF with aging)
----------------------------------------------------
Algorithms : FCFS, SJF (non-preemptive), SRTF, Priority (non-preemptive and
             preemptive), Round Robin, and SRTF-Aging (our proposed variant)
Metrics    : avg waiting / turnaround / response time, p95 / p99 / max waiting,
             throughput, CPU utilization, context switches, Jain fairness

Research question
-----------------
SRTF gives the lowest average waiting time but starves long jobs (very large
worst-case waits). Can a single "aging" parameter alpha fix the tail while
keeping most of SRTF's average-case advantage?

SRTF-Aging rule:   key(p) = remaining(p) - alpha * waiting(p)
    waiting(p) = time p has spent in the ready queue so far.
    alpha = 0  -> exactly SRTF.   large alpha -> behaves like FCFS.
The scheduler re-evaluates every `tick` time units (needed because keys change
continuously as processes wait).

Usage
-----
    python scheduler_sim.py --selftest      # correctness checks
    python scheduler_sim.py --all           # sweep -> pick alpha -> held-out test -> paired -> demo
    python scheduler_sim.py --sweep         # only the alpha tuning (seeds 1-10)
    python scheduler_sim.py --heldout       # only the final comparison (seeds 101-130)
    python scheduler_sim.py --paired        # only the paired SRTF-Aging vs SRTF analysis
    python scheduler_sim.py                 # single-seed demo run + Gantt charts
    python scheduler_sim.py --trace t.csv   # add a real trace (arrival,burst[,priority])

Convention: a LOWER priority number means HIGHER priority.
"""
import argparse
import csv
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INF = float("inf")
ALPHAS = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
TUNE_SEEDS = list(range(1, 11))        # used ONLY to choose alpha
TEST_SEEDS = list(range(101, 131))     # held-out seeds, never used for tuning
FEASIBLE_AVG_FACTOR = 1.2              # selection rule (see choose_alpha)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Proc:
    pid: str
    arrival: float
    burst: float
    priority: int = 0
    remaining: float = 0.0
    start: Optional[float] = None
    finish: Optional[float] = None

    def __post_init__(self):
        self.remaining = self.burst


Segment = Tuple[str, float, float]  # (pid or "IDLE", start, end)


def clone(procs: List[Proc]) -> List[Proc]:
    out = [Proc(p.pid, p.arrival, p.burst, p.priority) for p in procs]
    return sorted(out, key=lambda p: (p.arrival, p.pid))


def merge_segments(segs: List[Segment]) -> List[Segment]:
    merged: List[Segment] = []
    for pid, s, e in segs:
        if merged and merged[-1][0] == pid and merged[-1][2] == s:
            merged[-1] = (pid, merged[-1][1], e)
        else:
            merged.append((pid, s, e))
    return merged


# --------------------------------------------------------------------------
# Scheduling algorithms. Each returns (finished_procs, gantt_segments)
# --------------------------------------------------------------------------
def run_nonpreemptive(procs, key):
    ps = clone(procs)
    n, t, i, gantt, ready, done = len(ps), 0.0, 0, [], [], 0
    while done < n:
        while i < n and ps[i].arrival <= t:
            ready.append(ps[i]); i += 1
        if not ready:
            nxt = ps[i].arrival
            gantt.append(("IDLE", t, nxt)); t = nxt
            continue
        p = min(ready, key=key)
        ready.remove(p)
        p.start = t
        gantt.append((p.pid, t, t + p.burst))
        t += p.burst
        p.remaining = 0
        p.finish = t
        done += 1
    return ps, merge_segments(gantt)


def run_preemptive(procs, key, tick=INF, time_key=None):
    """Preemptive scheduler. Re-picks at every arrival/completion, and also
    every `tick` time units if tick is finite. `key` may take (p) or, when
    time_key is True, (p, t) so the key can depend on the current time."""
    ps = clone(procs)
    n, t, i, gantt, ready, done = len(ps), 0.0, 0, [], [], 0
    while done < n:
        while i < n and ps[i].arrival <= t:
            ready.append(ps[i]); i += 1
        if not ready:
            nxt = ps[i].arrival
            gantt.append(("IDLE", t, nxt)); t = nxt
            continue
        p = min(ready, key=(lambda q: key(q, t)) if time_key else key)
        if p.start is None:
            p.start = t
        next_arrival = ps[i].arrival if i < n else INF
        run = min(p.remaining, next_arrival - t, tick)
        gantt.append((p.pid, t, t + run))
        t += run
        p.remaining -= run
        if p.remaining <= 1e-12:
            p.remaining = 0
            p.finish = t
            ready.remove(p)
            done += 1
    return ps, merge_segments(gantt)


def fcfs(procs):
    return run_nonpreemptive(procs, key=lambda p: (p.arrival, p.pid))


def sjf(procs):
    return run_nonpreemptive(procs, key=lambda p: (p.burst, p.arrival, p.pid))


def priority_np(procs):
    return run_nonpreemptive(procs, key=lambda p: (p.priority, p.arrival, p.pid))


def srtf(procs):
    return run_preemptive(procs, key=lambda p: (p.remaining, p.arrival, p.pid))


def priority_p(procs):
    return run_preemptive(procs, key=lambda p: (p.priority, p.arrival, p.pid))


def srtf_aging(procs, alpha, tick=1.0):
    """SRTF with aging: key = remaining - alpha * (time spent waiting)."""
    def key(p, t):
        waited = t - p.arrival - (p.burst - p.remaining)
        return (p.remaining - alpha * waited, p.arrival, p.pid)
    return run_preemptive(procs, key=key, tick=tick, time_key=True)


def round_robin(procs, quantum):
    """RR. Arrivals during a slice are queued BEFORE the pre-empted process."""
    ps = clone(procs)
    n, t, i, gantt, queue, done = len(ps), 0.0, 0, [], deque(), 0
    while done < n:
        while i < n and ps[i].arrival <= t:
            queue.append(ps[i]); i += 1
        if not queue:
            nxt = ps[i].arrival
            gantt.append(("IDLE", t, nxt)); t = nxt
            continue
        p = queue.popleft()
        if p.start is None:
            p.start = t
        run = min(quantum, p.remaining)
        gantt.append((p.pid, t, t + run))
        t += run
        p.remaining -= run
        while i < n and ps[i].arrival <= t:
            queue.append(ps[i]); i += 1
        if p.remaining <= 1e-12:
            p.remaining = 0
            p.finish = t
            done += 1
        else:
            queue.append(p)
    return ps, merge_segments(gantt)


def get_algorithms(quantum, alpha=0.5, tick=1.0):
    return {
        "FCFS": fcfs,
        "SJF": sjf,
        "SRTF": srtf,
        "Priority-NP": priority_np,
        "Priority-P": priority_p,
        f"RR(q={quantum:g})": lambda ps: round_robin(ps, quantum),
        f"SRTF-Aging(a={alpha:g})": lambda ps: srtf_aging(ps, alpha, tick),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def percentile(values, q):
    """Nearest-rank percentile, q in (0, 100]."""
    xs = sorted(values)
    k = max(0, math.ceil(q / 100 * len(xs)) - 1)
    return xs[k]


def compute_metrics(done, gantt):
    n = len(done)
    waiting = [p.finish - p.arrival - p.burst for p in done]
    turnaround = [p.finish - p.arrival for p in done]
    response = [p.start - p.arrival for p in done]
    span = max(p.finish for p in done) - min(p.arrival for p in done)
    busy = sum(p.burst for p in done)
    runs = [s[0] for s in gantt if s[0] != "IDLE"]
    switches = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    slow = [ta / p.burst for ta, p in zip(turnaround, done)]
    jain = (sum(slow) ** 2) / (n * sum(x * x for x in slow))
    return {
        "avg_waiting": sum(waiting) / n,
        "avg_turnaround": sum(turnaround) / n,
        "avg_response": sum(response) / n,
        "p95_waiting": percentile(waiting, 95),
        "p99_waiting": percentile(waiting, 99),
        "max_waiting": max(waiting),
        "throughput": n / span,
        "cpu_util_pct": 100.0 * busy / span,
        "context_switches": switches,
        "jain_fairness": jain,
    }


METRIC_LABELS = {
    "avg_waiting": "Avg waiting time",
    "avg_turnaround": "Avg turnaround time",
    "avg_response": "Avg response time",
    "p95_waiting": "95th pct waiting",
    "p99_waiting": "99th pct waiting",
    "max_waiting": "Max waiting time",
    "cpu_util_pct": "CPU utilization (%)",
    "context_switches": "Context switches",
    "jain_fairness": "Jain fairness (1 = fair)",
    "throughput": "Throughput (jobs/time unit)",
}


def evaluate(procs, quantum, alpha, tick):
    out = {}
    for name, algo in get_algorithms(quantum, alpha, tick).items():
        done, gantt = algo(procs)
        out[name] = (compute_metrics(done, gantt), gantt)
    return out


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------
def workload_textbook():
    return [Proc("P1", 0, 5, 3), Proc("P2", 1, 3, 1), Proc("P3", 2, 8, 4), Proc("P4", 3, 6, 2)]


def workload_synthetic(n=100, seed=1):
    """Poisson-like arrivals, exponential bursts, random priorities 1-5."""
    rng = random.Random(seed)
    procs, t = [], 0
    for k in range(n):
        t += max(0, round(rng.expovariate(1 / 10)))
        burst = max(1, round(rng.expovariate(1 / 8)))
        procs.append(Proc(f"P{k+1}", t, burst, rng.randint(1, 5)))
    return procs


def workload_trace_like(n=200, seed=2):
    """Heavy-tailed (lognormal) bursts and bursty arrivals. SYNTHETIC."""
    rng = random.Random(seed)
    procs, t = [], 0
    for k in range(n):
        gap = 0 if rng.random() < 0.35 else max(1, round(rng.expovariate(1 / 17)))
        t += gap
        burst = max(1, round(rng.lognormvariate(1.5, 1.2)))
        procs.append(Proc(f"P{k+1}", t, burst,
                          rng.choices([1, 2, 3, 4, 5], [5, 15, 40, 30, 10])[0]))
    return procs


GENERATORS = {
    "Synthetic-Mixed": lambda s: workload_synthetic(100, s),
    "Trace-Like-HeavyTail": lambda s: workload_trace_like(200, s),
}


def load_trace_csv(path, limit=500, time_divisor=1.0):
    """CSV header: arrival, burst [, priority]. First arrival shifted to 0."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                a = float(r["arrival"]) / time_divisor
                b = float(r["burst"]) / time_divisor
                pr = int(float(r.get("priority", 0) or 0))
            except (KeyError, ValueError):
                continue
            if b > 0:
                rows.append((a, b, pr))
    rows.sort()
    rows = rows[:limit]
    if not rows:
        raise ValueError("No usable rows. Need columns: arrival, burst[, priority]")
    base = rows[0][0]
    return [Proc(f"T{k+1}", a - base, b, pr) for k, (a, b, pr) in enumerate(rows)]


# --------------------------------------------------------------------------
# Experiment 1: alpha sweep (TUNING seeds only)
# --------------------------------------------------------------------------
def run_sweep(outdir, tick):
    keys = ["avg_waiting", "p99_waiting", "max_waiting", "jain_fairness", "context_switches"]
    table = {}
    fcfs_ref = {}
    for wl, gen in GENERATORS.items():
        workloads = [gen(s) for s in TUNE_SEEDS]
        for a in ALPHAS:
            vals = {k: [] for k in keys}
            for procs in workloads:
                done, g = srtf_aging(procs, a, tick)
                m = compute_metrics(done, g)
                for k in keys:
                    vals[k].append(m[k])
            table[(wl, a)] = {k: (mean(v), stdev(v)) for k, v in vals.items()}
        vals = {k: [] for k in keys}
        for procs in workloads:
            done, g = fcfs(procs)
            m = compute_metrics(done, g)
            for k in keys:
                vals[k].append(m[k])
        fcfs_ref[wl] = {k: (mean(v), stdev(v)) for k, v in vals.items()}

    with open(os.path.join(outdir, "sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "alpha"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for (wl, a), d in table.items():
            w.writerow([wl, a] + [round(d[k][i], 4) for k in keys for i in (0, 1)])
        for wl, d in fcfs_ref.items():
            w.writerow([wl, "FCFS-ref"] + [round(d[k][i], 4) for k in keys for i in (0, 1)])

    best = choose_alpha(table)
    print_sweep(table, fcfs_ref, best)
    plot_sweep(table, fcfs_ref, best, outdir)
    with open(os.path.join(outdir, "chosen_alpha.txt"), "w") as f:
        f.write(str(best))
    return best


def choose_alpha(table):
    """PRE-DECLARED selection rule (fixed before looking at results):
    an alpha is FEASIBLE if, on every workload, its mean avg_waiting is at most
    1.2x that of plain SRTF (alpha = 0). Among feasible alphas > 0 choose the
    one minimising the mean (over workloads) of  p99_waiting(alpha)/p99_waiting(0).
    If no alpha > 0 improves on 1.0, return 0 (aging does not help)."""
    wls = list(GENERATORS)
    best_a, best_score = 0, 1.0
    for a in ALPHAS:
        if a == 0:
            continue
        ok = all(table[(w, a)]["avg_waiting"][0] <= FEASIBLE_AVG_FACTOR * table[(w, 0)]["avg_waiting"][0]
                 for w in wls)
        if not ok:
            continue
        score = mean(table[(w, a)]["p99_waiting"][0] / table[(w, 0)]["p99_waiting"][0] for w in wls)
        if score < best_score:
            best_a, best_score = a, score
    return best_a


def print_sweep(table, fcfs_ref, best):
    for wl in GENERATORS:
        print(f"\n=== Alpha sweep: {wl} (mean over {len(TUNE_SEEDS)} tuning seeds) ===")
        print(f"{'alpha':>8}{'avg_wait':>12}{'p99_wait':>12}{'max_wait':>12}{'jain':>10}{'ctx_sw':>10}")
        for a in ALPHAS:
            d = table[(wl, a)]
            tag = "  <- chosen" if a == best else ("  (= SRTF)" if a == 0 else "")
            print(f"{a:>8g}{d['avg_waiting'][0]:>12.2f}{d['p99_waiting'][0]:>12.2f}"
                  f"{d['max_waiting'][0]:>12.2f}{d['jain_fairness'][0]:>10.3f}"
                  f"{d['context_switches'][0]:>10.1f}{tag}")
        d = fcfs_ref[wl]
        print(f"{'FCFS':>8}{d['avg_waiting'][0]:>12.2f}{d['p99_waiting'][0]:>12.2f}"
              f"{d['max_waiting'][0]:>12.2f}{d['jain_fairness'][0]:>10.3f}{d['context_switches'][0]:>10.1f}")
    print(f"\nChosen alpha (pre-declared rule) = {best}")


def plot_sweep(table, fcfs_ref, best, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, wl in zip(axes, GENERATORS):
        xs = [table[(wl, a)]["avg_waiting"][0] for a in ALPHAS]
        ys = [table[(wl, a)]["p99_waiting"][0] for a in ALPHAS]
        ax.plot(xs, ys, "o-", color="#4C72B0")
        for a, x, y in zip(ALPHAS, xs, ys):
            ax.annotate(f"a={a:g}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.plot(table[(wl, best)]["avg_waiting"][0], table[(wl, best)]["p99_waiting"][0],
                "*", color="#2E8B57", markersize=16, label=f"chosen a={best:g}")
        ax.plot(fcfs_ref[wl]["avg_waiting"][0], fcfs_ref[wl]["p99_waiting"][0],
                "D", color="#C44E52", label="FCFS")
        ax.set_xlabel("Average waiting time (lower = better average)")
        ax.set_ylabel("99th-percentile waiting time (lower = better tail)")
        ax.set_title(wl)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Aging trade-off: average vs tail waiting (a=0 is plain SRTF)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "aging_tradeoff.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Experiment 2: held-out comparison of ALL algorithms (TEST seeds)
# --------------------------------------------------------------------------
def run_heldout(outdir, quantum, alpha, tick):
    keys = ["avg_waiting", "avg_response", "p99_waiting", "max_waiting",
            "jain_fairness", "context_switches"]
    all_res = {}
    for wl, gen in GENERATORS.items():
        acc = {}
        for s in TEST_SEEDS:
            res = evaluate(gen(s), quantum, alpha, tick)
            for name, (m, _) in res.items():
                acc.setdefault(name, {k: [] for k in keys})
                for k in keys:
                    acc[name][k].append(m[k])
        all_res[wl] = {n: {k: (mean(v), stdev(v)) for k, v in d.items()} for n, d in acc.items()}

    with open(os.path.join(outdir, "heldout.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "algorithm"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for wl, res in all_res.items():
            for name, d in res.items():
                w.writerow([wl, name] + [round(d[k][i], 4) for k in keys for i in (0, 1)])

    for wl, res in all_res.items():
        print(f"\n=== Held-out comparison: {wl} (mean over {len(TEST_SEEDS)} unseen seeds) ===")
        print(f"{'Algorithm':<20}{'avg_wait':>11}{'avg_resp':>11}{'p99_wait':>11}{'max_wait':>11}{'jain':>8}{'ctx_sw':>9}")
        for name, d in res.items():
            print(f"{name:<20}{d['avg_waiting'][0]:>11.2f}{d['avg_response'][0]:>11.2f}"
                  f"{d['p99_waiting'][0]:>11.2f}{d['max_waiting'][0]:>11.2f}"
                  f"{d['jain_fairness'][0]:>8.3f}{d['context_switches'][0]:>9.1f}")

    panels = ["avg_waiting", "avg_response", "p99_waiting", "max_waiting"]
    for wl, res in all_res.items():
        names = list(res.keys())
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, mk in zip(axes, panels):
            vals = [res[n][mk][0] for n in names]
            errs = [res[n][mk][1] for n in names]
            colors = ["#2E8B57" if "Aging" in n else "#4C72B0" for n in names]
            ax.bar(range(len(names)), vals, yerr=errs, color=colors, capsize=3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_title(METRIC_LABELS[mk], fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Held-out comparison: {wl} (mean ± std over {len(TEST_SEEDS)} seeds; green = SRTF-Aging)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"heldout_{wl.lower().replace('-', '_')}.png"), dpi=160)
        plt.close(fig)
    return all_res



# --------------------------------------------------------------------------
# Experiment 3: paired per-seed comparison, SRTF-Aging vs SRTF (TEST seeds)
# --------------------------------------------------------------------------
def run_paired(outdir, alpha, tick, n_boot=5000):
    """For each held-out seed, run SRTF and SRTF-Aging on the SAME workload and
    take the difference (Aging - SRTF). Report the mean difference, a bootstrap
    95% confidence interval, and how many seeds aging was lower / higher."""
    metrics = ["avg_waiting", "p99_waiting", "max_waiting", "jain_fairness", "context_switches"]
    rows = []
    for wl, gen in GENERATORS.items():
        diffs = {m: [] for m in metrics}
        for s in TEST_SEEDS:
            procs = gen(s)
            base = compute_metrics(*srtf(procs))
            aged = compute_metrics(*srtf_aging(procs, alpha, tick))
            for m in metrics:
                diffs[m].append(aged[m] - base[m])
        for m in metrics:
            d = diffs[m]
            n = len(d)
            rng = random.Random(0)
            boots = sorted(mean(rng.choices(d, k=n)) for _ in range(n_boot))
            lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
            rows.append((wl, m, mean(d), lo, hi,
                         sum(1 for x in d if x < 0), sum(1 for x in d if x > 0)))

    with open(os.path.join(outdir, "paired.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "metric", "mean_diff_aging_minus_srtf", "ci95_low", "ci95_high",
                    "seeds_aging_lower", "seeds_aging_higher"])
        for r in rows:
            w.writerow([r[0], r[1]] + [round(x, 4) for x in r[2:5]] + [r[5], r[6]])

    print(f"\n=== Paired analysis: SRTF-Aging(a={alpha:g}) minus SRTF, {len(TEST_SEEDS)} unseen seeds ===")
    print(f"{'workload':<22}{'metric':<18}{'mean diff':>11}{'95% CI':>24}{'lower':>7}{'higher':>8}")
    for wl, m, mu, lo, hi, lower, higher in rows:
        print(f"{wl:<22}{m:<18}{mu:>11.3f}{'[%.3f, %.3f]' % (lo, hi):>24}{lower:>7}{higher:>8}")
    return rows

# --------------------------------------------------------------------------
# Demo run (single seed) + charts
# --------------------------------------------------------------------------
def print_table(title, results):
    print(f"\n=== {title} ===")
    cols = ["avg_waiting", "avg_turnaround", "avg_response", "p99_waiting",
            "max_waiting", "context_switches", "jain_fairness"]
    head = f"{'Algorithm':<20}" + "".join(f"{c:>17}" for c in cols)
    print(head)
    print("-" * len(head))
    for name, (m, _) in results.items():
        print(f"{name:<20}" + "".join(f"{m[c]:>17.3f}" for c in cols))


def save_csv(all_results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "algorithm"] + list(METRIC_LABELS.keys()))
        for wl, res in all_results.items():
            for name, (m, _) in res.items():
                w.writerow([wl, name] + [round(m[k], 4) for k in METRIC_LABELS])


def plot_gantt(results, outdir, fname="gantt_textbook.png"):
    names = list(results.keys())
    pids = sorted({s[0] for _, (_, g) in results.items() for s in g if s[0] != "IDLE"})
    cmap = plt.get_cmap("tab10")
    colors = {pid: cmap(k % 10) for k, pid in enumerate(pids)}
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.1 * len(names) + 1), sharex=True)
    for ax, name in zip(axes, names):
        for pid, s, e in results[name][1]:
            if pid == "IDLE":
                continue
            ax.barh(0, e - s, left=s, color=colors[pid], edgecolor="black")
            ax.text((s + e) / 2, 0, pid, ha="center", va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Gantt charts: textbook workload", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)


def plot_quantum_sensitivity(procs, outdir, quanta=(1, 2, 4, 8, 16, 32, 64)):
    wait, resp, sw = [], [], []
    for q in quanta:
        done, g = round_robin(procs, q)
        m = compute_metrics(done, g)
        wait.append(m["avg_waiting"]); resp.append(m["avg_response"]); sw.append(m["context_switches"])
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(quanta, wait, "o-", label="Avg waiting")
    ax1.plot(quanta, resp, "s-", label="Avg response")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Time quantum"); ax1.set_ylabel("Time units")
    ax2 = ax1.twinx()
    ax2.plot(quanta, sw, "^--", color="gray", label="Context switches")
    ax2.set_ylabel("Context switches")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.set_title("Round Robin: effect of time quantum (synthetic workload)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rr_quantum_sensitivity.png"), dpi=160)
    plt.close(fig)


def run_demo(outdir, quantum, alpha, tick, seed, trace=None, trace_limit=500, time_divisor=1.0):
    workloads = {
        "Textbook": workload_textbook(),
        "Synthetic-Mixed": workload_synthetic(100, seed),
        "Trace-Like-HeavyTail": workload_trace_like(200, seed + 1),
    }
    if trace:
        workloads["Real-Trace"] = load_trace_csv(trace, trace_limit, time_divisor)
    all_results = {}
    for name, procs in workloads.items():
        all_results[name] = evaluate(procs, quantum, alpha, tick)
        print_table(f"{name}  ({len(procs)} processes, single seed {seed})", all_results[name])
    save_csv(all_results, os.path.join(outdir, "results.csv"))
    plot_gantt(all_results["Textbook"], outdir)
    plot_quantum_sensitivity(workloads["Synthetic-Mixed"], outdir)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def selftest():
    wl = workload_textbook()
    expected = {"FCFS": 5.75, "SJF": 5.25, "SRTF": 5.0}
    algos = get_algorithms(4, 0.5, 1.0)
    for name, exp in expected.items():
        done, g = algos[name](wl)
        got = compute_metrics(done, g)["avg_waiting"]
        assert math.isclose(got, exp), f"{name}: expected {exp}, got {got}"
        print(f"PASS  {name:<5} avg waiting = {got} (hand-computed {exp})")

    for name, algo in algos.items():
        done, g = algo(wl)
        assert sum(e - s for p, s, e in g if p != "IDLE") == sum(p.burst for p in wl), name
    print("PASS  work conservation for all algorithms (textbook)")

    # Aging with alpha = 0 must reproduce plain SRTF exactly
    for label, procs in [("textbook", wl),
                         ("synthetic", workload_synthetic(100, 5)),
                         ("heavy-tail", workload_trace_like(200, 6))]:
        d1, _ = srtf(procs)
        d2, _ = srtf_aging(procs, 0, 1.0)
        f1 = {p.pid: p.finish for p in d1}
        f2 = {p.pid: p.finish for p in d2}
        assert f1 == f2, f"aging(alpha=0) != SRTF on {label}"
    print("PASS  SRTF-Aging(alpha=0) == SRTF (identical finish times, 3 workloads)")

    # Every process finishes after it arrives and receives exactly its burst
    for procs in (workload_synthetic(100, 7), workload_trace_like(200, 8)):
        for name, algo in get_algorithms(4, 0.5, 1.0).items():
            done, g = algo(procs)
            assert all(p.finish >= p.arrival + p.burst - 1e-9 for p in done), name
            assert abs(sum(e - s for pid, s, e in g if pid != "IDLE") - sum(p.burst for p in procs)) < 1e-6, name
    print("PASS  all algorithms: finish >= arrival+burst and total work conserved (large workloads)")
    print("\nAll self-tests passed.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CPU scheduling simulator with SRTF-Aging")
    ap.add_argument("--quantum", type=float, default=4)
    ap.add_argument("--alpha", type=float, default=None, help="aging strength (default: chosen by sweep)")
    ap.add_argument("--tick", type=float, default=1.0, help="re-evaluation interval for aging")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trace")
    ap.add_argument("--trace-limit", type=int, default=500)
    ap.add_argument("--time-divisor", type=float, default=1.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    os.makedirs(args.outdir, exist_ok=True)

    alpha = args.alpha
    if args.sweep or args.all:
        best = run_sweep(args.outdir, args.tick)
        if alpha is None:
            alpha = best
    if alpha is None:
        p = os.path.join(args.outdir, "chosen_alpha.txt")
        alpha = float(open(p).read()) if os.path.exists(p) else 0.5

    if args.heldout or args.all:
        run_heldout(args.outdir, args.quantum, alpha, args.tick)
    if args.paired or args.all:
        run_paired(args.outdir, alpha, args.tick)
    if args.all or not (args.sweep or args.heldout or args.paired):
        run_demo(args.outdir, args.quantum, alpha, args.tick, args.seed,
                 args.trace, args.trace_limit, args.time_divisor)
    print(f"\nDone. alpha used = {alpha:g}. Files saved in ./{args.outdir}/")


if __name__ == "__main__":
    main()