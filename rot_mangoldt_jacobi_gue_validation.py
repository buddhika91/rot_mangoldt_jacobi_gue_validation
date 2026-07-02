#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact ROT/Mangoldt Jacobi GUE validation runner
================================================

This GitHub-ready validation runner deliberately reuses the exact benchmark
modules used in the discovery runs, instead of re-implementing the operator or
number-variance diagnostics. This avoids silent drift between the research
benchmark and the reproducibility package.

Required files in the same directory:
  - rot_xi_action_density_control_benchmark_TERMINAL.py
  - rot_canonical_density_renorm_control_benchmark.py
  - rot_canonical_rigidity_control_benchmark.py

Recommended N=8000 discovery command:

  python rot_mangoldt_jacobi_gue_validation_v3_exact.py \
      --N 8000 --prime-max 4000 --s0 0.60 --sigma -1.0 --shape cos4 \
      --controls 50 --out-dir rot_gue_N8000_exact_validation

The script generates a verification folder with machine-readable CSV/JSON data,
plots, and a terminal copy-paste report.

This is numerical evidence only. It is not a proof of RH.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import rot_xi_action_density_control_benchmark_TERMINAL as xi
    import rot_canonical_density_renorm_control_benchmark as dens
    import rot_canonical_rigidity_control_benchmark as core
except Exception as e:
    print("ERROR: required benchmark modules are missing or failed to import.")
    print("Put these files in the same folder as this script:")
    print("  rot_xi_action_density_control_benchmark_TERMINAL.py")
    print("  rot_canonical_density_renorm_control_benchmark.py")
    print("  rot_canonical_rigidity_control_benchmark.py")
    print(f"Import error: {e}")
    sys.exit(1)


def piter(it: Iterable, desc: str = "", total: int | None = None):
    if tqdm is not None:
        return tqdm(it, desc=desc, total=total)
    return it


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def clean_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def clean_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: clean_value(v) for k, v in r.items()}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(clean_row(r))


def parse_modes(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def make_exact_args(args: argparse.Namespace) -> SimpleNamespace:
    # This object supplies exactly the fields expected by xi.evaluate_xi / dens.evaluate.
    return SimpleNamespace(
        N=int(args.N),
        prime_max=int(args.prime_max),
        K0=float(args.K0),
        action_mode=str(args.action_mode),
        eps_mode=str(args.eps_mode),
        eps_max_abs=(None if args.eps_max_abs is None or float(args.eps_max_abs) <= 0 else float(args.eps_max_abs)),
        seed=int(args.seed),
        rep_offset=int(args.rep_offset),
        legacy_hash_controls=bool(args.legacy_hash_controls),
        rng_patch="unknown",
        weight_power=float(args.weight_power),
        chunk=int(args.chunk),
        trim=float(args.trim),
        L_values=str(args.L_values),
        nv_windows=int(args.nv_windows),
        nv_weight=float(args.nv_weight),
        hybrid_nv_weight=float(args.hybrid_nv_weight),
        nv_fail_penalty=float(args.nv_fail_penalty),
        score_metric=str(args.score_metric),
        hard_gates=bool(args.hard_gates),
        prefer_nv_pass=bool(args.prefer_nv_pass),
        gate_ks_max=float(args.gate_ks_max),
        gate_r_min=float(args.gate_r_min),
        gate_r_max=float(args.gate_r_max),
        gate_ks_weight=float(args.gate_ks_weight),
        gate_r_weight=float(args.gate_r_weight),
        gate_nv_fail_cost=float(args.gate_nv_fail_cost),
        scale_clip_min=float(args.scale_clip_min),
        scale_clip_max=float(args.scale_clip_max),
        tie_tol=float(args.tie_tol),
    )


def build_arrays_for_plot(exact_args: SimpleNamespace, eps: float, shape: str) -> Dict[str, np.ndarray]:
    # Uses the exact density builder from the discovery benchmark.
    diag, off, features = dens.build_jacobi_density(
        N=exact_args.N,
        prime_max=exact_args.prime_max,
        weight_mode="mangoldt",
        seed=int(exact_args.seed),
        s0=float(args_global.s0),
        eps=float(eps),
        shape=str(shape),
        weight_power=float(exact_args.weight_power),
        chunk=int(exact_args.chunk),
        clip_min=float(exact_args.scale_clip_min),
        clip_max=float(exact_args.scale_clip_max),
        quiet=True,
    )
    eigs = core.eigvals_tridiagonal(diag, off)
    x = core.unfolded_levels(eigs, trim=exact_args.trim)
    spacings, ratios = core.spacings_and_ratios(x)
    L_values = core.parse_L_values(exact_args.L_values)
    nv_obs = core.number_variance(x, L_values, windows=exact_args.nv_windows)
    nv_gue = np.asarray(core.gue_number_variance_tuple(tuple(float(v) for v in L_values)), dtype=float)
    nv_poi = L_values.copy()
    return {
        "diag": diag,
        "off": off,
        "eigenvalues": eigs,
        "unfolded_levels": x,
        "spacings": spacings,
        "ratios": ratios,
        "L_values": L_values,
        "number_variance_observed": nv_obs,
        "number_variance_gue": nv_gue,
        "number_variance_poisson": nv_poi,
        "density_phi": features.get("density_phi", np.array([], dtype=float)),
        "density_s_diag": features.get("density_s_diag", np.array([], dtype=float)),
    }


def compute_summary(mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    sc = xi.score_col(args.score_metric)
    mscore = float(mangoldt[sc])
    ctrl_scores = np.asarray([float(r[sc]) for r in controls], dtype=float) if controls else np.asarray([], dtype=float)
    better = int(np.sum(ctrl_scores < mscore)) if len(ctrl_scores) else 0
    better_tol = int(np.sum(ctrl_scores < mscore - float(args.tie_tol))) if len(ctrl_scores) else 0
    near = int(np.sum(np.abs(ctrl_scores - mscore) <= float(args.tie_tol))) if len(ctrl_scores) else 0
    min_idx = int(np.argmin(ctrl_scores)) if len(ctrl_scores) else -1
    ctrl_min = controls[min_idx] if min_idx >= 0 else None
    ctrl_min_score = float(ctrl_scores[min_idx]) if min_idx >= 0 else float("nan")
    margin = ctrl_min_score - mscore if len(ctrl_scores) else float("nan")
    strict = bool(mangoldt.get("hard_gate_pass") and mangoldt.get("nv_gue_better_than_poisson") and better == 0 and np.isfinite(margin) and margin > 0)
    return {
        "verdict": "PASS_STRICT" if strict else "NOT_STRICT",
        "score_metric": args.score_metric,
        "N": int(args.N),
        "prime_max": int(args.prime_max),
        "s0": float(args.s0),
        "sigma": float(args.sigma),
        "shape": str(args.shape),
        "L_values": str(args.L_values),
        "nv_windows": int(args.nv_windows),
        "mangoldt_score": mscore,
        "mangoldt_KS_GUE": float(mangoldt["ks_to_gue_wigner"]),
        "mangoldt_r_mean": float(mangoldt["r_mean"]),
        "mangoldt_nv_gue": float(mangoldt["nv_rmse_gue"]),
        "mangoldt_nv_poisson": float(mangoldt["nv_rmse_poisson"]),
        "mangoldt_nv_better": bool(mangoldt["nv_gue_better_than_poisson"]),
        "mangoldt_hard_gate_pass": bool(mangoldt["hard_gate_pass"]),
        "control_count": int(len(controls)),
        "control_score_min": ctrl_min_score if len(controls) else None,
        "control_score_median": float(np.median(ctrl_scores)) if len(ctrl_scores) else None,
        "control_score_mean": float(np.mean(ctrl_scores)) if len(ctrl_scores) else None,
        "control_min_margin_minus_mangoldt": margin if len(controls) else None,
        "control_min_weight_mode": ctrl_min.get("weight_mode") if ctrl_min else None,
        "control_min_rep": int(ctrl_min.get("control_rep_requested", ctrl_min.get("rep", -1))) if ctrl_min else None,
        "controls_beating_mangoldt": better,
        "controls_beating_mangoldt_tol": better_tol,
        "controls_near_tie_count": near,
        "tie_tol": float(args.tie_tol),
        "empirical_p_strict": float((better + 1.0) / (len(controls) + 1.0)) if len(controls) else None,
    }


def mode_summary_rows(controls: List[Dict[str, Any]], mangoldt: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    sc = xi.score_col(metric)
    mscore = float(mangoldt[sc])
    out = []
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in controls:
        by[str(r["weight_mode"])].append(r)
    for mode, rows in sorted(by.items()):
        rows2 = sorted(rows, key=lambda r: float(r[sc]))
        scores = np.asarray([float(r[sc]) for r in rows2], dtype=float)
        best = rows2[0]
        out.append({
            "mode": mode,
            "count": len(rows2),
            "min": float(np.min(scores)),
            "median": float(np.median(scores)),
            "mean": float(np.mean(scores)),
            "better": int(np.sum(scores < mscore)),
            "best_rep": int(best.get("control_rep_requested", best.get("rep", -1))),
            "best_KS": float(best["ks_to_gue_wigner"]),
            "best_r": float(best["r_mean"]),
            "best_NVg": float(best["nv_rmse_gue"]),
            "best_NVp": float(best["nv_rmse_poisson"]),
            "best_hard": bool(best["hard_gate_pass"]),
            "best_NVpass": bool(best["nv_gue_better_than_poisson"]),
        })
    return out


def save_datasets(out_dir: str, arrays: Dict[str, np.ndarray], mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], summary: Dict[str, Any], mode_rows: List[Dict[str, Any]]) -> None:
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(data_dir)
    write_csv(os.path.join(data_dir, "all_trials.csv"), [mangoldt] + controls)
    write_csv(os.path.join(data_dir, "control_mode_summary.csv"), mode_rows)
    write_csv(os.path.join(data_dir, "mangoldt_vs_controls.csv"), [summary])
    write_csv(os.path.join(data_dir, "operator_coefficients_mangoldt.csv"), [
        {"n": i + 1, "diag_a_n": float(arrays["diag"][i]), "offdiag_b_n": float(arrays["off"][i]) if i < len(arrays["off"]) else None}
        for i in range(len(arrays["diag"]))
    ])
    write_csv(os.path.join(data_dir, "spectrum_mangoldt.csv"), [
        {"index": i, "eigenvalue": float(v)} for i, v in enumerate(arrays["eigenvalues"])
    ])
    write_csv(os.path.join(data_dir, "unfolded_levels_mangoldt.csv"), [
        {"index": i, "unfolded_level": float(v)} for i, v in enumerate(arrays["unfolded_levels"])
    ])
    write_csv(os.path.join(data_dir, "spacings_mangoldt.csv"), [
        {"index": i, "spacing": float(v)} for i, v in enumerate(arrays["spacings"])
    ])
    write_csv(os.path.join(data_dir, "gap_ratios_mangoldt.csv"), [
        {"index": i, "ratio": float(v)} for i, v in enumerate(arrays["ratios"])
    ])
    write_csv(os.path.join(data_dir, "number_variance_mangoldt.csv"), [
        {"L": float(L), "observed": float(o), "gue_reference": float(g), "poisson_reference": float(p)}
        for L, o, g, p in zip(arrays["L_values"], arrays["number_variance_observed"], arrays["number_variance_gue"], arrays["number_variance_poisson"])
    ])
    write_csv(os.path.join(data_dir, "density_profile_mangoldt.csv"), [
        {"n": i + 1, "phi": float(arrays["density_phi"][i]), "s_n": float(arrays["density_s_diag"][i])}
        for i in range(len(arrays["density_s_diag"]))
    ])


def make_plots(out_dir: str, arrays: Dict[str, np.ndarray], mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if plt is None or args.no_plots:
        return
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    spacings = arrays["spacings"]
    if len(spacings) > 0:
        xs = np.linspace(0, max(4.0, float(np.percentile(spacings, 99.5))), 400)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(spacings, bins=80, density=True, alpha=0.55, label="Mangoldt spacings")
        ax.plot(xs, core.wigner_gue_pdf(xs), label="GUE Wigner surmise")
        ax.set_xlabel("Unfolded nearest-neighbor spacing")
        ax.set_ylabel("Density")
        ax.set_title("Spacing distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "spacing_histogram_mangoldt.png"), dpi=180)
        plt.close(fig)

        s_sorted = np.sort(spacings)
        emp = np.arange(1, len(s_sorted) + 1) / float(len(s_sorted))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(s_sorted, emp, label="Empirical CDF")
        ax.plot(xs, core.wigner_gue_cdf(xs), label="GUE CDF")
        ax.plot(xs, core.poisson_cdf(xs), label="Poisson CDF")
        ax.set_xlabel("Spacing")
        ax.set_ylabel("CDF")
        ax.set_title(f"Spacing CDF, KS(GUE)={mangoldt['ks_to_gue_wigner']:.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "spacing_cdf_mangoldt.png"), dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(arrays["L_values"], arrays["number_variance_observed"], marker="o", label="Observed")
    ax.plot(arrays["L_values"], arrays["number_variance_gue"], marker="o", label="GUE reference")
    ax.plot(arrays["L_values"], arrays["number_variance_poisson"], marker="o", label="Poisson reference")
    ax.set_xlabel("Window length L")
    ax.set_ylabel("Number variance")
    ax.set_title("Mesoscopic number variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "number_variance_mangoldt.png"), dpi=180)
    plt.close(fig)

    if controls:
        sc = xi.score_col(args.score_metric)
        labels = ["Mangoldt"] + [str(r["weight_mode"]) for r in controls]
        vals = [float(mangoldt[sc])] + [float(r[sc]) for r in controls]
        # Plot summarized by mode rather than all 200 bars.
        modes = sorted(set(str(r["weight_mode"]) for r in controls))
        data_labels = ["Mangoldt"] + modes
        data_vals = [float(mangoldt[sc])] + [float(np.min([float(r[sc]) for r in controls if str(r["weight_mode"]) == m])) for m in modes]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(data_labels, data_vals)
        ax.set_xlabel(f"{args.score_metric} score, lower is better")
        ax.set_title("Mangoldt vs best control by mode")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "control_score_by_mode.png"), dpi=180)
        plt.close(fig)

    n = np.arange(1, len(arrays["diag"]) + 1)
    stride = max(1, len(n) // 2500)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n[::stride], arrays["diag"][::stride], label="Diagonal a_n")
    if len(arrays["off"]) > 0:
        ax.plot(n[:-1:stride], arrays["off"][::stride], label="Off-diagonal b_n")
    ax.set_xlabel("n")
    ax.set_title("Jacobi coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "operator_coefficients_mangoldt.png"), dpi=180)
    plt.close(fig)

    if len(arrays["density_s_diag"]) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(n[::stride], arrays["density_s_diag"][::stride], label="s_n")
        ax.set_xlabel("n")
        ax.set_title("Xi-action density profile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "density_profile_mangoldt.png"), dpi=180)
        plt.close(fig)


def write_readme(out_dir: str, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    verdict = summary.get("verdict")
    text = f"""# ROT/Mangoldt Jacobi GUE validation

This folder was generated by `rot_mangoldt_jacobi_gue_validation_v3_exact.py`.

It validates a finite self-adjoint Jacobi operator generated from Mangoldt arithmetic phases and a Xi-action density profile.

## Command configuration

- N: `{args.N}`
- prime_max: `{args.prime_max}`
- s0: `{args.s0}`
- sigma: `{args.sigma}`
- shape: `{args.shape}`
- controls per mode: `{args.controls}`
- control modes: `{args.control_modes}`
- L-values: `{args.L_values}`
- number variance windows: `{args.nv_windows}`
- score metric: `{args.score_metric}`

## Summary

- Verdict: `{verdict}`
- Mangoldt KS(GUE): `{summary.get('mangoldt_KS_GUE')}`
- Mangoldt r mean: `{summary.get('mangoldt_r_mean')}`
- Mangoldt NV RMSE to GUE: `{summary.get('mangoldt_nv_gue')}`
- Mangoldt NV RMSE to Poisson: `{summary.get('mangoldt_nv_poisson')}`
- Controls beating Mangoldt: `{summary.get('controls_beating_mangoldt')}`

## Files

- `summary.json` and `metadata.json`: machine-readable run summary.
- `data/all_trials.csv`: Mangoldt and all control trials.
- `data/mangoldt_vs_controls.csv`: one-row verdict table.
- `data/control_mode_summary.csv`: best/median control performance by mode.
- `data/operator_coefficients_mangoldt.csv`: Jacobi coefficients `a_n`, `b_n`.
- `data/spectrum_mangoldt.csv`: eigenvalues of the finite Jacobi matrix.
- `data/unfolded_levels_mangoldt.csv`: unfolded bulk levels.
- `data/spacings_mangoldt.csv`: nearest-neighbor spacings.
- `data/gap_ratios_mangoldt.csv`: adjacent gap ratios.
- `data/number_variance_mangoldt.csv`: observed/GUE/Poisson number variance.
- `plots/*.png`: visual diagnostics.

This is numerical evidence only and does not prove RH.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def print_report(args: argparse.Namespace, const: Dict[str, float], eps: float, mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], summary: Dict[str, Any], mode_rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("COPY_PASTE_TERMINAL_REPORT_BEGIN")
    print("=" * 100)
    print(
        f"RUN N={args.N} prime_max={args.prime_max} s0={args.s0} sigma={args.sigma} shape={args.shape} "
        f"controls={len(controls)} seed={args.seed} L_values={args.L_values} nv_windows={args.nv_windows} "
        f"score_metric={args.score_metric}"
    )
    print(
        f"XI_ACTION K0={const['K0']:.18g} S_rec={const['action_used']:.15e} "
        f"logLambda={const['log_lambda_dimless']:.15e} eps={eps:.15e}"
    )
    print(
        f"MANGOLDT score={summary['mangoldt_score']:.15e} KS={summary['mangoldt_KS_GUE']:.15e} "
        f"r={summary['mangoldt_r_mean']:.15e} NVg={summary['mangoldt_nv_gue']:.15e} "
        f"NVp={summary['mangoldt_nv_poisson']:.15e} NVpass={int(summary['mangoldt_nv_better'])} "
        f"hard={int(summary['mangoldt_hard_gate_pass'])}"
    )
    print(
        f"CONTROLS verdict={summary['verdict']} count={summary['control_count']} "
        f"ctrl_min={summary['control_score_min']} ctrl_med={summary['control_score_median']} "
        f"margin={summary['control_min_margin_minus_mangoldt']} min_by={summary['control_min_weight_mode']}#{summary['control_min_rep']} "
        f"better={summary['controls_beating_mangoldt']} p_strict={summary['empirical_p_strict']}"
    )
    print("CONTROL_MODE_SUMMARY")
    for r in mode_rows:
        print(
            f"MODE {r['mode']} count={r['count']} min={r['min']:.15e} median={r['median']:.15e} "
            f"better={r['better']} best_rep={r['best_rep']} best_KS={r['best_KS']:.15e} "
            f"best_r={r['best_r']:.15e} best_NVg={r['best_NVg']:.15e} best_NVp={r['best_NVp']:.15e}"
        )
    print("VERDICT_RULE strict_pass = hard=1 AND NVpass=1 AND controls_beating_mangoldt=0 AND margin>0")
    print("COPY_PASTE_TERMINAL_REPORT_END")
    print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact ROT/Mangoldt Jacobi GUE validation runner.")
    p.add_argument("--N", type=int, default=8000)
    p.add_argument("--prime-max", type=int, default=4000)
    p.add_argument("--s0", type=float, default=0.60)
    p.add_argument("--sigma", type=float, default=-1.0)
    p.add_argument("--shape", default="cos4")
    p.add_argument("--controls", type=int, default=50, help="Controls per mode.")
    p.add_argument("--control-modes", default="gaussian,permuted,signflip,random-support")
    p.add_argument("--out-dir", default="rot_gue_N8000_exact_validation")
    p.add_argument("--seed", type=int, default=6783)
    p.add_argument("--rep-offset", type=int, default=0)
    p.add_argument("--legacy-hash-controls", action="store_true")
    p.add_argument("--K0", type=float, default=float(core.CANON["K0"]))
    p.add_argument("--action-mode", choices=["bare", "full"], default="full")
    p.add_argument("--eps-mode", choices=["sqrtlog", "log", "invroot", "rootloglog"], default="sqrtlog")
    p.add_argument("--eps-max-abs", type=float, default=0.0)
    p.add_argument("--L-values", default="1:12:1", help="Discovery profile: 1:12:1. Deep stress can use 1:20:1.")
    p.add_argument("--nv-windows", type=int, default=250, help="Discovery profile: 250. Deep stress can use 800.")
    p.add_argument("--trim", type=float, default=0.15)
    p.add_argument("--score-metric", choices=["local", "deep", "hybrid", "gate"], default="gate")
    p.add_argument("--hard-gates", action="store_true", default=True)
    p.add_argument("--prefer-nv-pass", action="store_true", default=True)
    p.add_argument("--gate-ks-max", type=float, default=0.13)
    p.add_argument("--gate-r-min", type=float, default=0.58)
    p.add_argument("--gate-r-max", type=float, default=0.62)
    p.add_argument("--gate-ks-weight", type=float, default=2.0)
    p.add_argument("--gate-r-weight", type=float, default=6.0)
    p.add_argument("--gate-nv-fail-cost", type=float, default=0.25)
    p.add_argument("--nv-weight", type=float, default=0.25)
    p.add_argument("--hybrid-nv-weight", type=float, default=0.20)
    p.add_argument("--nv-fail-penalty", type=float, default=0.20)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--scale-clip-min", type=float, default=0.02)
    p.add_argument("--scale-clip-max", type=float, default=2.0)
    p.add_argument("--tie-tol", type=float, default=0.001)
    p.add_argument("--no-plots", action="store_true")
    return p


# Global only used to pass s0 into build_arrays_for_plot without changing function signature.
args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = build_parser().parse_args()
    args_global = args
    t0 = time.time()
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "data"))
    ensure_dir(os.path.join(args.out_dir, "plots"))

    exact_args = make_exact_args(args)
    exact_args.rng_patch = xi.patch_core_control_rng(use_legacy_hash=bool(args.legacy_hash_controls), clear_cache=True)
    const = xi.xi_constants(args.K0, action_mode=args.action_mode)
    eps = xi.epsilon_from_sigma(args.N, sigma=args.sigma, action=const["action_used"], eps_mode=args.eps_mode)
    if args.eps_max_abs is not None and float(args.eps_max_abs) > 0:
        eps = float(np.clip(eps, -float(args.eps_max_abs), float(args.eps_max_abs)))

    print("=" * 100)
    print("EXACT ROT MANGOLDT JACOBI GUE VALIDATION")
    print("=" * 100)
    print(f"N                  : {args.N}")
    print(f"prime_max          : {args.prime_max}")
    print(f"candidate           : s0={args.s0}, sigma={args.sigma}, eps={eps:.12e}, shape={args.shape}")
    print(f"K0                 : {const['K0']:.18g}")
    print(f"Xi action S_rec     : {const['action_used']:.12e}")
    print(f"L-values           : {args.L_values}")
    print(f"nv-windows         : {args.nv_windows}")
    print(f"controls            : {args.controls} per mode; modes={args.control_modes}")
    print(f"control RNG         : {exact_args.rng_patch}")
    print(f"output directory    : {args.out_dir}")
    print("=" * 100)

    print("Building/evaluating Mangoldt operator with exact discovery code...")
    mangoldt = xi.evaluate_xi(exact_args, "mangoldt", 0, s0=args.s0, sigma=args.sigma, shape=args.shape, const=const, quiet=True)
    sc = xi.score_col(args.score_metric)
    print(
        f"Mangoldt: score={float(mangoldt[sc]):.12e} KS={float(mangoldt['ks_to_gue_wigner']):.6e} "
        f"r={float(mangoldt['r_mean']):.6e} NVg={float(mangoldt['nv_rmse_gue']):.6e} "
        f"NVp={float(mangoldt['nv_rmse_poisson']):.6e} hard={bool(mangoldt['hard_gate_pass'])}"
    )

    controls: List[Dict[str, Any]] = []
    modes = parse_modes(args.control_modes)
    total = int(args.controls) * len(modes)
    print("Evaluating matched controls with exact discovery code...")
    for mode in piter(modes, desc="control modes", total=len(modes)):
        for rep in range(int(args.controls)):
            controls.append(xi.evaluate_xi(exact_args, mode, rep, s0=args.s0, sigma=args.sigma, shape=args.shape, const=const, quiet=True))

    summary = compute_summary(mangoldt, controls, args)
    mode_rows = mode_summary_rows(controls, mangoldt, args.score_metric)
    arrays = build_arrays_for_plot(exact_args, eps=eps, shape=args.shape)

    metadata = {
        "script": os.path.basename(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib_available": plt is not None,
        "dependencies": [
            "rot_xi_action_density_control_benchmark_TERMINAL.py",
            "rot_canonical_density_renorm_control_benchmark.py",
            "rot_canonical_rigidity_control_benchmark.py",
        ],
        "xi_constants": clean_row(const),
        "epsilon": eps,
        "args": vars(args),
    }
    write_json(os.path.join(args.out_dir, "metadata.json"), metadata)
    write_json(os.path.join(args.out_dir, "summary.json"), clean_row(summary))
    save_datasets(args.out_dir, arrays, mangoldt, controls, summary, mode_rows)
    make_plots(args.out_dir, arrays, mangoldt, controls, args)
    write_readme(args.out_dir, args, summary)
    print_report(args, const, eps, mangoldt, controls, summary, mode_rows)

    print("\nVALIDATION SUMMARY")
    print("-" * 100)
    print(json.dumps(clean_row(summary), indent=2, sort_keys=True))
    print(f"\nWrote verification folder:\n  {os.path.abspath(args.out_dir)}")
    print(f"elapsed_seconds: {time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
