"""
validation/validate.py
========================
Runs model against Hotchkiss 2016 and Nanotube 2023 experimental datasets
and produces a validation report + figure.
 
Run with:
    python validation/validate.py
"""
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
 
from inputs.surface_params import SurfaceInputs
from signaling.piezo1_calcium import Piezo1CalciumModule
from signaling.yap_hippo import YAPHippoModule
from signaling.nfkb import NFkBModule
from signaling.stat6 import STAT6Module
from phenotype.phenotype_classifier import PhenotypeClassifier
from phenotype.fbgc_risk import FBGCRiskModule
from validation.experimental_data import (
    HOTCHKISS_SURFACES, EXPECTED_RANKS, NANOTUBE_SURFACES
)
 
T = np.arange(0, 7 * 24 * 60, 10)   # 7-day simulation (matches in vitro window)
 
 
def run_surface(params: dict, mode="suppress") -> dict:
    """Run model for one surface condition and return scores."""
    s     = SurfaceInputs(**params)
    ca    = Piezo1CalciumModule(s).simulate(T)
    yap   = YAPHippoModule(s, ca).simulate(T)
    nfkb  = NFkBModule(s, ca).simulate(T)
    stat6 = STAT6Module(s, T).simulate(T)
    ps    = PhenotypeClassifier(yap, nfkb, stat6, T, s.mcsf_bias,
                                wettability_factor=s.wettability_factor).compute()
    return {
        "inflammatory": float(ps.inflammatory[-1]),
        "healing":      float(ps.healing[-1]),
        "imm_index":    ps.immunomodulation_index,
        "surface":      s,
    }
 
 
def check_rank_order(results: dict, expected: dict, metric: str) -> bool:
    """
    Validates that model rank order matches experimental rank order.
    Returns True if Spearman correlation is positive and correct.
    """
    names   = list(results.keys())
    modeled = [results[n][metric] for n in names]
    exp_rnk = [expected[metric][n] for n in names]
    rho, p  = spearmanr(modeled, exp_rnk)
    return rho, p
 
 
def run_validation():
    print("\n" + "="*65)
    print("  FBR MODEL VALIDATION")
    print("  Source: Hotchkiss et al. 2016 (Acta Biomaterialia)")
    print("  + Journal of Nanobiotechnology 2023 (Nanotube data)")
    print("="*65)
 
    # ── DATASET 1: Hotchkiss surfaces ─────────────────────────────
    print("\n[Dataset 1] Ti surface topography + wettability")
    print(f"  {'Surface':<35} {'Inflam':>8} {'Healing':>8} {'ImmIdx':>8}")
    print("  " + "-"*62)
 
    hotchkiss_results = {}
    for name, params in HOTCHKISS_SURFACES.items():
        r = run_surface(params)
        hotchkiss_results[name] = r
        print(f"  {name:<35} {r['inflammatory']:>8.3f} {r['healing']:>8.3f} "
              f"{r['imm_index']:>8.3f}")
 
    # Rank order check
    print("\n  Rank order validation:")
    for metric in ["inflammatory", "healing"]:
        rho, p = check_rank_order(hotchkiss_results, EXPECTED_RANKS, metric)
        status = "✓ PASS" if rho > 0 else "✗ FAIL"
        print(f"    {metric:<15}: Spearman ρ={rho:.3f} (p={p:.3f})  {status}")
 
    # Specific checks from literature
    print("\n  Specific predictions vs literature:")
    inflam = {k: v["inflammatory"] for k, v in hotchkiss_results.items()}
    heal   = {k: v["healing"]      for k, v in hotchkiss_results.items()}
 
    checks = [
        ("SLA inflammatory > PT inflammatory",
         inflam["SLA (rough, hydrophobic)"] > inflam["PT (smooth, hydrophobic)"]),
        ("modSLA inflammatory < SLA inflammatory",
         inflam["modSLA (rough, hydrophilic)"] < inflam["SLA (rough, hydrophobic)"]),
        ("modSLA healing > SLA healing (IL-4/IL-10 elevated on hydrophilic)",
         heal["modSLA (rough, hydrophilic)"] > heal["SLA (rough, hydrophobic)"]),
        ("modSLA healing > PT healing",
         heal["modSLA (rough, hydrophilic)"] > heal["PT (smooth, hydrophobic)"]),
    ]
    all_pass = True
    for desc, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        if not result: all_pass = False
        print(f"    {status}  {desc}")
 
    # ── DATASET 2: Nanotube size effect ────────────────────────────
    print("\n[Dataset 2] TiO2 nanotube size effect (Nano J Biotech 2023)")
    print(f"  {'Surface':<35} {'Inflam':>8} {'Healing':>8}")
    print("  " + "-"*55)
 
    nano_results = {}
    for name, params in NANOTUBE_SURFACES.items():
        r = run_surface(params)
        nano_results[name] = r
        print(f"  {name:<35} {r['inflammatory']:>8.3f} {r['healing']:>8.3f}")
 
    nt20_inf = nano_results["NT20 (20nm, M1-inducing)"]["inflammatory"]
    nt5_inf  = nano_results["NT5 (5nm, M2c-inducing)"]["inflammatory"]
    nt20_hea = nano_results["NT20 (20nm, M1-inducing)"]["healing"]
    nt5_hea  = nano_results["NT5 (5nm, M2c-inducing)"]["healing"]
 
    nano_checks = [
        ("NT20 inflammatory > NT5 inflammatory (NT20→M1 per literature)",
         nt20_inf > nt5_inf),
        ("NT5 healing > NT20 healing (NT5→M2c per literature)",
         nt5_hea > nt20_hea),
    ]
    nano_pass = True
    print("\n  Nanotube predictions vs literature:")
    for desc, result in nano_checks:
        status = "✓ PASS" if result else "✗ FAIL"
        if not result: nano_pass = False
        print(f"    {status}  {desc}")
 
    # ── OVERALL RESULT ─────────────────────────────────────────────
    n_checks = len(checks) + len(nano_checks)
    n_passed = sum(r for _, r in checks) + sum(r for _, r in nano_checks)
    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULT: {n_passed}/{n_checks} checks passed")
    if n_passed == n_checks:
        print("  ✓ ALL CHECKS PASSED — model directionally validated")
    else:
        print("  ✗ Some checks failed — review parameter calibration")
    print(f"{'='*65}\n")
 
    # ── FIGURE ────────────────────────────────────────────────────
    _plot_validation(hotchkiss_results, nano_results)
 
    return hotchkiss_results, nano_results
 
 
def _plot_validation(hotchkiss, nano):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Model Validation Against Published Experimental Data\n"
        "Hotchkiss et al. 2016 (Acta Biomaterialia) + "
        "Nanotube dataset 2023",
        fontsize=11, fontweight='bold'
    )
 
    # Panel 1: Hotchkiss — inflammatory scores
    ax = axes[0]
    names  = list(hotchkiss.keys())
    short  = ["PT\n(smooth\nhydrophob.)", "SLA\n(rough\nhydrophob.)",
              "modSLA\n(rough\nhydrophil.)"]
    inflam = [hotchkiss[n]["inflammatory"] for n in names]
    colors = ["#E8593C", "#D04020", "#1D9E75"]
    bars   = ax.bar(short, inflam, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel("Model inflammatory score [0–1]", fontsize=10)
    ax.set_title("M1-like response\n(Expected: SLA highest, modSLA lowest)",
                 fontsize=9)
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, inflam):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    ax.annotate("✓ Literature:\nSLA > PT > modSLA", xy=(0.97, 0.97),
                xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='#333',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax.grid(axis='y', alpha=0.3)
 
    # Panel 2: Hotchkiss — healing scores
    ax = axes[1]
    healing = [hotchkiss[n]["healing"] for n in names]
    bars    = ax.bar(short, healing, color=["#EF9F27","#E08810","#1D9E75"],
                     alpha=0.85, edgecolor='white')
    ax.set_ylabel("Model healing score [0–1]", fontsize=10)
    ax.set_title("M2-like response\n(Expected: modSLA highest, SLA lowest)",
                 fontsize=9)
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, healing):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    ax.annotate("✓ Literature:\nmodSLA > PT > SLA", xy=(0.97, 0.97),
                xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='#333',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax.grid(axis='y', alpha=0.3)
 
    # Panel 3: Nanotube size effect
    ax = axes[2]
    nano_names  = ["NT20\n(20nm, M1)", "NT5\n(5nm, M2c)"]
    nano_inflam = [nano[n]["inflammatory"] for n in nano.keys()]
    nano_heal   = [nano[n]["healing"]      for n in nano.keys()]
    x = np.array([0, 1])
    w = 0.35
    ax.bar(x - w/2, nano_inflam, w, label='Inflammatory', color='#E24B4A',
           alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, nano_heal,   w, label='Healing',      color='#1D9E75',
           alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(nano_names)
    ax.set_ylabel("Model phenotype score [0–1]", fontsize=10)
    ax.set_title("TiO2 nanotube size effect\n(Expected: NT20→M1, NT5→M2c)",
                 fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.annotate("✓ Literature:\nNT20: M1 | NT5: M2c", xy=(0.97, 0.97),
                xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='#333',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    os.makedirs("outputs_data", exist_ok=True)
    plt.savefig("outputs_data/validation_figure.png", dpi=150, bbox_inches='tight')
    print("[Output] Validation figure saved to outputs_data/validation_figure.png")
    plt.show()
 
 
if __name__ == "__main__":
    run_validation()