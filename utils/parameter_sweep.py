"""
utils/parameter_sweep.py
=========================
Run a grid sweep over surface parameters and collect final phenotype scores.
Useful for bifurcation analysis and surface design optimization.
 
Example usage:
    from utils.parameter_sweep import run_sweep
    df = run_sweep(
        stiffness_range=[1, 10, 50, 100, 200],
        roughness_range=[0.1, 0.5, 1.0, 5.0],
        t_days=28
    )
    df.to_csv("sweep_results.csv", index=False)
"""
 
import numpy as np
import pandas as pd
from itertools import product
from main import run_simulation
 
 
def run_sweep(
    stiffness_range=None,
    roughness_range=None,
    feature_size_range=None,
    contact_angle_range=None,
    t_days: float = 28.0,
    dt_min: float = 10.0,  # coarser timestep for sweep efficiency
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid sweep over surface parameters.
    Returns a DataFrame with one row per condition.
    """
 
    if stiffness_range is None:
        stiffness_range = [1, 5, 20, 50, 100, 200]
    if roughness_range is None:
        roughness_range = [0.05, 0.5, 2.0]
    if feature_size_range is None:
        feature_size_range = [100]
    if contact_angle_range is None:
        contact_angle_range = [40, 70, 100]
 
    conditions = list(product(
        stiffness_range, roughness_range, feature_size_range, contact_angle_range
    ))
 
    rows = []
    for i, (E, Ra, feat, angle) in enumerate(conditions):
        if verbose:
            print(f"[Sweep {i+1}/{len(conditions)}] E={E} kPa | Ra={Ra} µm | θ={angle}°")
 
        try:
            res = run_simulation(
                E_kPa=E, Ra_um=Ra, feature_size_nm=feat,
                contact_angle_deg=angle, t_days=t_days,
                dt_min=dt_min, save_outputs=False, plot=False
            )
            ps = res.phenotype_scores
            fb = res.fbgc_risk
 
            rows.append({
                "E_kPa": E,
                "Ra_um": Ra,
                "feature_size_nm": feat,
                "contact_angle_deg": angle,
                "mechano_input": res.surface.mechano_input_effective,
                "damp_signal": res.surface.damp_signal,
                "Ca_final_uM": res.ca_timeseries[-1],
                "YAP_nuclear_final": res.yap_nuclear[-1],
                "NFkB_final": res.nfkb_active[-1],
                "STAT6_final": res.stat6_active[-1],
                "score_inflammatory": ps.inflammatory[-1],
                "score_transitional": ps.transitional[-1],
                "score_healing": ps.healing[-1],
                "score_fbgc_prone": ps.fbgc_prone[-1],
                "dominant_phenotype": ps.dominant_phenotype(),
                "fbgc_peak_risk": fb["peak_risk"],
                "fbgc_risk_category": fb["risk_category"],
                "fbgc_onset_day": fb["onset_day"],
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
 
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n[Sweep] Complete. {len(df)} conditions simulated.")
    return df