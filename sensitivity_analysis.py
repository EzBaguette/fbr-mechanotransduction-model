# sensitivity_analysis.py
"""
One-at-a-time sensitivity analysis.
Varies each rate constant in each signaling module ±50%
and measures effect on final phenotype scores.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib, sys

# Base simulation at fixed surface condition
from inputs.surface_params import SurfaceInputs
from signaling.piezo1_calcium import Piezo1CalciumModule
from signaling.yap_hippo import YAPHippoModule
from signaling.nfkb import NFkBModule
from signaling.stat6 import STAT6Module
from phenotype.phenotype_classifier import PhenotypeClassifier
from phenotype.fbgc_risk import FBGCRiskModule
import signaling.piezo1_calcium as p1mod
import signaling.yap_hippo as yapmod
import signaling.nfkb as nfkbmod

T = np.arange(0, 28*24*60, 10)
BASE_SURFACE = dict(E_kPa=50, Ra_um=0.5, contact_angle_deg=70)

def run_base():
    s = SurfaceInputs(**BASE_SURFACE)
    ca   = Piezo1CalciumModule(s).simulate(T)
    yap  = YAPHippoModule(s, ca).simulate(T)
    nfkb = NFkBModule(s, ca).simulate(T)
    stat6 = STAT6Module(s, T).simulate(T)
    ps = PhenotypeClassifier(yap, nfkb, stat6, T, s.mcsf_bias).compute()
    return {
        "inflammatory": ps.inflammatory[-1],
        "healing": ps.healing[-1],
        "fbgc_risk": FBGCRiskModule(s, stat6, nfkb, T).compute_risk()["peak_risk"]
    }

# Parameters to perturb: (module, attribute_name, display_name)
params = [
    (p1mod, "K_IN",    "Piezo1 k_in"),
    (p1mod, "K_OUT",   "Piezo1 k_out"),
    (p1mod, "ALPHA",   "Piezo1 α (actin fb)"),
    (yapmod,"K_NUC",   "YAP k_nuc"),
    (yapmod,"K_LATS",  "YAP LATS K"),
    (nfkbmod,"K_IKK",  "NF-κB k_IKK"),
    (nfkbmod,"K_IKBA_SYNTH", "IκBα synthesis"),
]

base = run_base()
rows = []

for mod, attr, name in params:
    orig = getattr(mod, attr)
    for factor, label in [(0.5, "-50%"), (1.5, "+50%")]:
        setattr(mod, attr, orig * factor)
        result = run_base()
        setattr(mod, attr, orig)  # restore
        rows.append({
            "parameter": name,
            "perturbation": label,
            "Δ_inflammatory": result["inflammatory"] - base["inflammatory"],
            "Δ_healing":      result["healing"] - base["healing"],
            "Δ_fbgc_risk":    result["fbgc_risk"] - base["fbgc_risk"],
        })

df = pd.DataFrame(rows)
df.to_csv("sensitivity_results.csv", index=False)

# Tornado plot for FBGC risk sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
param_names = df["parameter"].unique()
for i, pname in enumerate(param_names):
    subset = df[df["parameter"] == pname]
    lo = subset[subset["perturbation"] == "-50%"]["Δ_fbgc_risk"].values[0]
    hi = subset[subset["perturbation"] == "+50%"]["Δ_fbgc_risk"].values[0]
    ax.barh(i, hi, left=0, color="#E24B4A", alpha=0.7, height=0.5)
    ax.barh(i, lo, left=0, color="#1D9E75", alpha=0.7, height=0.5)

ax.set_yticks(range(len(param_names)))
ax.set_yticklabels(param_names, fontsize=10)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("ΔFBGC risk from baseline", fontsize=11)
ax.set_title("Sensitivity analysis — FBGC risk\n(±50% parameter perturbation, E=50 kPa baseline)", fontsize=11)
plt.tight_layout()
plt.savefig("sensitivity_tornado.png", dpi=150)
plt.show()
print(df.to_string())