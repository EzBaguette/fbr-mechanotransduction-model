"""
outputs/results.py  (v3)
=========================
NEW IN V3:
  - Prints immunomodulation_index and trajectory_match_score
  - Prints scaffold mode interpretation when mode='scaffold'
  - Prints coating and tissue_context in surface summary
  - Prints drug_eluting status
"""
import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass
from inputs.surface_params import SurfaceInputs
from phenotype.phenotype_classifier import PhenotypeTimeSeries
 
 
@dataclass
class SimulationResults:
    surface:          SurfaceInputs
    t_minutes:        np.ndarray
    ca_timeseries:    np.ndarray
    yap_nuclear:      np.ndarray
    nfkb_active:      np.ndarray
    stat6_active:     np.ndarray
    phenotype_scores: PhenotypeTimeSeries
    fbgc_risk:        dict
    mode:             str = "suppress"
 
    @property
    def t_days(self):
        return self.t_minutes / (24 * 60)
 
    def print_summary(self):
        s  = self.surface
        ps = self.phenotype_scores
        fb = self.fbgc_risk
 
        print(f"\n{'='*62}")
        print(f"  FBR MECHANOTRANSDUCTION MODEL  — mode: {self.mode.upper()}")
        print(f"{'='*62}")
 
        print(f"\nSurface inputs:")
        print(f"  Stiffness:       {s.E_kPa} kPa")
        print(f"  Roughness Ra:    {s.Ra_um} µm")
        print(f"  Feature size:    {s.feature_size_nm} nm")
        print(f"  Contact angle:   {s.contact_angle_deg}°")
        print(f"  Tissue context:  {s.tissue_context}")        # NEW
        print(f"  Coating:         {s.coating} "
              f"(suppression={s.coating_suppression:.0%})")    # NEW
        print(f"  Drug eluting:    "
              f"{'YES' if s.drug_release_profile else 'no'}")  # NEW
 
        print(f"\nDerived biological scores:")
        print(f"  Mechano-input:      {s.mechano_input_effective:.3f}")
        print(f"  DAMP signal:        {s.damp_signal:.3f}")
        print(f"  Protein adsorption: {s.protein_adsorption_score:.3f}")
        print(f"  M-CSF bias:         {s.mcsf_bias:.3f}")
 
        print(f"\nSteady-state signaling (day 28):")
        print(f"  [Ca2+]i:    {self.ca_timeseries[-1]:.3f} µM")
        print(f"  YAP nuclear:{self.yap_nuclear[-1]:.3f}")
        print(f"  NF-kB:      {self.nfkb_active[-1]:.3f}")
        print(f"  STAT6:      {self.stat6_active[-1]:.3f}")
 
        print(f"\nPhenotype scores (day 28):")
        print(f"  Inflammatory: {ps.inflammatory[-1]:.3f}")
        print(f"  Transitional: {ps.transitional[-1]:.3f}")
        print(f"  Healing:      {ps.healing[-1]:.3f}")
        print(f"  FBGC-prone:   {ps.fbgc_prone[-1]:.3f}")
        print(f"  → Dominant:   {ps.dominant_phenotype()}")
 
        # NEW v3: immunomodulation index
        print(f"\nImmuno-modulation index (M2/M1 AUC): "
              f"{ps.immunomodulation_index:.3f}")
        idx = ps.immunomodulation_index
        if   idx < 0.5:  interp = "M1-dominated — chronic inflammation risk"
        elif idx < 1.5:  interp = "Balanced — mixed response"
        elif idx < 3.0:  interp = "M2-favorable — good biocompatibility"
        else:            interp = "Strong M2 — excellent immunomodulation"
        print(f"  → {interp}")
 
        # NEW v3: scaffold trajectory score
        if self.mode == "scaffold":
            print(f"\nScaffold trajectory match: "
                  f"{ps.trajectory_match_score:.3f} / 1.000")
            if   ps.trajectory_match_score > 0.7:
                print("  → EXCELLENT — surface supports productive fibrosis")
            elif ps.trajectory_match_score > 0.4:
                print("  → MODERATE — consider drug release profile tuning")
            else:
                print("  → POOR — surface unlikely to support guided regeneration")
 
        print(f"\nFBGC Risk:")
        print(f"  Peak risk: {fb['peak_risk']:.3f} ({fb['risk_category']})")
        if fb['onset_day'] > 0:
            print(f"  Onset day: {fb['onset_day']:.1f}")
        else:
            print(f"  Onset day: Not reached")
        print(f"{'='*62}\n")
 
    def save_csv(self, path: Path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["t_days","Ca_uM","YAP_nuclear","NFkB","STAT6",
                        "inflam","transit","healing","fbgc_prone","fbgc_risk"])
            for i, t in enumerate(self.t_days):
                w.writerow([
                    round(t,4),
                    round(self.ca_timeseries[i],5),
                    round(self.yap_nuclear[i],5),
                    round(self.nfkb_active[i],5),
                    round(self.stat6_active[i],5),
                    round(self.phenotype_scores.inflammatory[i],5),
                    round(self.phenotype_scores.transitional[i],5),
                    round(self.phenotype_scores.healing[i],5),
                    round(self.phenotype_scores.fbgc_prone[i],5),
                    round(self.fbgc_risk["risk_timeseries"][i],5),
                ])
        print(f"[Output] Saved to {path}")