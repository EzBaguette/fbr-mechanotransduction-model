"""
phenotype/phenotype_classifier.py  (v4 — validation-calibrated)
=================================================================
CHANGE FROM V3:
  - healing score now includes wettability_factor from SurfaceInputs
    Low contact angle directly boosts M2 healing score, consistent with
    Hotchkiss 2016 finding that wettability has stronger immunomodulatory
    effect than roughness alone.
  - np.trapz → np.trapezoid (NumPy 2.0 compatibility)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
 
 
def _scaffold_target(t_minutes: np.ndarray) -> Dict[str, np.ndarray]:
    t_days  = t_minutes / (24 * 60)
    inflam  = 0.5 * np.exp(-0.693 * t_days / 3.0)
    transit = 0.6 * np.exp(-((t_days - 7)**2) / (2 * 3**2))
    healing = 0.7 * (1.0 / (1.0 + np.exp(-(t_days - 10) / 3.0)))
    return {"inflammatory": inflam, "transitional": transit, "healing": healing}
 
 
@dataclass
class PhenotypeTimeSeries:
    inflammatory:           np.ndarray = field(default_factory=lambda: np.array([]))
    transitional:           np.ndarray = field(default_factory=lambda: np.array([]))
    healing:                np.ndarray = field(default_factory=lambda: np.array([]))
    fbgc_prone:             np.ndarray = field(default_factory=lambda: np.array([]))
    t_minutes:              np.ndarray = field(default_factory=lambda: np.array([]))
    immunomodulation_index: float = 0.0
    trajectory_match_score: float = 0.0
 
    def dominant_phenotype(self) -> str:
        final = {
            "Inflammatory": self.inflammatory[-1],
            "Transitional":  self.transitional[-1],
            "Healing":       self.healing[-1],
            "FBGC-prone":    self.fbgc_prone[-1],
        }
        return max(final, key=final.get)
 
    def to_dict(self):
        return {k: getattr(self, k) for k in
                ["inflammatory","transitional","healing","fbgc_prone"]}
 
 
class PhenotypeClassifier:
    def __init__(self, yap_nuclear, nfkb_active, stat6_active,
                 t_minutes, mcsf_bias=0.5, mode="suppress",
                 wettability_factor=0.0):
        self.yap   = yap_nuclear
        self.nfkb  = nfkb_active
        self.stat6 = stat6_active
        self.t     = t_minutes
        self.mcsf_bias = mcsf_bias
        self.mode  = mode
        self.wf    = wettability_factor  # NEW v4
 
    def _inflammatory(self):
        return np.clip(0.6*self.nfkb + 0.4*self.yap, 0, 1)
 
    def _transitional(self):
        nmod = self.nfkb  * (1.0 - self.nfkb)
        smod = self.stat6 * (1.0 - self.stat6)
        return np.clip(4.0 * nmod * (0.5 + smod), 0, 1)
 
    def _healing(self):
        # NEW v4: wettability_factor adds direct M2 boost for hydrophilic surfaces
        # Biological basis: hydrophilic surfaces promote integrin αvβ3/β5 signaling
        # that biases macrophages toward M2 independently of cytokine environment
        # (Hotchkiss 2016: wettability stronger than roughness for M2 induction)
        base_healing = 0.7*self.stat6 + 0.3*(1.0 - self.yap)
        wettability_boost = self.wf * 0.3  # up to +0.3 for fully hydrophilic
        return np.clip(base_healing + wettability_boost, 0, 1)
 
    def _fbgc_prone(self):
        nfkb_low = 1.0 - np.clip(self.nfkb * 2.0, 0, 1)
        return np.clip(self.stat6 * nfkb_low * self.mcsf_bias, 0, 1)
 
    def _immunomodulation_index(self, inflam, healing) -> float:
        auc_m1 = float(np.trapezoid(inflam, self.t)) + 1e-9
        auc_m2 = float(np.trapezoid(healing, self.t))
        return round(auc_m2 / auc_m1, 3)
 
    def _trajectory_match(self, inflam, transit, healing) -> float:
        target = _scaffold_target(self.t)
        err_i  = np.mean((inflam  - target["inflammatory"])**2)
        err_t  = np.mean((transit - target["transitional"])**2)
        err_h  = np.mean((healing - target["healing"])**2)
        rmse   = float(np.sqrt((err_i + err_t + err_h) / 3.0))
        return round(float(np.clip(1.0 - rmse * 3.0, 0, 1)), 3)
 
    def compute(self) -> PhenotypeTimeSeries:
        inflam  = self._inflammatory()
        transit = self._transitional()
        healing = self._healing()
        fbgc    = self._fbgc_prone()
        return PhenotypeTimeSeries(
            inflammatory=inflam, transitional=transit,
            healing=healing, fbgc_prone=fbgc, t_minutes=self.t,
            immunomodulation_index=self._immunomodulation_index(inflam, healing),
            trajectory_match_score=self._trajectory_match(inflam, transit, healing),
        )