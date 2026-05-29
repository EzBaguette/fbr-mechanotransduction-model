"""
inputs/surface_params.py  (v4 — validation-calibrated)
=========================================================
CHANGES FROM V3:
  1. Nanotopography Hill function: K_feat 200→15 nm, n_feat 1.5→2.0
     Rationale: literature shows 5nm TiO2 nanotubes → M2, 20nm → M1
     Old K_feat=200 could not differentiate 5nm vs 20nm (Δ=0.027)
     New K_feat=15  clearly differentiates: 5nm=0.90, 20nm=0.36
 
  2. DAMP signal: wettability weight increased, direct wettability
     anti-inflammatory bonus added for hydrophilic surfaces.
     Rationale: Hotchkiss 2016 — wettability has STRONGER immunomodulatory
     effect than roughness. Contact angle must dominate DAMP at low values.
 
  3. Healing score wettability bonus: low contact angle directly boosts
     healing score via a wettability_factor passed to phenotype classifier.
     Rationale: hydrophilic surfaces promote integrin signaling pathways
     that directly bias macrophages toward M2 independently of DAMP reduction.
 
  4. tissue_context, coating, drug_release_profile — all preserved from v3.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
 
TISSUE_PRESETS = {
    "brain":       2.0,
    "soft_tissue": 10.0,
    "cartilage":   20.0,
    "bone":        60.0,
}
 
COATING_PRESETS = {
    "none":         0.00,
    "peg":          0.75,
    "zwitterionic": 0.85,
    "phospholipid": 0.90,
    "custom":       None,
}
 
 
@dataclass
class SurfaceInputs:
    E_kPa:             float    = 50.0
    Ra_um:             float    = 0.5
    feature_size_nm:   float    = 200.0
    contact_angle_deg: float    = 70.0
    tissue_context:    str      = "cartilage"
    coating:           str      = "none"
    surface_chemistry_modifier: float = 0.0
    drug_release_profile: Optional[Callable] = field(default=None, repr=False)
 
    def __post_init__(self):
        self._validate()
        self._compute_scores()
 
    def _validate(self):
        assert 0.01  <= self.E_kPa            <= 5000
        assert 0.001 <= self.Ra_um            <= 50
        assert 1     <= self.feature_size_nm  <= 100_000
        assert 0     <= self.contact_angle_deg<= 180
        assert self.tissue_context in TISSUE_PRESETS
        assert self.coating        in COATING_PRESETS
        assert 0.0   <= self.surface_chemistry_modifier <= 1.0
 
    def _compute_scores(self):
        # ── Mechano-input ─────────────────────────────────────────
        K_E = TISSUE_PRESETS[self.tissue_context]
        n_E = 2.0
        self.mechano_input = (self.E_kPa**n_E) / (K_E**n_E + self.E_kPa**n_E)
 
        # FIXED v4: K_feat=15nm, n_feat=2.0 — differentiates sub-50nm features
        K_feat, n_feat = 15.0, 2.0
        self.nanotopo_modifier = 1.0 - (
            self.feature_size_nm**n_feat /
            (K_feat**n_feat + self.feature_size_nm**n_feat)
        )
        # Nanotopo can REDUCE mechano_input (small features reduce Piezo1 drive)
        # Per literature: small nanotubes attenuate inflammatory response
        self.mechano_input_effective = float(np.clip(
            self.mechano_input * (1.0 - 0.4 * self.nanotopo_modifier), 0, 1
        ))
 
        # ── Coating suppression ───────────────────────────────────
        if self.coating == "custom":
            suppression = self.surface_chemistry_modifier
        else:
            suppression = COATING_PRESETS[self.coating]
        self.coating_suppression = suppression
 
        # ── Protein adsorption ───────────────────────────────────
        hydrophobicity  = self.contact_angle_deg / 90.0
        roughness_score = np.log1p(self.Ra_um) / np.log1p(10)
        raw_adsorption  = float(np.clip(
            0.35 * hydrophobicity + 0.65 * roughness_score, 0, 1
        ))
        self.protein_adsorption_score = float(np.clip(
            raw_adsorption * (1.0 - suppression), 0, 1
        ))
 
        # ── DAMP signal (v4: wettability dominant) ────────────────
        # Wettability effect: very hydrophilic surfaces (<30°) strongly
        # suppress DAMP via reduced protein denaturation at surface
        wettability_suppression = float(np.clip(
            1.0 - (self.contact_angle_deg / 90.0), 0, 1
        ))  # 0° → 1.0 suppression, 90°+ → 0 suppression
        raw_damp = float(np.clip(
            0.5 * self.protein_adsorption_score +
            0.5 * (self.contact_angle_deg / 120.0),
            0, 1
        ))
        self.damp_signal = float(np.clip(
            raw_damp * (1.0 - 0.6 * wettability_suppression) * (1.0 - suppression),
            0, 1
        ))
 
        # ── Wettability factor (NEW v4) ───────────────────────────
        # Passed to phenotype classifier to boost healing score
        # Low contact angle → high wettability_factor → higher M2 bias
        self.wettability_factor = float(np.clip(
            1.0 - self.contact_angle_deg / 90.0, 0, 1
        ))
 
        # ── M-CSF bias ────────────────────────────────────────────
        smoothness = 1.0 - roughness_score
        self.mcsf_bias = float(np.clip(
            0.5 * smoothness + 0.5 * (1 - (1.0 - self.mechano_input) * 0.5),
            0, 1
        ))
 
    def summary(self) -> dict:
        return {
            "E_kPa":                    self.E_kPa,
            "Ra_um":                    self.Ra_um,
            "feature_size_nm":          self.feature_size_nm,
            "contact_angle_deg":        self.contact_angle_deg,
            "tissue_context":           self.tissue_context,
            "coating":                  self.coating,
            "coating_suppression":      round(self.coating_suppression, 4),
            "mechano_input_effective":  round(self.mechano_input_effective, 4),
            "nanotopo_modifier":        round(self.nanotopo_modifier, 4),
            "protein_adsorption_score": round(self.protein_adsorption_score, 4),
            "damp_signal":              round(self.damp_signal, 4),
            "wettability_factor":       round(self.wettability_factor, 4),
            "mcsf_bias":                round(self.mcsf_bias, 4),
            "drug_eluting":             self.drug_release_profile is not None,
        }