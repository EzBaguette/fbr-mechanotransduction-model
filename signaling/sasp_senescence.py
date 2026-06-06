"""
signaling/sasp_senescence.py  (v5 — NEW)
==========================================
Two-component senescent cell burden + SASP/IL-17 ODE module.

S_total = S_age(tissue, patient_age) + S_material(E_kPa, contact_angle, adsorption)

Sources:
  Tuttle et al. 2020 (Aging) — tissue-specific p16/p21 by age
  Mrozik et al. 2024 — stiffness-induced SIPS threshold 15 kPa
  Yang et al. 2023 (IJMS) — ROS/hydrophobic SIPS
  Chung et al. 2020 (Sci Transl Med) — IL-17/senescence FBR loop
  Herbstein et al. 2024 (Aging Cell) — IL-6 paracrine senescence
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass

# ── Tissue-specific S_age table (young, middle, old) ──────────────
# Values = senescent fraction of stromal cells (p16/p21+)
# Source: Tuttle et al. 2020
S_AGE_TABLE = {
    "soft_tissue": (0.006, 0.035, 0.10),
    "brain":       (0.003, 0.020, 0.06),
    "cartilage":   (0.005, 0.030, 0.09),
    "bone":        (0.004, 0.025, 0.08),
}
AGE_YOUNG_MAX  = 35
AGE_MIDDLE_MAX = 65

# ── ODE rate constants ─────────────────────────────────────────────
K_IMPLANT     = 0.002    # /min — NF-kB drives new local senescence
K_SASP_SPREAD = 0.001    # /min — paracrine IL-6 spread
K_CLEARANCE   = 0.0005   # /min — immune clearance (reduced in aged)
K_SECRETE     = 0.05     # /min — SASP secretion from senescent cells
K_SASP_DECAY  = 0.02     # /min — SASP cytokine clearance (~2-3h HL)
K_IL17_SEN    = 0.0008   # /min — IL-17 drives new senescence
K_IL17_RISE   = 0.010    # /min — IL-17 rise rate (SASP-driven Th17)
K_IL17_DECAY  = 0.005    # /min — IL-17 decay
S_MAX         = 0.80     # maximum senescent fraction


def compute_s_age(tissue_context: str, patient_age: float) -> float:
    key = tissue_context if tissue_context in S_AGE_TABLE else "soft_tissue"
    young, middle, old = S_AGE_TABLE[key]
    if patient_age <= AGE_YOUNG_MAX:
        return float(young * (patient_age / AGE_YOUNG_MAX))
    elif patient_age <= AGE_MIDDLE_MAX:
        frac = (patient_age - AGE_YOUNG_MAX) / (AGE_MIDDLE_MAX - AGE_YOUNG_MAX)
        return float(young + frac * (middle - young))
    else:
        age_capped = min(patient_age, 85)
        frac = (age_capped - AGE_MIDDLE_MAX) / (85 - AGE_MIDDLE_MAX)
        return float(middle + frac * (old - middle))


def compute_s_material(E_kPa: float, contact_angle_deg: float,
                        protein_adsorption_score: float) -> float:
    # 1. Stiffness SIPS (threshold 15 kPa, Mrozik 2024)
    if E_kPa < 15:
        stiff = 0.0
    elif E_kPa < 50:
        stiff = 0.020 * ((E_kPa - 15) / 35)
    elif E_kPa < 200:
        stiff = 0.020 + 0.020 * ((E_kPa - 50) / 150)
    else:
        stiff = 0.040

    # 2. Hydrophobicity/ROS SIPS
    theta = contact_angle_deg
    hydro = 0.0 if theta < 30 else min(0.025 * ((theta - 30) / 90) ** 1.5, 0.025)

    # 3. Protein adsorption → paracrine senescence
    adsorb = 0.0 if protein_adsorption_score < 0.2 else \
             0.020 * ((protein_adsorption_score - 0.2) / 0.8)

    return float(np.clip(stiff + hydro + adsorb, 0.0, 0.10))


@dataclass
class SenescenceInputs:
    patient_age:        float
    tissue_context:     str
    E_kPa:              float
    contact_angle_deg:  float
    protein_adsorption: float

    def __post_init__(self):
        self.s_age      = compute_s_age(self.tissue_context, self.patient_age)
        self.s_material = compute_s_material(
            self.E_kPa, self.contact_angle_deg, self.protein_adsorption)
        self.s_total    = float(np.clip(self.s_age + self.s_material, 0.0, S_MAX))

    def age_label(self) -> str:
        if self.patient_age <= AGE_YOUNG_MAX: return "Young"
        elif self.patient_age <= AGE_MIDDLE_MAX: return "Middle-aged"
        else: return "Older"

    def risk_category(self) -> str:
        s = self.s_total
        if s < 0.02: return "Minimal"
        elif s < 0.05: return "Low"
        elif s < 0.10: return "Moderate"
        elif s < 0.15: return "High"
        else: return "Very High"

    def summary(self) -> dict:
        return {
            "patient_age":    self.patient_age,
            "age_group":      self.age_label(),
            "tissue_context": self.tissue_context,
            "s_age":          round(self.s_age, 5),
            "s_material":     round(self.s_material, 5),
            "s_total":        round(self.s_total, 5),
            "risk_category":  self.risk_category(),
        }


class SASPSenescenceModule:
    def __init__(self, sen_inputs: SenescenceInputs,
                 nfkb_timeseries: np.ndarray, t_minutes: np.ndarray):
        self.sen  = sen_inputs
        self.nfkb = nfkb_timeseries
        self.t    = t_minutes
        age_factor = max(0.3, 1.0 - 0.007 * max(0, self.sen.patient_age - 35))
        self.k_clear = K_CLEARANCE * age_factor

    def _odes(self, t, y, t_arr, nfkb_arr):
        S, SASP, IL17 = [max(v, 0.0) for v in y]
        nfkb   = float(np.interp(t, t_arr, nfkb_arr))
        t_days = t / (24 * 60)
        il17_gate = 1.0 / (1.0 + np.exp(-(t_days - 7) / 2.0))

        dS    = (K_IMPLANT * nfkb * (S_MAX - S)
                 + K_SASP_SPREAD * SASP * (S_MAX - S)
                 + K_IL17_SEN * IL17 * (S_MAX - S)
                 - self.k_clear * S)
        dSASP = K_SECRETE * S - K_SASP_DECAY * SASP
        dIL17 = K_IL17_RISE * SASP * il17_gate - K_IL17_DECAY * IL17
        return [dS, dSASP, dIL17]

    def simulate(self, t_minutes: np.ndarray) -> dict:
        S0    = self.sen.s_total
        SASP0 = float(np.clip(K_SECRETE / K_SASP_DECAY * S0, 0, 1))
        sol = solve_ivp(self._odes, (t_minutes[0], t_minutes[-1]),
                        [S0, SASP0, 0.0], t_eval=t_minutes,
                        args=(t_minutes, self.nfkb),
                        method="RK45", rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"SASP ODE failed: {sol.message}")
        S_ts    = np.clip(sol.y[0], 0, S_MAX)
        SASP_ts = np.clip(sol.y[1], 0, 1)
        IL17_ts = np.clip(sol.y[2], 0, 1)
        return {
            "senescent_fraction": S_ts,
            "sasp_drive":         SASP_ts,
            "il17_timeseries":    IL17_ts,
            "peak_senescent":     float(np.max(S_ts)),
            "peak_sasp":          float(np.max(SASP_ts)),
            "peak_il17":          float(np.max(IL17_ts)),
            "s_age":              self.sen.s_age,
            "s_material":         self.sen.s_material,
            "s_total_initial":    self.sen.s_total,
        }


def sasp_to_nfkb_amplification(sasp_drive: np.ndarray,
                                 s_total: float) -> np.ndarray:
    """Convert SASP timeseries to NF-kB amplification (additive to DAMP)."""
    scale = min(s_total / 0.15, 1.0) * 0.25
    return np.clip(sasp_drive * scale, 0.0, 0.30)
