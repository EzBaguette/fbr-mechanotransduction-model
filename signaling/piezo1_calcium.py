"""
signaling/piezo1_calcium.py
============================
Module 1: Piezo1 channel → intracellular Ca2+ dynamics ODE.
 
PARAMETER CALIBRATION (v2):
-----------------------------
Previous values produced Ca2+ ss of 2.85 µM at 50 kPa — approximately
5-10x above Salsa6f reporter measurements in Bhatt et al. 2021.
 
Root cause: K_IN was too high, K_OUT too low, and no basal leak was
present to maintain the resting Ca2+ floor at ~0.1 µM.
 
FIX: Added k_leak (basal Ca2+ leak independent of Piezo1).
     Reduced k_in from 0.05 → 0.010 µM/min.
     Increased k_out from 0.02 → 0.024 /min.
 
Verified steady states (analytical):
  1 kPa   → 0.101 µM  (resting, matches experimental baseline)
  20 kPa  → 0.366 µM  (moderate activation)
  50 kPa  → 0.575 µM  (within Bhatt 2021 Salsa6f range: 0.3–0.8 µM)
  200 kPa → 0.579 µM  (plateau — Piezo1 saturates above ~100 kPa)
"""
 
import numpy as np
from scipy.integrate import solve_ivp
from inputs.surface_params import SurfaceInputs
 
# ── Rate constants (v2 — calibrated against Bhatt et al. 2021) ──────
K_LEAK  = 0.0024   # µM/min — basal Ca2+ leak (maintains 0.1 µM at rest)
K_IN    = 0.010    # µM/min — Piezo1-driven Ca2+ influx (was 0.05)
K_OUT   = 0.024    # /min   — SERCA + PMCA clearance (was 0.02)
K_ACTIN = 0.010    # /min/µM — actin polymerization driven by Ca2+
K_DEACT = 0.005    # /min   — actin depolymerization
ALPHA   = 0.15     # feedback strength (stability bound: < K_OUT/(2*K_IN) = 1.2)
 
CA_BASAL = 0.10    # µM — resting [Ca2+]i
FA_BASAL = 0.20    # normalized basal F-actin
 
 
class Piezo1CalciumModule:
    """
    Simulates Piezo1-mediated Ca2+ influx with basal leak correction.
 
    Inputs: SurfaceInputs object
    Output: [Ca2+]i time-series in µM
    """
 
    def __init__(self, surface: SurfaceInputs):
        self.surface = surface
        self.P_open = surface.mechano_input_effective
 
    def _odes(self, t: float, y: list) -> list:
        ca, f_actin = y
        ca = max(ca, 0.0)
        f_actin_c = min(max(f_actin, 0.0), 1.0)  # clamp to [0,1]
 
        d_ca = (K_LEAK
                + K_IN * self.P_open * (1.0 + ALPHA * f_actin_c)
                - K_OUT * ca)
 
        d_factin = K_ACTIN * ca - K_DEACT * f_actin
 
        return [d_ca, d_factin]
 
    def simulate(self, t_minutes: np.ndarray) -> np.ndarray:
        """Returns [Ca2+]i time-series in µM."""
        y0 = [CA_BASAL, FA_BASAL]
        t_span = (t_minutes[0], t_minutes[-1])
 
        sol = solve_ivp(
            self._odes,
            t_span,
            y0,
            t_eval=t_minutes,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
 
        if not sol.success:
            raise RuntimeError(f"Piezo1 Ca2+ ODE failed: {sol.message}")
 
        return sol.y[0]