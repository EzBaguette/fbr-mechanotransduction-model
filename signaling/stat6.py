"""
signaling/stat6.py
==================
Module 4: STAT6 activation — IL-4/IL-13 late-phase M2a/FBGC signal.
 
Biological basis:
-----------------
IL-4 and IL-13 are produced by Th2 cells and mast cells at the implant site
during the chronic phase of FBR (typically after day 7–14). They signal
through IL-4Rα → JAK1/TYK2 → STAT6 phosphorylation, driving:
  - M2a (healing/anti-inflammatory) phenotype
  - FBGC fusion competency (via mannose receptor, DC-STAMP upregulation)
  - Arginase-1, CD206 expression
 
FBGC note (Stewart et al. Front Immunol 2024; Anderson group):
  IL-4 alone can drive fusion in M-CSF-differentiated macrophages.
  IL-4 + IL-13 together produce larger FBGCs with faster kinetics.
  STAT6 is the gating transcription factor for fusion competency.
 
Model structure:
----------------
Rather than modeling T-cell cytokine production explicitly (out of scope),
we model the IL-4/IL-13 signal as an exogenous wave with:
  - Onset at ~day 7 (chronic phase transition)
  - Peak at ~day 14
  - Sustained plateau at ~day 21–28
  - Amplitude scaled by surface DAMP signal (more DAMP → stronger Th2 response)
 
STAT6 ODE (single-equation activated by IL-4/IL-13 signal):
-----------------------------------------------------------
dSTAT6_p/dt = k_stat6 * IL4_signal * (1 - STAT6_p) - k_stat6_off * STAT6_p
 
Rate constants:
  k_stat6     = 0.02 /min  (STAT6 phosphorylation rate)
  k_stat6_off = 0.005 /min (STAT6 dephosphorylation / turnover)
 
Output: STAT6_phospho fraction [0,1] — the M2a/FBGC permissive signal.
"""
 
import numpy as np
from scipy.integrate import solve_ivp
from inputs.surface_params import SurfaceInputs
 
K_STAT6     = 0.02
K_STAT6_OFF = 0.005
 
# IL-4/IL-13 wave timing (in minutes)
IL4_ONSET_MIN  = 7  * 24 * 60   # day 7
IL4_PEAK_MIN   = 14 * 24 * 60   # day 14
IL4_PLATEAU    = 21 * 24 * 60   # day 21 (plateau)
IL4_RISE_WIDTH = 5  * 24 * 60   # sigmoid rise width ~5 days
 
 
def il4_signal_profile(t_minutes: np.ndarray, damp_score: float) -> np.ndarray:
    """
    Generate IL-4/IL-13 cytokine wave as a smooth time profile.
    Amplitude is scaled by surface DAMP score (higher inflammation → stronger Th2 wave).
    Uses logistic rise + sustained plateau.
    """
    amplitude = 0.4 + 0.5 * damp_score  # range 0.4–0.9 depending on DAMP
 
    # Sigmoid rise from onset
    rise = 1.0 / (1.0 + np.exp(-(t_minutes - IL4_ONSET_MIN) / (IL4_RISE_WIDTH / 5)))
 
    # Slight wane after peak (gradual resolution)
    wane = np.where(
        t_minutes > IL4_PEAK_MIN,
        1.0 - 0.2 * (t_minutes - IL4_PEAK_MIN) / (IL4_PLATEAU - IL4_PEAK_MIN),
        1.0
    )
    wane = np.clip(wane, 0.8, 1.0)  # floor at 80% — signal persists
 
    return amplitude * rise * wane
 
 
class STAT6Module:
    """
    Simulates STAT6 activation via IL-4/IL-13 late-phase wave.
 
    Inputs:
        surface: SurfaceInputs
        t_minutes: np.ndarray — time array
    Output:
        stat6_active: time-series of phospho-STAT6 fraction [0,1]
    """
 
    def __init__(self, surface: SurfaceInputs, t_minutes: np.ndarray):
        self.surface = surface
        self.il4_wave = il4_signal_profile(t_minutes, surface.damp_signal)
        self.t_minutes = t_minutes
 
    def _odes(self, t: float, y: list, t_arr: np.ndarray, il4_arr: np.ndarray) -> list:
        stat6 = y[0]
        il4 = float(np.interp(t, t_arr, il4_arr))
 
        d_stat6 = (K_STAT6 * il4 * (1.0 - stat6)
                   - K_STAT6_OFF * stat6)
        return [d_stat6]
 
    def simulate(self, t_minutes: np.ndarray) -> np.ndarray:
        """Returns phospho-STAT6 fraction time-series [0,1]."""
        y0 = [0.0]
        t_span = (t_minutes[0], t_minutes[-1])
 
        sol = solve_ivp(
            self._odes,
            t_span,
            y0,
            t_eval=t_minutes,
            args=(t_minutes, self.il4_wave),
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
 
        if not sol.success:
            raise RuntimeError(f"STAT6 ODE failed: {sol.message}")
 
        return np.clip(sol.y[0], 0.0, 1.0)