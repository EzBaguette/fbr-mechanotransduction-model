"""
signaling/nfkb.py  (v3)
========================
NEW IN V3:
  - Drug release suppression integrated into DAMP drive.
    If surface.drug_release_profile is set, the DAMP signal at each
    timepoint is multiplied by (1 - drug_suppression(t)).
    This models anti-inflammatory drug reducing NF-kB activation.
 
All other parameters unchanged from v2 calibration:
  T_HALF_ACUTE=1.5d, CHRONIC_FLOOR=0.04, K_IKBA_SYNTH=0.08
"""
import numpy as np
from scipy.integrate import solve_ivp
from inputs.surface_params import SurfaceInputs
 
K_IKK        = 0.04;   K_IKK_OFF    = 0.02
K_NUC        = 0.06;   K_EXP        = 0.01
K_IKBA       = 0.08;   K_IKBA_SYNTH = 0.08
K_IKBA_DEG   = 0.04;   K_IKBA_BASE  = 0.003
T_HALF_ACUTE  = 1.5 * 24 * 60   # minutes
CHRONIC_FLOOR = 0.04
IKK_A_0 = 0.0;  NFKB_N_0 = 0.05;  IKBA_0 = 0.80
 
 
def damp_drive(t_min: float, damp_score: float,
               drug_fn=None) -> float:
    """
    DAMP signal with optional drug suppression.
    drug_fn(t_min) returns suppression factor [0,1].
    """
    decay   = np.exp(-0.693 * t_min / T_HALF_ACUTE)
    drive   = float(np.clip(
        damp_score * decay + damp_score * CHRONIC_FLOOR, 0, 1
    ))
    # NEW v3: apply drug suppression if profile provided
    if drug_fn is not None:
        suppression = float(np.clip(drug_fn(t_min), 0, 1))
        drive = drive * (1.0 - suppression)
    return drive
 
 
class NFkBModule:
    def __init__(self, surface: SurfaceInputs, ca_timeseries: np.ndarray):
        self.surface     = surface
        self.ca          = ca_timeseries
        self.damp_score  = surface.damp_signal
        self.drug_fn     = surface.drug_release_profile  # NEW v3
 
    def _odes(self, t, y, t_arr, ca_arr):
        ikk, nfkb_n, ikba = y
        ikk    = max(ikk, 0.0)
        nfkb_n = max(nfkb_n, 0.0)
        ikba   = max(ikba, 0.0)
 
        ca    = float(np.interp(t, t_arr, ca_arr))
        drive = damp_drive(t, self.damp_score, self.drug_fn)  # NEW v3
        ca_amp = 1.0 + 0.15 * min(ca / 0.5, 1.0)
 
        d_ikk  = K_IKK * drive * ca_amp * (1.0 - ikk) - K_IKK_OFF * ikk
        nfkb_c = max(1.0 - nfkb_n, 0.0)
        d_nfkb = (K_NUC*ikk*nfkb_c - K_EXP*nfkb_n - K_IKBA*ikba*nfkb_n)
        d_ikba = (K_IKBA_SYNTH*nfkb_n - K_IKBA_DEG*ikk*ikba - K_IKBA_BASE*ikba)
        return [d_ikk, d_nfkb, d_ikba]
 
    def simulate(self, t_minutes: np.ndarray) -> np.ndarray:
        sol = solve_ivp(self._odes, (t_minutes[0], t_minutes[-1]),
                        [IKK_A_0, NFKB_N_0, IKBA_0],
                        t_eval=t_minutes, args=(t_minutes, self.ca),
                        method="RK45", rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"NF-kB ODE failed: {sol.message}")
        return np.clip(sol.y[1], 0, 1)