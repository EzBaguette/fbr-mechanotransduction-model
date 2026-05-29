"""
signaling/yap_hippo.py
========================
Module 2: YAP nuclear/cytoplasmic translocation ODE (Hippo pathway).
 
Biological basis:
-----------------
In macrophages, YAP (not TAZ — see Golgi-TAZ note below) responds to
mechanical inputs via the LATS1/2 axis driven by Ca2+/FAK/RhoA signaling.
 
Key finding (Moroishi et al. Genes Dev 2015; Zanconato et al. Cancer Cell 2016):
  - High stiffness → Piezo1 → Ca2+ → FAK/RhoA → LATS1/2 inhibited
    → YAP dephosphorylated → nuclear translocation → M1 gene expression
  - Low stiffness → LATS1/2 active → YAP phosphorylated → cytoplasmic
    retention → 14-3-3 binding → proteasomal degradation
 
IMPORTANT macrophage-specific caveat (Science Advances 2025):
  - TAZ in macrophages localizes to Golgi (LATS1/2-independent)
  - YAP is the mechanotransduction effector in this cell type
  - Model tracks only YAP_nuclear / YAP_cytoplasmic
 
ODE model (3-state: YAP_cyto, YAP_nuclear, YAP_phospho):
----------------------------------------------------------
dYAP_cyto/dt  = k_synth - k_nuc*f_lats_off*YAP_cyto + k_export*YAP_nuclear
dYAP_nuc/dt   = k_nuc*f_lats_off*YAP_cyto - k_export*YAP_nuclear - k_deg_nuc*YAP_nuclear
dYAP_phos/dt  = k_phos*f_lats_on*YAP_cyto - k_dephos*YAP_phos - k_deg_p*YAP_phos
 
Where:
  f_lats_on  = LATS1/2 activity (inhibited by high Ca2+/stiffness)
  f_lats_off = 1 - f_lats_on
 
LATS1/2 activity function (Hill-suppressed by Ca2+):
  f_lats_on = K_lats^n / (K_lats^n + [Ca2+]^n)
 
Rate constants (literature-grounded estimates):
  k_synth   = 0.001 /min  (YAP synthesis, normalized)
  k_nuc     = 0.05  /min  (nuclear import rate when LATS inactive)
  k_export  = 0.02  /min  (nuclear export rate)
  k_phos    = 0.08  /min  (LATS1/2-mediated phosphorylation rate)
  k_dephos  = 0.01  /min  (phosphatase activity)
  k_deg_nuc = 0.001 /min  (nuclear YAP turnover)
  k_deg_p   = 0.02  /min  (phospho-YAP degradation via SCF-β-TRCP)
  K_lats    = 0.25 µM     (Ca2+ half-saturation for LATS suppression)
  n_lats    = 2           (cooperativity)
 
Output: YAP_nuclear/(YAP_nuclear + YAP_cyto) — nuclear fraction [0,1]
  This is the biologically relevant readout (nuclear fraction drives M1 genes).
  High nuclear YAP → M1 bias; low nuclear YAP → M2-permissive state.
"""
 
import numpy as np
from scipy.integrate import solve_ivp
from inputs.surface_params import SurfaceInputs
 
# ---- Rate constants ----
K_SYNTH   = 0.001   # /min
K_NUC     = 0.05    # /min — nuclear import when LATS off
K_EXPORT  = 0.02    # /min — nuclear export
K_PHOS    = 0.08    # /min — LATS-mediated phosphorylation
K_DEPHOS  = 0.01    # /min — dephosphorylation
K_DEG_NUC = 0.001   # /min — nuclear YAP degradation
K_DEG_P   = 0.02    # /min — phospho-YAP degradation
 
# LATS1/2 Ca2+ suppression parameters
K_LATS    = 0.25    # µM — half-saturation (Ca2+ suppresses LATS)
N_LATS    = 2.0     # Hill coefficient
 
# Initial conditions (resting macrophage on soft substrate)
YAP_CYTO_0  = 0.7  # most YAP cytoplasmic at rest
YAP_NUC_0   = 0.1
YAP_PHOS_0  = 0.2
 
 
class YAPHippoModule:
    """
    Simulates YAP nuclear translocation driven by Ca2+-mediated LATS1/2 inhibition.
 
    Inputs:
        surface: SurfaceInputs — provides stiffness context
        ca_timeseries: np.ndarray — [Ca2+]i time-series from Piezo1 module (µM)
    Output:
        yap_nuclear_fraction: time-series of YAP_nuclear / (YAP_nuclear + YAP_cyto)
    """
 
    def __init__(self, surface: SurfaceInputs, ca_timeseries: np.ndarray):
        self.surface = surface
        self.ca = ca_timeseries
 
    def _lats_activity(self, ca: float) -> float:
        """
        LATS1/2 kinase activity: inhibited by high Ca2+ (stiffness signal).
        Hill function: active at low Ca2+, suppressed at high Ca2+.
        """
        return (K_LATS ** N_LATS) / (K_LATS ** N_LATS + ca ** N_LATS)
 
    def _odes(self, t: float, y: list, t_arr: np.ndarray, ca_arr: np.ndarray) -> list:
        yap_c, yap_n, yap_p = y
 
        # Interpolate Ca2+ at current t
        ca = float(np.interp(t, t_arr, ca_arr))
 
        lats_on  = self._lats_activity(ca)   # LATS active when Ca2+ low (soft)
        lats_off = 1.0 - lats_on              # LATS inactive when Ca2+ high (stiff)
 
        # Cytoplasmic YAP
        d_cyto = (K_SYNTH
                  - K_NUC * lats_off * yap_c     # import when LATS inactive
                  - K_PHOS * lats_on * yap_c      # phosphorylation when LATS active
                  + K_EXPORT * yap_n              # export back from nucleus
                  + K_DEPHOS * yap_p)             # dephosphorylation back to cyto
 
        # Nuclear YAP (drives M1 gene program when high)
        d_nuc = (K_NUC * lats_off * yap_c
                 - K_EXPORT * yap_n
                 - K_DEG_NUC * yap_n)
 
        # Phospho-YAP (cytoplasmic, targeted for degradation)
        d_phos = (K_PHOS * lats_on * yap_c
                  - K_DEPHOS * yap_p
                  - K_DEG_P * yap_p)
 
        return [d_cyto, d_nuc, d_phos]
 
    def simulate(self, t_minutes: np.ndarray) -> np.ndarray:
        """
        Returns YAP nuclear fraction time-series.
        Range: [0, 1]   High → M1-promoting; Low → M2-permissive
        """
        y0 = [YAP_CYTO_0, YAP_NUC_0, YAP_PHOS_0]
        t_span = (t_minutes[0], t_minutes[-1])
 
        sol = solve_ivp(
            self._odes,
            t_span,
            y0,
            t_eval=t_minutes,
            args=(t_minutes, self.ca),
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
 
        if not sol.success:
            raise RuntimeError(f"YAP Hippo ODE failed: {sol.message}")
 
        yap_cyto   = sol.y[0]
        yap_nuclear = sol.y[1]
 
        total = np.clip(yap_cyto + yap_nuclear, 1e-10, None)
        yap_nuclear_fraction = yap_nuclear / total
 
        return yap_nuclear_fraction