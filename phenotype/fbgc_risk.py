"""
phenotype/fbgc_risk.py
========================
FBGC Risk Module: Computes probability of foreign body giant cell formation.
 
Biological basis:
-----------------
FBGC formation requires a joint set of conditions (logical AND thresholds):
  1. IL-4/IL-13 signal sustained above threshold (STAT6 driven)
  2. M-CSF-biased differentiation (surface-dependent, not GM-CSF suppressed)
  3. Adequate macrophage adhesion to surface (protein adsorption dependent)
  4. NF-κB below high-inflammation suppression threshold
     (high NF-κB → GM-CSF-like program → fusion incompetent)
  5. Sufficient time elapsed (>7 days — fusion competency not instant)
 
Source: Stewart et al. Front Immunol 2024; Anderson group reviews;
        Ballabiyev podosome zipper mechanism;
        Grounded in: "whether the phagocytic→fusogenic switch is graded
        or a physiologic switch" (Bhatt et al. PMC 2010 — open question)
 
This model treats it as GRADED (not a hard switch) — this is a prediction.
The probability rises continuously as conditions are met.
 
FBGC Risk Score (0–1):
  P_fbgc(t) = stat6_sustained * mcsf_bias * adhesion_score * (1 - nfkb_excess)
 
  Where stat6_sustained = rolling mean of STAT6 over 72h window
  (fusion requires sustained IL-4/IL-13, not a transient spike)
 
Final risk: maximum P_fbgc reached over simulation, clamped to [0,1].
Risk categories:
  Low:      0.0–0.25
  Moderate: 0.25–0.60
  High:     0.60–0.80
  Very High: 0.80–1.0
"""
 
import numpy as np
from inputs.surface_params import SurfaceInputs
 
NFKB_EXCESS_THRESHOLD = 0.6  # above this NF-κB level, fusion competency is suppressed
STAT6_WINDOW_HOURS    = 72   # rolling window for STAT6 sustained signal (hours)
MIN_ONSET_DAYS        = 7    # minimum days before FBGC can begin forming
 
 
class FBGCRiskModule:
    """
    Computes FBGC formation probability time-series and final risk score.
    """
 
    def __init__(
        self,
        surface: SurfaceInputs,
        il4_signal: np.ndarray,
        nfkb_active: np.ndarray,
        t_minutes: np.ndarray,
    ):
        self.surface    = surface
        self.il4        = il4_signal
        self.nfkb       = nfkb_active
        self.t          = t_minutes
        self.dt_min     = t_minutes[1] - t_minutes[0] if len(t_minutes) > 1 else 1.0
 
    def _rolling_mean(self, arr: np.ndarray, window_hours: float) -> np.ndarray:
        """Compute rolling mean over a window of hours."""
        window_steps = max(1, int(window_hours * 60 / self.dt_min))
        out = np.convolve(arr, np.ones(window_steps) / window_steps, mode='same')
        # Fix edge effects at start
        for i in range(min(window_steps, len(out))):
            out[i] = np.mean(arr[:i+1])
        return out
 
    def compute_risk(self) -> dict:
        """
        Returns dict with:
          - 'risk_timeseries': P_fbgc(t) array
          - 'peak_risk': maximum risk reached
          - 'risk_category': str label
          - 'onset_day': first day risk exceeds 0.25 (-1 if never)
        """
        # Sustained STAT6 (72h rolling mean — requires persistent IL-4/IL-13)
        stat6_sustained = self._rolling_mean(self.il4, STAT6_WINDOW_HOURS)
 
        # NF-κB suppression of fusion competency
        nfkb_excess = np.clip((self.nfkb - NFKB_EXCESS_THRESHOLD) / 0.4, 0.0, 1.0)
        fusion_permissive = 1.0 - nfkb_excess
 
        # Adhesion score from surface (protein adsorption enables macrophage-surface adhesion)
        adhesion = self.surface.protein_adsorption_score
 
        # mCSF bias from surface
        mcsf = self.surface.mcsf_bias
 
        # Time gate: no FBGC before day 7
        onset_gate = np.where(
            self.t >= MIN_ONSET_DAYS * 24 * 60,
            1.0,
            self.t / (MIN_ONSET_DAYS * 24 * 60)  # ramp-up before day 7
        )
 
        # Joint probability
        risk_ts = stat6_sustained * fusion_permissive * adhesion * mcsf * onset_gate
        risk_ts = np.clip(risk_ts, 0.0, 1.0)
 
        peak_risk = float(np.max(risk_ts))
 
        # Risk category
        if peak_risk < 0.25:
            category = "Low"
        elif peak_risk < 0.60:
            category = "Moderate"
        elif peak_risk < 0.80:
            category = "High"
        else:
            category = "Very High"
 
        # Onset day
        onset_idx = np.where(risk_ts > 0.25)[0]
        onset_day = float(self.t[onset_idx[0]]) / (24 * 60) if len(onset_idx) > 0 else -1.0
 
        return {
            "risk_timeseries": risk_ts,
            "peak_risk": peak_risk,
            "risk_category": category,
            "onset_day": onset_day,
        }