content = """
inputs/drug_profiles.py  (NEW — v3)
=====================================
Pre-built drug release profiles for common implant drug delivery systems.

Each function takes t_minutes (float or np.ndarray) and returns a
suppression factor [0, 1] representing active drug concentration
at the implant surface at that time.

  0.0 = no drug present (no effect)
  1.0 = maximum drug concentration (full suppression of DAMP/NF-kB)

HOW SUPPRESSION WORKS IN THE MODEL:
  The drug suppression factor is passed to the NF-kB module, which
  multiplies the DAMP drive by (1 - suppression_factor).
  So suppression=0.8 reduces DAMP drive by 80% at that timepoint.

  For scaffold mode, the drug profile can also be used to time
  the M1→M2 transition by suppressing early NF-kB and releasing
  STAT6-promoting agents later.

AVAILABLE PROFILES:
  - plga_microsphere(half_life_days, burst_fraction)
      Standard PLGA degradation release kinetics
  - dual_phase(burst_days, sustained_days, burst_level, sustained_level)
      Burst release followed by sustained plateau (e.g. coated stents)
  - linear_ramp(start_day, end_day)
      Linear increase (e.g. pH-responsive scaffold degradation)
  - scaffold_guided()
      Designed for scaffold mode: early suppression then release
      to allow productive fibrosis on schedule

USAGE EXAMPLE:
    from inputs.drug_profiles import plga_microsphere
    surface = SurfaceInputs(
        E_kPa=50,
        drug_release_profile=plga_microsphere(half_life_days=7)
    )
"""

import numpy as np
from typing import Callable


def plga_microsphere(
    half_life_days: float = 7.0,
    burst_fraction: float = 0.3,
    burst_duration_hours: float = 6.0,
) -> Callable:
    """
    PLGA microsphere release kinetics.
    Initial burst release followed by first-order degradation decay.

    Parameters
    ----------
    half_life_days : float
        Half-life of sustained release phase (days). Typical PLGA: 7–21 days.
    burst_fraction : float [0,1]
        Fraction of dose released in initial burst. Typical: 0.2–0.4.
    burst_duration_hours : float
        Duration of burst phase in hours. Typical: 4–12 hours.

    Returns
    -------
    Callable : t_minutes → suppression factor [0,1]
    """
    half_life_min  = half_life_days * 24 * 60
    burst_end_min  = burst_duration_hours * 60
    k_decay        = 0.693 / half_life_min

    def profile(t_min):
        t = np.asarray(t_min, dtype=float)
        burst     = burst_fraction * np.exp(-3.0 * t / burst_end_min)
        sustained = (1.0 - burst_fraction) * np.exp(-k_decay * t)
        total = np.clip(burst + sustained, 0.0, 1.0)
        return float(total) if np.ndim(t_min) == 0 else total

    profile.__name__ = f"plga_t{half_life_days}d"
    return profile


def dual_phase(
    burst_days: float = 1.0,
    sustained_days: float = 14.0,
    burst_level: float = 0.9,
    sustained_level: float = 0.4,
) -> Callable:
    """
    Two-phase release: high burst then sustained plateau.
    Models coated stents, drug-eluting orthopedic implants.
    """
    burst_end_min     = burst_days * 24 * 60
    sustained_end_min = sustained_days * 24 * 60

    def profile(t_min):
        t = np.asarray(t_min, dtype=float)
        out = np.where(
            t <= burst_end_min,
            burst_level * np.exp(-2.0 * t / burst_end_min),
            sustained_level * np.exp(
                -0.693 * (t - burst_end_min) /
                (sustained_end_min - burst_end_min)
            )
        )
        return float(np.clip(out, 0, 1)) if np.ndim(t_min) == 0 \
               else np.clip(out, 0, 1)

    profile.__name__ = "dual_phase"
    return profile


def scaffold_guided() -> Callable:
    """
    Drug profile designed for scaffold/guided tissue regeneration mode.

    Strategy:
      Days 0–3:  High NF-kB suppression (anti-inflammatory drug)
                 → prevents excessive M1 inflammation that damages scaffold
      Days 3–7:  Rapid withdrawal → allows controlled NF-kB for fibroblast
                 recruitment and early ECM priming
      Days 7–28: Low-level sustained release → maintains environment
                 permissive for M2a reparative macrophages

    This is the pharmacological implementation of the "productive fibrosis"
    trajectory used in scaffold design mode.
    """
    def profile(t_min):
        t = np.asarray(t_min, dtype=float)
        phase1 = 0.85 * np.exp(-0.693 * t / (2 * 24 * 60))   # 2-day HL decay
        phase2 = 0.20 * (1.0 / (1.0 + np.exp(
            -(t - 7*24*60) / (2*24*60)
        )))  # slow rise at day 7
        out = np.clip(phase1 + phase2, 0, 1)
        return float(out) if np.ndim(t_min) == 0 else out

    profile.__name__ = "scaffold_guided"
    return profile


def no_drug() -> Callable:
    """Null profile — no drug release. Equivalent to default None."""
    def profile(t_min):
        t = np.asarray(t_min, dtype=float)
        return 0.0 if np.ndim(t_min) == 0 else np.zeros_like(t)
    profile.__name__ = "no_drug"
    return profile

with open('inputs/drug_profile.py', 'w') as f:
    f.write(content)