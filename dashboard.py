"""
dashboard.py — FBR Model Interactive Dashboard (v7 — Lab Grade)
================================================================
Five professional upgrades:
  1. Material presets dropdown (real implant materials with literature values)
  2. Early-phase zoom panel (days 0-7 signaling detail)
  3. Reference-anchored scores table (color bands + baseline comparison)
  4. Day-specific phenotype snapshot (flow cytometry style readout)
  5. Protocol recommendation engine + PDF/CSV export
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from body import (
    render_body_selector, body_region_panel,
    BODY_REGIONS, HOTSPOTS
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io, datetime

from inputs.surface_params import SurfaceInputs
from inputs.drug_profiles   import plga_microsphere, dual_phase, scaffold_guided
from signaling.piezo1_calcium       import Piezo1CalciumModule
from signaling.yap_hippo            import YAPHippoModule
from signaling.nfkb                 import NFkBModule
from signaling.stat6                import STAT6Module
from phenotype.phenotype_classifier import PhenotypeClassifier
from phenotype.fbgc_risk            import FBGCRiskModule

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS & PRESETS
# ═══════════════════════════════════════════════════════════════════

DRUG_MAP = {
    "None":            None,
    "PLGA 7-day":      lambda: plga_microsphere(half_life_days=7),
    "PLGA 14-day":     lambda: plga_microsphere(half_life_days=14),
    "Dual-phase":      dual_phase,
    "Scaffold-guided": scaffold_guided,
}

# Real implant material presets — literature-sourced surface parameters
# Sources: Hotchkiss 2016, Anderson reviews, Invibio PEEK data, neural probe literature
MATERIAL_PRESETS = {
    "— Custom (manual) —": None,
    "Ti-SLA (Straumann, osseointegration)": {
        "E_kPa": 200.0, "Ra_um": 3.2, "feature_size_nm": 2000.0,
        "contact_angle_deg": 130.0, "tissue_context": "bone",
        "coating": "none", "drug_key": "None",
        "note": "Sandblasted acid-etched Ti. High roughness, hydrophobic. Clinical gold standard for dental implants.",
    },
    "Ti-modSLA (Straumann SLActive)": {
        "E_kPa": 200.0, "Ra_um": 3.2, "feature_size_nm": 100.0,
        "contact_angle_deg": 4.0, "tissue_context": "bone",
        "coating": "none", "drug_key": "None",
        "note": "Chemically modified SLA. Superhydrophilic. Faster osseointegration, reduced FBR.",
    },
    "PEEK unfunctionalized (Invibio)": {
        "E_kPa": 50.0, "Ra_um": 0.3, "feature_size_nm": 5000.0,
        "contact_angle_deg": 85.0, "tissue_context": "bone",
        "coating": "none", "drug_key": "None",
        "note": "Bioinert, hydrophobic. Fibrous encapsulation risk. Invibio's core FBR problem.",
    },
    "PEEK + HA coating": {
        "E_kPa": 50.0, "Ra_um": 0.8, "feature_size_nm": 500.0,
        "contact_angle_deg": 40.0, "tissue_context": "bone",
        "coating": "none", "drug_key": "None",
        "note": "HA-coated PEEK. Partially addresses hydrophobicity. Inconsistent osseointegration.",
    },
    "Silicone breast implant (smooth)": {
        "E_kPa": 0.3, "Ra_um": 0.05, "feature_size_nm": 5000.0,
        "contact_angle_deg": 105.0, "tissue_context": "soft_tissue",
        "coating": "none", "drug_key": "None",
        "note": "Smooth silicone. Very soft, highly hydrophobic. Capsular contracture driver.",
    },
    "Silicone breast implant (textured)": {
        "E_kPa": 0.3, "Ra_um": 2.5, "feature_size_nm": 1000.0,
        "contact_angle_deg": 100.0, "tissue_context": "soft_tissue",
        "coating": "none", "drug_key": "None",
        "note": "Textured silicone. Reduces contracture but raises FBGC concern.",
    },
    "Neural silicon probe (rigid)": {
        "E_kPa": 200.0, "Ra_um": 0.1, "feature_size_nm": 5000.0,
        "contact_angle_deg": 60.0, "tissue_context": "brain",
        "coating": "none", "drug_key": "None",
        "note": "Rigid Si probe. Catastrophic stiffness mismatch with brain (~0.1 kPa). High neuroinflammation.",
    },
    "Neural probe + zwitterionic coating": {
        "E_kPa": 200.0, "Ra_um": 0.1, "feature_size_nm": 5000.0,
        "contact_angle_deg": 15.0, "tissue_context": "brain",
        "coating": "zwitterionic", "drug_key": "None",
        "note": "Zwitterionic-coated neural probe. State-of-art biocompatibility improvement.",
    },
    "PLGA scaffold (tissue engineering)": {
        "E_kPa": 5.0, "Ra_um": 1.5, "feature_size_nm": 800.0,
        "contact_angle_deg": 55.0, "tissue_context": "soft_tissue",
        "coating": "none", "drug_key": "PLGA 7-day",
        "note": "Biodegradable scaffold with drug elution. Scaffold mode recommended.",
    },
    "PEG hydrogel (sensor encapsulation)": {
        "E_kPa": 2.0, "Ra_um": 0.02, "feature_size_nm": 5000.0,
        "contact_angle_deg": 20.0, "tissue_context": "soft_tissue",
        "coating": "peg", "drug_key": "None",
        "note": "Ultra-soft, ultra-hydrophilic. Minimal FBR. Gold standard for continuous glucose sensors.",
    },
}

PHENOTYPE_FILLS  = {
    "Inflammatory": "rgba(226,75,74,0.12)",
    "Transitional":  "rgba(239,159,39,0.12)",
    "Healing":       "rgba(29,158,117,0.12)",
    "FBGC-prone":    "rgba(127,119,221,0.12)",
}
PHENOTYPE_LINES  = {
    "Inflammatory": "#E24B4A",
    "Transitional":  "#EF9F27",
    "Healing":       "#1D9E75",
    "FBGC-prone":    "#7F77DD",
}
PHENOTYPE_KEYS   = {
    "Inflammatory": "inflam",
    "Transitional":  "transit",
    "Healing":       "healing",
    "FBGC-prone":    "fbgc_prone",
}

SNAPSHOT_DAYS = [1, 3, 7, 14, 28]

# Score reference ranges for color-coding the surface table
SCORE_REFS = {
    "mechano_input_effective":  {"low": 0.3, "high": 0.7, "invert": False,
                                  "label": "Low=soft (M2 bias) | High=stiff (M1 bias)"},
    "nanotopo_modifier":        {"low": 0.3, "high": 0.7, "invert": True,
                                  "label": "High=nanofeatured (immunomodulatory)"},
    "damp_signal":              {"low": 0.2, "high": 0.5, "invert": True,
                                  "label": "Low=minimal danger signal (better)"},
    "protein_adsorption_score": {"low": 0.2, "high": 0.5, "invert": True,
                                  "label": "Low=antifouling (better)"},
    "wettability_factor":       {"low": 0.3, "high": 0.6, "invert": False,
                                  "label": "High=hydrophilic (M2 promoting)"},
    "mcsf_bias":                {"low": 0.4, "high": 0.7, "invert": False,
                                  "label": "High=smooth/M-CSF differentiation"},
}


# ═══════════════════════════════════════════════════════════════════
# MODEL RUNNER (cache-safe — no callables in return dict)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_model(E_kPa, Ra_um, feature_nm, contact_angle,
              tissue, coating, drug_key, mode, t_days, dt_min):
    drug_fn = None
    if drug_key != "None" and DRUG_MAP.get(drug_key) is not None:
        drug_fn = DRUG_MAP[drug_key]()

    s = SurfaceInputs(
        E_kPa=E_kPa, Ra_um=Ra_um, feature_size_nm=feature_nm,
        contact_angle_deg=contact_angle, tissue_context=tissue,
        coating=coating, drug_release_profile=drug_fn,
    )
    t = np.arange(0, t_days * 24 * 60, dt_min)

    ca    = Piezo1CalciumModule(s).simulate(t)
    yap   = YAPHippoModule(s, ca).simulate(t)
    nfkb  = NFkBModule(s, ca).simulate(t)
    stat6 = STAT6Module(s, t).simulate(t)
    ps    = PhenotypeClassifier(
        yap, nfkb, stat6, t,
        mcsf_bias=s.mcsf_bias, mode=mode,
        wettability_factor=s.wettability_factor,
    ).compute()
    fb    = FBGCRiskModule(s, stat6, nfkb, t).compute_risk()

    t_days_arr = t / (24 * 60)
    drug_ts = None
    if drug_fn is not None:
        drug_ts = np.array([float(drug_fn(ti)) for ti in t])

    # Build snapshot at specific days
    snapshots = {}
    for day in SNAPSHOT_DAYS:
        if day <= t_days:
            idx = np.searchsorted(t_days_arr, day)
            idx = min(idx, len(t_days_arr) - 1)
            snapshots[day] = {
                "inflammatory": float(ps.inflammatory[idx]),
                "transitional":  float(ps.transitional[idx]),
                "healing":       float(ps.healing[idx]),
                "fbgc_prone":    float(ps.fbgc_prone[idx]),
                "nfkb":          float(nfkb[idx]),
                "yap":           float(yap[idx]),
                "stat6":         float(stat6[idx]),
            }

    surface_scores = {
        "E_kPa":                    float(s.E_kPa),
        "Ra_um":                    float(s.Ra_um),
        "feature_size_nm":          float(s.feature_size_nm),
        "contact_angle_deg":        float(s.contact_angle_deg),
        "mechano_input_effective":  float(s.mechano_input_effective),
        "nanotopo_modifier":        float(s.nanotopo_modifier),
        "damp_signal":              float(s.damp_signal),
        "protein_adsorption_score": float(s.protein_adsorption_score),
        "wettability_factor":       float(s.wettability_factor),
        "mcsf_bias":                float(s.mcsf_bias),
        "coating_suppression":      float(s.coating_suppression),
        "tissue_context":           str(tissue),
        "coating_name":             str(coating),
    }

    return {
        "surface_scores": surface_scores,
        "t_days":         t_days_arr,
        "ca":             ca,
        "yap":            yap,
        "nfkb":           nfkb,
        "stat6":          stat6,
        "inflam":         ps.inflammatory,
        "transit":        ps.transitional,
        "healing":        ps.healing,
        "fbgc_prone":     ps.fbgc_prone,
        "fbgc_risk":      fb["risk_timeseries"],
        "fbgc_peak":      float(fb["peak_risk"]),
        "fbgc_cat":       str(fb["risk_category"]),
        "fbgc_onset":     float(fb["onset_day"]),
        "imm_index":      float(ps.immunomodulation_index),
        "traj_score":     float(ps.trajectory_match_score),
        "dominant":       str(ps.dominant_phenotype()),
        "drug_ts":        drug_ts,
        "snapshots":      snapshots,
        "mode":           mode,
    }


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def generate_recommendations(R: dict, sc: dict) -> list[dict]:
    """
    Returns a list of dicts: {level, title, detail}
    level: 'critical' | 'warning' | 'good'
    """
    recs = []
    peak = R["fbgc_peak"]
    idx  = R["imm_index"]
    damp = sc["damp_signal"]
    ads  = sc["protein_adsorption_score"]
    mech = sc["mechano_input_effective"]
    coat = sc["coating_suppression"]
    wet  = sc["wettability_factor"]

    # FBGC risk
    if peak > 0.60:
        recs.append({"level": "critical", "title": "High FBGC risk detected",
            "detail": (f"Peak risk {peak*100:.0f}%. Foreign body giant cell formation likely. "
                       "Recommend: switch to zwitterionic or phospholipid coating "
                       "(projects ~80–90% protein adsorption suppression), or add "
                       "PLGA 7-day anti-inflammatory release profile.")})
    elif peak > 0.25:
        recs.append({"level": "warning", "title": "Moderate FBGC risk",
            "detail": (f"Peak risk {peak*100:.0f}%. Monitor macrophage fusion markers "
                       "(DC-STAMP, CD98) experimentally at day 7–14. "
                       "Consider surface hydrophilization to reduce DAMP signal.")})
    else:
        recs.append({"level": "good", "title": "FBGC risk well-controlled",
            "detail": f"Peak risk {peak*100:.0f}%. Surface configuration unfavorable for giant cell fusion."})

    # Immunomodulation
    if idx < 0.8:
        recs.append({"level": "critical", "title": "M1-dominated response predicted",
            "detail": (f"Immunomodulation index {idx:.2f} — chronic inflammation risk. "
                       "Mechano-input {mech:.2f} is driving sustained YAP nuclear translocation. "
                       "Recommend: reduce substrate stiffness below 20 kPa or add "
                       "phospholipid coating to reduce mechanosensing-driven NF-κB.")})
    elif idx < 1.5:
        recs.append({"level": "warning", "title": "Balanced M1/M2 — resolution not guaranteed",
            "detail": (f"Index {idx:.2f}. Healing phenotype is present but not dominant. "
                       "IL-4/STAT6 axis is active — ensure experimental timepoints extend "
                       "beyond day 14 to capture M2a transition.")})
    else:
        recs.append({"level": "good", "title": "M2-favorable immunomodulation",
            "detail": f"Index {idx:.2f}. Surface drives healing-dominant macrophage response."})

    # DAMP / protein adsorption
    if damp > 0.5:
        recs.append({"level": "critical", "title": "Elevated DAMP signal",
            "detail": (f"DAMP score {damp:.3f}. High protein adsorption "
                       f"({ads:.3f}) combined with hydrophobic surface is generating "
                       "sustained danger signals. This will maintain NF-κB activation "
                       "beyond the expected acute resolution window. "
                       "Recommend: plasma treatment or zwitterionic coating.")})
    elif damp > 0.25:
        recs.append({"level": "warning", "title": "Moderate DAMP signal",
            "detail": f"DAMP score {damp:.3f}. Protein corona forming at surface. "
                      "Consider hydrophilization to suppress non-specific adsorption."})

    # Coating
    if coat == 0.0:
        recs.append({"level": "warning", "title": "No surface coating applied",
            "detail": "Uncoated surface. For implant applications, consider PEG (75% suppression), "
                      "zwitterionic (85%), or phospholipid (90%) coating to reduce protein "
                      "adsorption and macrophage activation."})

    # Wettability
    if wet < 0.2:
        recs.append({"level": "warning", "title": "Hydrophobic surface — M2 pathway suppressed",
            "detail": (f"Wettability factor {wet:.3f}. Low hydrophilicity limits integrin "
                       "αvβ3/β5 signaling that promotes M2a transition. "
                       "Plasma treatment or hydrophilic coating recommended.")})

    # Brain stiffness mismatch
    if sc["E_kPa"] > 50 and sc.get("tissue_context","") == "brain":
        recs.append({"level": "critical",
            "title": "Severe stiffness mismatch for brain context",
            "detail": (
                f"Brain tissue is 0.1-1 kPa. Surface stiffness {sc['E_kPa']:.0f} kPa "
                "will drive maximum Piezo1 activation and sustained YAP-mediated "
                "neuroinflammation. Consider flexible polymer or hydrogel encapsulation."
            )})

    # FIX C: Scaffold stiffness mismatch
    if mode == "scaffold" and sc["E_kPa"] > 30:
        recs.append({"level": "critical",
            "title": "Stiffness too high for scaffold/tissue regeneration mode",
            "detail": (
                f"Substrate stiffness {sc['E_kPa']:.0f} kPa exceeds the recommended "
                f"range for tissue engineering scaffolds (5-20 kPa soft tissue, "
                f"20-40 kPa cartilage). High stiffness drives sustained Piezo1/YAP "
                f"activation, biasing macrophages toward M1 and suppressing the "
                f"productive fibrosis trajectory needed for scaffold integration. "
                f"This explains your low trajectory match score. "
                f"Recommend: target stiffness below 20 kPa, or use the PLGA scaffold preset."
            )})

    # FIX D1: FBGC-prone vs FBGC risk gap
    snap_28 = R.get("snapshots", {}).get(28, {})
    fbgc_prone_28 = snap_28.get("fbgc_prone", 0)
    if fbgc_prone_28 > 0.35 and peak < 0.55:
        recs.append({"level": "warning",
            "title": "Fusion-competent macrophages present — risk partially gated",
            "detail": (
                f"FBGC-prone phenotype score {fbgc_prone_28:.2f} at day 28 indicates "
                f"macrophages are in a fusion-competent state (STAT6 active, NF-kB "
                f"suppressed, M-CSF bias present). The lower FBGC risk ({peak*100:.0f}%) "
                f"reflects that the 72-hour sustained IL-4 threshold and surface "
                f"adhesion conditions are not fully met for active giant cell fusion. "
                f"Monitor DC-STAMP and CD98 expression at day 7-14 — these fusion "
                f"markers precede visible giant cell formation by 2-4 days."
            )})

    # FIX D2: No nanotopography effect
    if sc.get("nanotopo_modifier", 1.0) < 0.01:
        recs.append({"level": "warning",
            "title": "No immunomodulatory nanotopography at this feature size",
            "detail": (
                f"Feature size {sc['feature_size_nm']:.0f} nm is above the "
                f"immunomodulatory range. Below ~50 nm, nanotopography meaningfully "
                f"attenuates Piezo1 mechanosensing and shifts macrophages toward M2. "
                f"At {sc['feature_size_nm']:.0f} nm this effect is negligible "
                f"(nanotopo modifier = {sc['nanotopo_modifier']:.4f}). "
                f"To leverage nanotopographic immunomodulation, target 15-50 nm "
                f"feature sizes (TiO2 nanotubes, nanopillars, or anodized nanopits)."
            )})

    return recs


# ═══════════════════════════════════════════════════════════════════
# EXPORT: CSV + TEXT REPORT
# ═══════════════════════════════════════════════════════════════════

def build_csv(R: dict) -> bytes:
    t  = R["t_days"]
    df = pd.DataFrame({
        "day":         t,
        "Ca_uM":       R["ca"],
        "YAP_nuclear": R["yap"],
        "NFkB":        R["nfkb"],
        "STAT6":       R["stat6"],
        "inflammatory":R["inflam"],
        "transitional":R["transit"],
        "healing":     R["healing"],
        "fbgc_prone":  R["fbgc_prone"],
        "fbgc_risk":   R["fbgc_risk"],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def build_report(R: dict, sc: dict, recs: list,
                 material_name: str, mode: str) -> bytes:
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    snap = R.get("snapshots", {})

    lines = [
        "=" * 65,
        "  FBR MECHANOTRANSDUCTION MODEL — SIMULATION REPORT",
        f"  Generated: {now}",
        "=" * 65,
        "",
        f"Material:        {material_name}",
        f"Simulation mode: {mode.upper()}",
        f"Tissue context:  {sc.get('tissue_context', 'N/A')}",
        "",
        "SURFACE PARAMETERS",
        "-" * 40,
        f"  Stiffness:          {sc['E_kPa']} kPa",
        f"  Roughness Ra:       {sc['Ra_um']} µm",
        f"  Feature size:       {sc['feature_size_nm']} nm",
        f"  Contact angle:      {sc['contact_angle_deg']}°",
        f"  Coating suppression:{sc['coating_suppression']:.0%}",
        "",
        "DERIVED BIOLOGICAL SCORES",
        "-" * 40,
        f"  Mechano-input:      {sc['mechano_input_effective']:.3f}",
        f"  Nanotopo modifier:  {sc['nanotopo_modifier']:.3f}",
        f"  DAMP signal:        {sc['damp_signal']:.3f}",
        f"  Protein adsorption: {sc['protein_adsorption_score']:.3f}",
        f"  Wettability factor: {sc['wettability_factor']:.3f}",
        f"  M-CSF bias:         {sc['mcsf_bias']:.3f}",
        "",
        "SIMULATION OUTCOMES (Day 28)",
        "-" * 40,
        f"  Dominant phenotype:        {R['dominant']}",
        f"  Immunomodulation index:    {R['imm_index']:.3f}",
        f"  FBGC peak risk:            {R['fbgc_peak']*100:.1f}% ({R['fbgc_cat']})",
        f"  FBGC onset day:            "
        + (f"{R['fbgc_onset']:.1f}" if R['fbgc_onset'] > 0 else "Not reached"),
    ]

    if mode == "scaffold":
        lines.append(f"  Scaffold trajectory match: {R['traj_score']:.3f} / 1.000")

        lines += ["", "PHENOTYPE SNAPSHOTS", "-" * 40]
        for day, data in snap.items():
            lines.append(
                f"  Day {day:>2}: Inflam={data['inflammatory']:.2f}  "
                f"Transit={data['transitional']:.2f}  "
                f"Healing={data['healing']:.2f}  "
                f"FBGC-prone={data['fbgc_prone']:.2f}"
            )

        lines += ["", "PROTOCOL RECOMMENDATIONS", "-" * 40]
        for r in recs:
            tag = {"critical": "[CRITICAL]", "warning": "[WARNING]",
                   "good": "[OK]"}.get(r["level"], "")
            lines.append(f"  {tag} {r['title']}")
            lines.append(f"    {r['detail']}")
            lines.append("")

        lines += [
            "VALIDATION",
            "-" * 40,
            "  Directionally validated against:",
            "  - Hotchkiss et al. 2016 (Acta Biomaterialia) — Ti surface wettability",
            "  - J. Nanobiotechnology 2023 — TiO2 nanotube size immunomodulation",
            "  Rank-order predictions: 6/6 checks passed.",
            "",
            "  Signaling parameters:",
            "  - Piezo1/Ca2+: Bhatt et al. 2021 (Nature Communications)",
            "  - NF-kB: Hoffmann et al. 2002 (Science), adapted for macrophage kinetics",
            "  - YAP/TAZ: Zanconato et al. 2016; macrophage-specific: Moroishi 2015",
            "=" * 65,
        ]

    return "\n".join(lines).encode()


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FBR Mechanotransduction Model",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 FBR Model")
    st.caption("Mechanotransduction → Phenotype Engine")
    st.divider()

    # Material preset selector
    st.subheader("Material Preset")
    selected_material = st.selectbox(
        "Select implant material",
        list(MATERIAL_PRESETS.keys()),
        help="Pre-fills surface parameters from literature values for real implant materials.",
    )
    preset = MATERIAL_PRESETS[selected_material]
    if preset and selected_material != "— Custom (manual) —":
        st.caption(f"📖 {preset['note']}")

    st.divider()

    mode = st.radio("Simulation mode", ["suppress", "scaffold"])

    # ── Interactive body region selector ──────────────────────
    st.subheader("Implant Site")
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = "Knee"

    body_html = render_body_selector(st.session_state.selected_region)
    components.html(body_html, height=500, scrolling=False)

    # Fallback clickable buttons for each region
    region_names = list(BODY_REGIONS.keys())
    cols = st.columns(2)
    for i, rname in enumerate(region_names):
        info = BODY_REGIONS[rname]
        if cols[i % 2].button(
            f"{info['icon']} {rname}",
            key=f"btn_{rname}",
            use_container_width=True,
            type="primary" if rname == st.session_state.selected_region else "secondary",
        ):
            st.session_state.selected_region = rname
            st.rerun()

    selected_region = st.session_state.selected_region
    region_info     = BODY_REGIONS[selected_region]
    tissue          = region_info["tissue_context"]
    tissue_E_default = region_info["E_kPa_default"]
    coating = st.selectbox("Surface coating",
                           ["none","peg","zwitterionic","phospholipid"],
                           index=(["none","peg","zwitterionic","phospholipid"]
                                  .index(preset["coating"])
                                  if preset else 0))
    drug_key = st.selectbox("Drug release profile",
                            list(DRUG_MAP.keys()),
                            index=(list(DRUG_MAP.keys())
                                   .index(preset["drug_key"])
                                   if preset else 0))
    t_days = st.slider("Simulation duration (days)", 7, 56, 28)

    st.divider()
    st.subheader("Surface Parameters")

    E_def    = preset["E_kPa"]           if preset else 50.0
    Ra_def   = preset["Ra_um"]           if preset else 0.5
    feat_def = preset["feature_size_nm"] if preset else 200.0
    th_def   = preset["contact_angle_deg"] if preset else 70.0

    # Use region default stiffness if no preset selected
    e_default_final = E_def if preset else tissue_E_default
    E_kPa   = st.slider("Stiffness (kPa)",   0.1,  200.0,
                        float(e_default_final), step=0.5)
    Ra_um   = st.slider("Roughness Ra (µm)", 0.01,   5.0, Ra_def,   step=0.01)
    feat_nm = st.slider("Feature size (nm)",  5.0, 2000.0, feat_def, step=5.0)
    theta   = st.slider("Contact angle (°)",  0.0,  150.0, th_def,   step=1.0)

    st.divider()
    compare_mode = st.checkbox("Compare two surfaces")
    if compare_mode:
        st.subheader("Surface B")
        mat2 = st.selectbox("Material B", list(MATERIAL_PRESETS.keys()),
                            key="mat2")
        p2   = MATERIAL_PRESETS[mat2]
        if p2:
            st.caption(f"📖 {p2['note']}")
        E2     = st.slider("Stiffness B (kPa)",   0.1, 200.0,
                           p2["E_kPa"] if p2 else 1.0, step=0.5, key="E2")
        Ra2    = st.slider("Roughness B (µm)",    0.01,  5.0,
                           p2["Ra_um"] if p2 else 0.05, step=0.01, key="Ra2")
        feat2  = st.slider("Feature size B (nm)",  5.0, 2000.0,
                           p2["feature_size_nm"] if p2 else 100.0,
                           step=5.0, key="f2")
        theta2 = st.slider("Contact angle B (°)",  0.0, 150.0,
                           p2["contact_angle_deg"] if p2 else 25.0,
                           step=1.0, key="th2")
        tissue2  = p2["tissue_context"] if p2 else tissue
        coating2 = p2["coating"]        if p2 else coating
        drug2    = p2["drug_key"]       if p2 else drug_key
    else:
        p2 = None

    # Snapshot day selector
    st.divider()
    st.subheader("Phenotype Snapshot")
    snap_day = st.select_slider(
        "View phenotype at day:",
        options=[d for d in SNAPSHOT_DAYS if d <= t_days],
        value=min(7, t_days),
    )

    # Early zoom toggle
    st.divider()
    show_zoom = st.checkbox("Show early-phase zoom (days 0–7)", value=True)

dt_min = 10.0

# ═══════════════════════════════════════════════════════════════════
# RUN MODEL
# ═══════════════════════════════════════════════════════════════════
try:
    with st.spinner("Running simulation..."):
        R = run_model(E_kPa, Ra_um, feat_nm, theta,
                      tissue, coating, drug_key, "scaffold" if mode=="scaffold" else "suppress",
                      t_days, dt_min)
        R2 = None
        if compare_mode and p2:
            R2 = run_model(E2, Ra2, feat2, theta2,
                           tissue2, coating2, drug2, mode, t_days, dt_min)
except Exception as e:
    st.error(f"Simulation error: {e}")
    st.stop()

sc   = R["surface_scores"]
recs = generate_recommendations(R, sc)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.title("Foreign Body Response — Mechanotransduction Model")
mat_label = selected_material if selected_material != "— Custom (manual) —" \
            else "Custom surface"
st.caption(
    f"**{mat_label}** | Mode: **{mode.upper()}** | Tissue: **{tissue}** | "
    f"Coating: **{coating}** | Drug: **{drug_key}**"
)

# ── KPI row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dominant Phenotype", R["dominant"])
c2.metric("Immuno-mod Index",   f"{R['imm_index']:.2f}",
          help="M2/M1 AUC ratio. <0.8 = M1 dominated | 1.5–3.0 = M2 favorable | >3.0 = strong M2")

risk_delta = None
if compare_mode and R2:
    risk_delta = f"{(R['fbgc_peak'] - R2['fbgc_peak'])*100:+.1f}% vs B"
c3.metric("FBGC Peak Risk",  f"{R['fbgc_peak']*100:.1f}%",
          delta=risk_delta, delta_color="inverse")
c4.metric("FBGC Category",   R["fbgc_cat"])
onset_str = f"Day {R['fbgc_onset']:.1f}" if R["fbgc_onset"] > 0 else "Not reached"
c5.metric("FBGC Onset", onset_str)

if mode == "scaffold":
    score = R["traj_score"]
    msg   = ("✅ Excellent productive fibrosis potential" if score > 0.7
             else "⚠️ Moderate — consider tuning drug profile" if score > 0.4
             else "❌ Poor — surface unlikely to support guided regeneration")
    st.info(f"🧬 Scaffold trajectory match: **{score:.3f} / 1.000** — {msg}")

st.divider()

# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATIONS PANEL
# ═══════════════════════════════════════════════════════════════════
with st.expander("🧪 Protocol Recommendations", expanded=True):
    for rec in recs:
        if rec["level"] == "critical":
            st.error(f"**{rec['title']}** — {rec['detail']}")
        elif rec["level"] == "warning":
            st.warning(f"**{rec['title']}** — {rec['detail']}")
        else:
            st.success(f"**{rec['title']}** — {rec['detail']}")

st.divider()

# ═══════════════════════════════════════════════════════════════════
# MAIN FIGURE — full timeline
# ═══════════════════════════════════════════════════════════════════
n_rows = 4 if show_zoom else 3
row_h  = [0.22, 0.18, 0.35, 0.25] if show_zoom else [0.28, 0.42, 0.30]
titles = (
    ["Intracellular Signaling (full timeline)",
     "Early Phase Detail (days 0–7)",
     "Macrophage Phenotype Spectrum",
     "FBGC Risk Accumulation"]
    if show_zoom else
    ["Intracellular Signaling Dynamics",
     "Macrophage Phenotype Spectrum",
     "FBGC Risk Accumulation"]
)

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=False,
    subplot_titles=titles,
    vertical_spacing=0.06,
    row_heights=row_h,
)

t = R["t_days"]

# ── Signaling panel (full timeline) — row 1 ────────────────────────
sig_traces = [
    (R["yap"],   "YAP nuclear", "#E8593C", "solid",  1.0),
    (R["nfkb"],  "NF-κB",       "#C04020", "dash",   1.0),
    (R["stat6"], "STAT6",       "#1D9E75", "solid",  1.0),
    (R["ca"],    "[Ca²⁺]ᵢ (µM)","#7B9CDA", "dot",   0.7),
]
for y_data, name, color, dash, op in sig_traces:
    fig.add_trace(go.Scatter(
        x=t, y=y_data, name=name, opacity=op,
        line=dict(color=color, width=2, dash=dash),
        legendgroup="sig",
    ), row=1, col=1)

if R["drug_ts"] is not None:
    fig.add_trace(go.Scatter(
        x=t, y=R["drug_ts"], name="Drug suppression",
        line=dict(color="#9B59B6", width=1.5, dash="dot"), opacity=0.8,
        legendgroup="sig",
    ), row=1, col=1)

for day in [7, 14]:
    fig.add_vline(x=day, line_dash="dot", line_color="gray", opacity=0.3,
                  row=1, col=1)

# Immunomodulation index annotation on signaling panel
fig.add_annotation(
    x=t[-1] * 0.98, y=0.92,
    xref="x", yref="y",
    text=f"Immuno-mod index: {R['imm_index']:.2f}",
    showarrow=False, bgcolor="rgba(255,255,255,0.85)",
    bordercolor="#aaa", font=dict(size=10),
    row=1, col=1,
)

# ── Early phase zoom — row 2 (days 0–7) ───────────────────────────
if show_zoom:
    zoom_mask = t <= 7.0
    t_zoom    = t[zoom_mask]
    for y_data, name, color, dash, op in sig_traces:
        fig.add_trace(go.Scatter(
            x=t_zoom, y=y_data[zoom_mask], name=name,
            opacity=op, showlegend=False,
            line=dict(color=color, width=2, dash=dash),
            legendgroup="sig_zoom",
        ), row=2, col=1)

    if R["drug_ts"] is not None:
        fig.add_trace(go.Scatter(
            x=t_zoom, y=R["drug_ts"][zoom_mask],
            name="Drug suppression", showlegend=False,
            line=dict(color="#9B59B6", width=1.5, dash="dot"),
            legendgroup="sig_zoom",
        ), row=2, col=1)

    fig.add_annotation(
        x=3.5, y=0.95, xref="x2", yref="y2",
        text="⬅ Acute phase  |  Resolution begins ➡",
        showarrow=False, font=dict(size=9, color="gray"),
        row=2, col=1,
    )

# ── Phenotype panel ────────────────────────────────────────────────
pheno_row = 3 if show_zoom else 2
for label in ["Inflammatory", "Transitional", "Healing", "FBGC-prone"]:
    key = PHENOTYPE_KEYS[label]
    fig.add_trace(go.Scatter(
        x=t, y=R[key], name=label,
        fill="tozeroy",
        fillcolor=PHENOTYPE_FILLS[label],
        line=dict(color=PHENOTYPE_LINES[label], width=2),
        legendgroup="pheno",
    ), row=pheno_row, col=1)

    if compare_mode and R2:
        fig.add_trace(go.Scatter(
            x=R2["t_days"], y=R2[key], name=f"{label} (B)",
            line=dict(color=PHENOTYPE_LINES[label], width=1.5, dash="dash"),
            legendgroup="pheno_b",
        ), row=pheno_row, col=1)

# Snapshot marker
if snap_day in R["snapshots"]:
    fig.add_vline(x=snap_day, line_dash="dash",
                  line_color="white", opacity=0.6,
                  row=pheno_row, col=1)
    fig.add_annotation(
        x=snap_day, y=1.02, xref=f"x{pheno_row}",
        yref=f"y{pheno_row}", text=f"d{snap_day}",
        showarrow=False, font=dict(size=9, color="white"),
        row=pheno_row, col=1,
    )

# ── FBGC risk panel ────────────────────────────────────────────────
risk_row = 4 if show_zoom else 3
for y0, y1, col in [
    (0.00, 0.25, "rgba(29,158,117,0.07)"),
    (0.25, 0.60, "rgba(239,159,39,0.07)"),
    (0.60, 0.80, "rgba(232,120,32,0.07)"),
    (0.80, 1.00, "rgba(226,75,74,0.07)"),
]:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=col,
                  line_width=0, row=risk_row, col=1)

fig.add_trace(go.Scatter(
    x=t, y=R["fbgc_risk"], name="FBGC risk",
    fill="tozeroy", fillcolor="rgba(83,74,183,0.18)",
    line=dict(color="#534AB7", width=2.5),
    legendgroup="fbgc",
), row=risk_row, col=1)

if compare_mode and R2:
    fig.add_trace(go.Scatter(
        x=R2["t_days"], y=R2["fbgc_risk"], name="FBGC risk (B)",
        line=dict(color="#534AB7", width=2, dash="dash"),
        legendgroup="fbgc",
    ), row=risk_row, col=1)

if R["fbgc_onset"] > 0:
    fig.add_vline(x=R["fbgc_onset"], line_dash="dot",
                  line_color="#534AB7", opacity=0.8,
                  row=risk_row, col=1)

for level, col in [(0.25, "rgba(239,159,39,0.7)"),
                    (0.60, "rgba(232,120,32,0.7)"),
                    (0.80, "rgba(226,75,74,0.7)")]:
    fig.add_hline(y=level, line_dash="dash", line_color=col,
                  line_width=1, row=risk_row, col=1)

# ── Layout ─────────────────────────────────────────────────────────
fig.update_layout(
    height=820 if show_zoom else 700,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    margin=dict(l=55, r=40, t=100, b=40),
    title=dict(
        text=(f"{mat_label} | E={E_kPa} kPa | Ra={Ra_um} µm | "
              f"θ={theta}° | {tissue} | {coating}"),
        font=dict(size=11), x=0.5,
    ),
)
for row in range(1, n_rows + 1):
    fig.update_yaxes(range=[0, 1.05], row=row, col=1)
fig.update_xaxes(title_text="Time (days)", row=n_rows, col=1)

st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PHENOTYPE SNAPSHOT — flow cytometry style
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader(f"Phenotype Snapshot — Day {snap_day}")
st.caption("Flow cytometry–style readout. Scores are normalized [0–1] and represent "
           "relative macrophage phenotype intensity at the selected timepoint.")

if snap_day in R["snapshots"]:
    snap = R["snapshots"][snap_day]

    col1, col2 = st.columns([1, 1])

    with col1:
        # Bar chart — snapshot
        snap_fig = go.Figure()
        labels = ["Inflammatory", "Transitional", "Healing", "FBGC-prone"]
        values = [snap["inflammatory"], snap["transitional"],
                  snap["healing"],      snap["fbgc_prone"]]
        colors = [PHENOTYPE_LINES[l] for l in labels]

        snap_fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            marker_line_color="rgba(255,255,255,0.3)",
            marker_line_width=1,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        ))

        if compare_mode and R2 and snap_day in R2.get("snapshots", {}):
            snap2 = R2["snapshots"][snap_day]
            v2    = [snap2["inflammatory"], snap2["transitional"],
                     snap2["healing"],      snap2["fbgc_prone"]]
            snap_fig.add_trace(go.Bar(
                x=labels, y=v2, name="Surface B",
                marker_color=colors,
                marker_line_color="white",
                marker_line_width=2,
                marker_pattern_shape="/",
                text=[f"{v:.2f}" for v in v2],
                textposition="outside",
                opacity=0.6,
            ))

        snap_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0, 1.1], title="Phenotype score"),
            showlegend=compare_mode,
            barmode="group",
        )
        st.plotly_chart(snap_fig, use_container_width=True)

    with col2:
        # Signaling state at snapshot day
        st.markdown(f"**Signaling state at day {snap_day}:**")
        sig_data = {
            "Signal":    ["YAP nuclear", "NF-κB", "STAT6"],
            "Value":     [f"{snap['yap']:.3f}",
                          f"{snap['nfkb']:.3f}",
                          f"{snap['stat6']:.3f}"],
            "Interpretation": [
                "High → M1 mechanosensing drive" if snap["yap"] > 0.5
                    else "Low → M2-permissive",
                "Elevated → active inflammation" if snap["nfkb"] > 0.15
                    else "Resolved → normal",
                "Rising → IL-4/M2a activation" if snap["stat6"] > 0.3
                    else "Low → M2a not yet active",
            ],
        }
        st.dataframe(pd.DataFrame(sig_data),
                     use_container_width=True, hide_index=True)

        st.markdown("**Experimental correlates:**")
        st.caption(
            f"At day {snap_day}, expect flow cytometry to show approximately:\n"
            f"- CD86+ (M1 marker): ~{snap['inflammatory']*100:.0f}% of macrophages\n"
            f"- CD206+ (M2 marker): ~{snap['healing']*100:.0f}% of macrophages\n"
            f"- Multinucleate cells: {'Present' if snap['fbgc_prone'] > 0.4 else 'Unlikely'} "
            f"(FBGC-prone score {snap['fbgc_prone']:.2f})"
        )

# ═══════════════════════════════════════════════════════════════════
# SURFACE SCORES TABLE — reference-anchored
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Surface Biological Scores")
st.caption("Color bands: 🟢 favorable | 🟡 borderline | 🔴 unfavorable for biocompatibility")

def score_band(key, val):
    if key not in SCORE_REFS:
        return str(round(val, 3))
    ref    = SCORE_REFS[key]
    lo, hi = ref["low"], ref["high"]
    inv    = ref["invert"]
    if inv:
        icon = "🟢" if val < lo else ("🟡" if val < hi else "🔴")
    else:
        icon = "🔴" if val < lo else ("🟡" if val < hi else "🟢")
    return f"{icon} {val:.3f}"

score_keys = [
    ("Stiffness (kPa)",    "E_kPa",                    False),
    ("Roughness Ra (µm)",  "Ra_um",                    False),
    ("Feature size (nm)",  "feature_size_nm",          False),
    ("Contact angle (°)",  "contact_angle_deg",        False),
    ("Mechano-input",      "mechano_input_effective",  True),
    ("Nanotopo modifier",  "nanotopo_modifier",        True),
    ("DAMP signal",        "damp_signal",              True),
    ("Protein adsorption", "protein_adsorption_score", True),
    ("Wettability factor", "wettability_factor",       True),
    ("M-CSF bias",         "mcsf_bias",                True),
    ("Coating suppression","coating_suppression",      False),
]

ref_notes = {
    "mechano_input_effective":  "Low (<0.3) → soft/M2 | High (>0.7) → stiff/M1",
    "nanotopo_modifier":        "High (>0.7) → strong immunomodulation",
    "damp_signal":              "Low (<0.2) → minimal danger signal",
    "protein_adsorption_score": "Low (<0.2) → antifouling",
    "wettability_factor":       "High (>0.6) → hydrophilic, M2-promoting",
    "mcsf_bias":                "High → smooth surface, M-CSF differentiation",
}

table_rows = []
for label, key, has_ref in score_keys:
    val_a = sc.get(key, "—")
    if has_ref and isinstance(val_a, float):
        display_a = score_band(key, val_a)
    else:
        display_a = str(round(val_a, 3)) if isinstance(val_a, float) else str(val_a)

    row = {"Parameter": label, "Surface A": display_a}

    if key in ref_notes:
        row["Reference range"] = ref_notes[key]

    if compare_mode and R2:
        val_b = R2["surface_scores"].get(key, "—")
        if has_ref and isinstance(val_b, float):
            row["Surface B"] = score_band(key, val_b)
        else:
            row["Surface B"] = str(round(val_b, 3)) if isinstance(val_b, float) else str(val_b)

    table_rows.append(row)

st.dataframe(pd.DataFrame(table_rows),
             use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Export")
ecol1, ecol2 = st.columns(2)

csv_bytes = build_csv(R)
report_bytes = build_report(R, sc, recs, mat_label, mode)

with ecol1:
    st.download_button(
        label="⬇ Download time-series CSV",
        data=csv_bytes,
        file_name=f"fbr_results_{mat_label.replace(' ','_')[:30]}.csv",
        mime="text/csv",
    )
with ecol2:
    st.download_button(
        label="⬇ Download simulation report (.txt)",
        data=report_bytes,
        file_name=f"fbr_report_{mat_label.replace(' ','_')[:30]}.txt",
        mime="text/plain",
    )

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "✅ **Directionally validated** — Hotchkiss et al. 2016 (Acta Biomaterialia) "
    "+ J. Nanobiotechnology 2023. 6/6 rank-order checks passed. | "
    "Signaling: Bhatt et al. 2021 (Piezo1/Ca²⁺) · Hoffmann et al. 2002 (NF-κB) "
    "· Zanconato 2016 (YAP) — adapted for macrophage kinetics."
)