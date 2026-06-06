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

import os, sys
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io, datetime

# Ensure local package imports work even if Streamlit is launched from a different CWD.
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from inputs.surface_params import SurfaceInputs
from inputs.drug_profiles   import plga_microsphere, dual_phase, scaffold_guided
from signaling.piezo1_calcium       import Piezo1CalciumModule
from signaling.yap_hippo            import YAPHippoModule
from signaling.nfkb                 import NFkBModule
from signaling.stat6                import STAT6Module
try:
    from signaling.sasp_senescence      import (SenescenceInputs,
                                                 SASPSenescenceModule,
                                                 sasp_to_nfkb_amplification)
except Exception as exc:
    raise ImportError(
        f"Failed to import signaling.sasp_senescence: {exc!r}. "
        "Ensure the repository root is on PYTHONPATH, and that required packages are installed: "
        "pip install -r requirements.txt"
    ) from exc
from phenotype.phenotype_classifier import PhenotypeClassifier
from phenotype.fbgc_risk            import FBGCRiskModule
from body import (
    render_body_selector,
    body_region_panel,
    BODY_REGIONS,
    HOTSPOTS
)

# ==================================================================
# CONSTANTS
# ==================================================================
DRUG_MAP = {
    "None":            None,
    "PLGA 7-day":      lambda: plga_microsphere(half_life_days=7),
    "PLGA 14-day":     lambda: plga_microsphere(half_life_days=14),
    "Dual-phase":      dual_phase,
    "Scaffold-guided": scaffold_guided,
}

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
        "note": "Bioinert, hydrophobic. Fibrous encapsulation risk.",
    },
    "PEEK + HA coating": {
        "E_kPa": 50.0, "Ra_um": 0.8, "feature_size_nm": 500.0,
        "contact_angle_deg": 40.0, "tissue_context": "bone",
        "coating": "none", "drug_key": "None",
        "note": "HA-coated PEEK. Partially addresses hydrophobicity.",
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
        "note": "Rigid Si probe. Catastrophic stiffness mismatch with brain.",
    },
    "Neural probe + zwitterionic coating": {
        "E_kPa": 200.0, "Ra_um": 0.1, "feature_size_nm": 5000.0,
        "contact_angle_deg": 15.0, "tissue_context": "brain",
        "coating": "zwitterionic", "drug_key": "None",
        "note": "Zwitterionic-coated neural probe. State-of-art biocompatibility.",
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
        "note": "Ultra-soft, ultra-hydrophilic. Minimal FBR. Gold standard for CGM sensors.",
    },
}

PHENOTYPE_FILLS = {
    "Inflammatory": "rgba(226,75,74,0.12)",
    "Transitional":  "rgba(239,159,39,0.12)",
    "Healing":       "rgba(29,158,117,0.12)",
    "FBGC-prone":    "rgba(127,119,221,0.12)",
}
PHENOTYPE_LINES = {
    "Inflammatory": "#E24B4A",
    "Transitional":  "#EF9F27",
    "Healing":       "#1D9E75",
    "FBGC-prone":    "#7F77DD",
}
PHENOTYPE_KEYS = {
    "Inflammatory": "inflam",
    "Transitional":  "transit",
    "Healing":       "healing",
    "FBGC-prone":    "fbgc_prone",
}
SNAPSHOT_DAYS = [1, 3, 7, 14, 28]
SCORE_REFS = {
    "mechano_input_effective":  {"low": 0.3, "high": 0.7, "invert": False},
    "nanotopo_modifier":        {"low": 0.3, "high": 0.7, "invert": True},
    "damp_signal":              {"low": 0.2, "high": 0.5, "invert": True},
    "protein_adsorption_score": {"low": 0.2, "high": 0.5, "invert": True},
    "wettability_factor":       {"low": 0.3, "high": 0.6, "invert": False},
    "mcsf_bias":                {"low": 0.4, "high": 0.7, "invert": False},
}


# ═══════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_model(E_kPa, Ra_um, feature_nm, contact_angle,
              tissue, coating, drug_key, mode,
              patient_age, t_days, dt_min):

    drug_fn = None
    if drug_key != "None" and DRUG_MAP.get(drug_key) is not None:
        drug_fn = DRUG_MAP[drug_key]()

    s = SurfaceInputs(
        E_kPa=E_kPa, Ra_um=Ra_um, feature_size_nm=feature_nm,
        contact_angle_deg=contact_angle, tissue_context=tissue,
        coating=coating, drug_release_profile=drug_fn,
    )
    t = np.arange(0, t_days * 24 * 60, dt_min)

    ca = Piezo1CalciumModule(s).simulate(t)

    # ── Senescence: two-pass NF-kB ────────────────────────────────
    nfkb_prelim = NFkBModule(s, ca).simulate(t)

    sen_inputs = SenescenceInputs(
        patient_age        = patient_age,
        tissue_context     = tissue,
        E_kPa              = E_kPa,
        contact_angle_deg  = contact_angle,
        protein_adsorption = s.protein_adsorption_score,
    )
    sasp_module  = SASPSenescenceModule(sen_inputs, nfkb_prelim, t)
    sasp_results = sasp_module.simulate(t)
    sasp_amp     = sasp_to_nfkb_amplification(
        sasp_results["sasp_drive"], sen_inputs.s_total
    )
    nfkb = NFkBModule(s, ca, sasp_amplification=sasp_amp).simulate(t)
    # ── End senescence ────────────────────────────────────────────

    yap   = YAPHippoModule(s, ca).simulate(t)
    stat6 = STAT6Module(s, t).simulate(t)
    ps    = PhenotypeClassifier(
        yap, nfkb, stat6, t,
        mcsf_bias=s.mcsf_bias, mode=mode,
        wettability_factor=s.wettability_factor,
    ).compute()
    fb = FBGCRiskModule(s, stat6, nfkb, t).compute_risk()

    t_days_arr = t / (24 * 60)
    drug_ts = None
    if drug_fn is not None:
        drug_ts = np.array([float(drug_fn(ti)) for ti in t])

    snapshots = {}
    for day in SNAPSHOT_DAYS:
        if day <= t_days:
            idx = min(np.searchsorted(t_days_arr, day), len(t_days_arr)-1)
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
        # Core outputs
        "surface_scores":     surface_scores,
        "t_days":             t_days_arr,
        "ca":                 ca,
        "yap":                yap,
        "nfkb":               nfkb,
        "stat6":              stat6,
        "inflam":             ps.inflammatory,
        "transit":            ps.transitional,
        "healing":            ps.healing,
        "fbgc_prone":         ps.fbgc_prone,
        "fbgc_risk":          fb["risk_timeseries"],
        "fbgc_peak":          float(fb["peak_risk"]),
        "fbgc_cat":           str(fb["risk_category"]),
        "fbgc_onset":         float(fb["onset_day"]),
        "imm_index":          float(ps.immunomodulation_index),
        "traj_score":         float(ps.trajectory_match_score),
        "dominant":           str(ps.dominant_phenotype()),
        "drug_ts":            drug_ts,
        "snapshots":          snapshots,
        "mode":               mode,
        # Senescence outputs (all pickle-safe)
        "patient_age":        float(patient_age),
        "age_group":          str(sen_inputs.age_label()),
        "s_age":              float(sen_inputs.s_age),
        "s_material":         float(sen_inputs.s_material),
        "s_total":            float(sen_inputs.s_total),
        "sen_risk_cat":       str(sen_inputs.risk_category()),
        "senescent_fraction": sasp_results["senescent_fraction"],
        "sasp_drive":         sasp_results["sasp_drive"],
        "il17_timeseries":    sasp_results["il17_timeseries"],
        "peak_senescent":     float(sasp_results["peak_senescent"]),
        "peak_il17":          float(sasp_results["peak_il17"]),
        "sasp_amp":           sasp_amp,
    }


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════
def generate_recommendations(R: dict, sc: dict) -> list:
    recs = []
    peak = R["fbgc_peak"]
    idx  = R["imm_index"]
    damp = sc["damp_signal"]
    ads  = sc["protein_adsorption_score"]
    mech = sc["mechano_input_effective"]
    coat = sc["coating_suppression"]
    mode = R["mode"]
    s_total = R["s_total"]
    age  = R["patient_age"]

    # FBGC risk
    if peak > 0.60:
        recs.append({"level": "critical", "title": "High FBGC risk detected",
            "detail": f"Peak risk {peak*100:.0f}%. Giant cell formation likely. "
                      "Recommend: zwitterionic or phospholipid coating, or PLGA 7-day drug release."})
    elif peak > 0.25:
        recs.append({"level": "warning", "title": "Moderate FBGC risk",
            "detail": f"Peak risk {peak*100:.0f}%. Monitor DC-STAMP and CD98 at day 7–14. "
                      "Consider surface hydrophilization."})
    else:
        recs.append({"level": "good", "title": "FBGC risk well-controlled",
            "detail": f"Peak risk {peak*100:.0f}%. Surface unfavorable for giant cell fusion."})

    # Immunomodulation
    if idx < 0.8:
        recs.append({"level": "critical", "title": "M1-dominated response predicted",
            "detail": f"Index {idx:.2f}. Chronic inflammation risk. Reduce stiffness "
                      "below 20 kPa or add phospholipid coating."})
    elif idx < 1.5:
        recs.append({"level": "warning", "title": "Balanced M1/M2 — resolution not guaranteed",
            "detail": f"Index {idx:.2f}. Extend experimental timepoints beyond day 14."})
    else:
        recs.append({"level": "good", "title": "M2-favorable immunomodulation",
            "detail": f"Index {idx:.2f}. Surface drives healing-dominant response."})

    # DAMP
    if damp > 0.5:
        recs.append({"level": "critical", "title": "Elevated DAMP signal",
            "detail": f"DAMP score {damp:.3f}. Protein adsorption {ads:.3f}. "
                      "Recommend: plasma treatment or zwitterionic coating."})
    elif damp > 0.25:
        recs.append({"level": "warning", "title": "Moderate DAMP signal",
            "detail": f"DAMP score {damp:.3f}. Consider hydrophilization."})

    # Coating
    if coat == 0.0:
        recs.append({"level": "warning", "title": "No surface coating applied",
            "detail": "Consider PEG (75%), zwitterionic (85%), or phospholipid (90%) coating."})

    # Scaffold stiffness
    if mode == "scaffold" and sc["E_kPa"] > 30:
        recs.append({"level": "critical",
            "title": "Stiffness too high for scaffold mode",
            "detail": f"Stiffness {sc['E_kPa']:.0f} kPa exceeds scaffold range (5–20 kPa soft tissue). "
                      "This explains your low trajectory match score. Target <20 kPa."})

    # FBGC-prone vs risk gap
    snap_28 = R.get("snapshots", {}).get(28, {})
    fbgc_prone_28 = snap_28.get("fbgc_prone", 0)
    if fbgc_prone_28 > 0.35 and peak < 0.55:
        recs.append({"level": "warning",
            "title": "Fusion-competent macrophages present — risk partially gated",
            "detail": f"FBGC-prone score {fbgc_prone_28:.2f} at day 28. Macrophages are "
                      f"fusion-competent but 72h sustained IL-4 threshold not fully met. "
                      "Monitor DC-STAMP and CD98 at day 7–14."})

    # Nanotopo absent
    if sc.get("nanotopo_modifier", 1.0) < 0.01:
        recs.append({"level": "warning",
            "title": "No immunomodulatory nanotopography at this feature size",
            "detail": f"Feature size {sc['feature_size_nm']:.0f} nm is above immunomodulatory "
                      "range. Target 15–50 nm for nanotopographic immunomodulation."})

    # Brain stiffness
    if sc["E_kPa"] > 50 and sc.get("tissue_context","") == "brain":
        recs.append({"level": "critical",
            "title": "Severe stiffness mismatch for brain context",
            "detail": "Brain tissue is 0.1–1 kPa. This stiffness will drive maximum "
                      "neuroinflammation. Consider flexible polymer or hydrogel encapsulation."})

    # ── SENESCENCE RECOMMENDATIONS ─────────────────────────────────
    if s_total > 0.10:
        recs.append({"level": "critical",
            "title": "High senescent burden — SASP amplifying NF-κB",
            "detail": f"Total senescent fraction {s_total*100:.1f}% "
                      f"(age-related: {R['s_age']*100:.1f}%, surface-induced: {R['s_material']*100:.1f}%). "
                      "SASP cytokines (IL-6, IL-1β, TNF-α) are significantly amplifying baseline "
                      "NF-κB. Consider senolytic preconditioning (navitoclax) before implantation. "
                      "Source: Chung et al. 2020 (Sci Transl Med)."})
    elif s_total > 0.05:
        recs.append({"level": "warning",
            "title": "Moderate senescent burden — SASP pre-load present",
            "detail": f"Senescent fraction {s_total*100:.1f}%. Moderate SASP amplification "
                      "of NF-κB. IL-17 feedback loop will activate after day 7. "
                      "Monitor for delayed fibrotic response beyond day 14."})
    elif s_total > 0.02:
        recs.append({"level": "warning",
            "title": "Low senescent burden detected",
            "detail": f"Senescent fraction {s_total*100:.1f}%. Minimal SASP contribution. "
                      "Model behavior close to young healthy patient baseline."})

    if R["s_material"] > 0.02:
        recs.append({"level": "warning",
            "title": "Surface is inducing local stress senescence",
            "detail": f"Surface-induced senescence S_material={R['s_material']*100:.1f}%. "
                      "High stiffness (>{:.0f} kPa) and/or hydrophobicity are driving "
                      "stress-induced premature senescence (SIPS) in peri-implant fibroblasts. "
                      "This is independent of patient age. "
                      "Source: Mrozik et al. 2024; Yang et al. 2023.".format(15)})

    if age > 65 and peak > 0.20:
        recs.append({"level": "critical",
            "title": "Age + FBGC risk combination — elevated clinical concern",
            "detail": f"Patient age {age:.0f}y combined with FBGC risk {peak*100:.0f}% "
                      "represents a clinically significant combination. Aged tissue has reduced "
                      "immune clearance of senescent cells, sustaining the IL-17 feedback loop. "
                      "Anti-senescence preconditioning strongly recommended."})

    return recs


# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════
def build_csv(R: dict) -> bytes:
    t  = R["t_days"]
    df = pd.DataFrame({
        "day":                t,
        "Ca_uM":              R["ca"],
        "YAP_nuclear":        R["yap"],
        "NFkB":               R["nfkb"],
        "STAT6":              R["stat6"],
        "inflammatory":       R["inflam"],
        "transitional":       R["transit"],
        "healing":            R["healing"],
        "fbgc_prone":         R["fbgc_prone"],
        "fbgc_risk":          R["fbgc_risk"],
        "senescent_fraction": R["senescent_fraction"],
        "sasp_drive":         R["sasp_drive"],
        "il17":               R["il17_timeseries"],
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
        "  FBR MECHANOTRANSDUCTION MODEL — SIMULATION REPORT (v10)",
        f"  Generated: {now}",
        "=" * 65,
        "",
        f"Material:        {material_name}",
        f"Simulation mode: {mode.upper()}",
        f"Tissue context:  {sc.get('tissue_context','N/A')}",
        f"Coating:         {sc.get('coating_name','N/A')} ({sc.get('coating_suppression',0):.0%} suppression)",
        "",
        "PATIENT BIOLOGY",
        "-" * 40,
        f"  Patient age:       {R['patient_age']:.0f} years ({R['age_group']})",
        f"  S_age (baseline):  {R['s_age']*100:.2f}%",
        f"  S_material:        {R['s_material']*100:.2f}%",
        f"  S_total:           {R['s_total']*100:.2f}% ({R['sen_risk_cat']} senescent burden)",
        f"  Peak IL-17:        {R['peak_il17']:.3f}",
        "",
        "SURFACE PARAMETERS",
        "-" * 40,
        f"  Stiffness:          {sc['E_kPa']} kPa",
        f"  Roughness Ra:       {sc['Ra_um']} µm",
        f"  Feature size:       {sc['feature_size_nm']} nm",
        f"  Contact angle:      {sc['contact_angle_deg']}°",
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
        f"  FBGC onset day:            " +
            (f"{R['fbgc_onset']:.1f}" if R['fbgc_onset'] > 0 else "Not reached"),
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
        tag = {"critical":"[CRITICAL]","warning":"[WARNING]",
               "good":"[OK]"}.get(r["level"],"")
        lines.append(f"  {tag} {r['title']}")
        lines.append(f"    {r['detail']}")
        lines.append("")

    lines += [
        "VALIDATION",
        "-" * 40,
        "  Hotchkiss et al. 2016 (Acta Biomaterialia) — 6/6 rank-order checks passed",
        "  J. Nanobiotechnology 2023 (TiO2 nanotubes) — 2/2 checks passed",
        "  Senescence: Tuttle 2020 (Aging); Chung 2020 (Sci Transl Med);",
        "              Mrozik 2024; Yang 2023 (IJMS); Herbstein 2024 (Aging Cell)",
        "=" * 65,
    ]
    return "\n".join(lines).encode()


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FBR Mechanotransduction Model",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 FBR Model")
    st.caption("Mechanotransduction → Phenotype Engine")
    st.divider()

    # Material preset
    st.subheader("Material Preset")
    selected_material = st.selectbox(
        "Select implant material", list(MATERIAL_PRESETS.keys()))
    preset = MATERIAL_PRESETS[selected_material]
    if preset:
        st.caption(f"📖 {preset['note']}")

    st.divider()

    mode    = st.radio("Simulation mode", ["suppress", "scaffold"])
    tissue  = st.selectbox("Tissue context",
                           ["cartilage","soft_tissue","brain","bone"],
                           index=(["cartilage","soft_tissue","brain","bone"]
                                  .index(preset["tissue_context"]) if preset else 0))
    coating = st.selectbox("Surface coating",
                           ["none","peg","zwitterionic","phospholipid"],
                           index=(["none","peg","zwitterionic","phospholipid"]
                                  .index(preset["coating"]) if preset else 0))
    drug_key = st.selectbox("Drug release profile", list(DRUG_MAP.keys()),
                            index=(list(DRUG_MAP.keys())
                                   .index(preset["drug_key"]) if preset else 0))
    t_days  = st.slider("Simulation duration (days)", 7, 56, 28)

    st.divider()

    # ── Body map ──────────────────────────────────────────────────
    st.subheader("Implant Site")
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = "Knee"
    body_html = render_body_selector(st.session_state.selected_region)
    components.html(body_html, height=500, scrolling=False)
    region_names = list(BODY_REGIONS.keys())
    cols = st.columns(2)
    for i, rname in enumerate(region_names):
        info = BODY_REGIONS[rname]
        if cols[i % 2].button(
            f"{info['icon']} {rname}", key=f"btn_{rname}",
            use_container_width=True,
            type="primary" if rname == st.session_state.selected_region else "secondary",
        ):
            st.session_state.selected_region = rname
            st.rerun()
    selected_region  = st.session_state.selected_region
    region_info      = BODY_REGIONS[selected_region]
    tissue_E_default = region_info["E_kPa_default"]

    st.divider()
    st.subheader("Surface Parameters")
    E_def    = preset["E_kPa"]            if preset else tissue_E_default
    Ra_def   = preset["Ra_um"]            if preset else 0.5
    feat_def = preset["feature_size_nm"]  if preset else 200.0
    th_def   = preset["contact_angle_deg"] if preset else 70.0

    E_kPa   = st.slider("Stiffness (kPa)",   0.1,  200.0, float(E_def),    step=0.5)
    Ra_um   = st.slider("Roughness Ra (µm)", 0.01,   5.0, float(Ra_def),   step=0.01)
    feat_nm = st.slider("Feature size (nm)",  5.0, 2000.0, float(feat_def), step=5.0)
    theta   = st.slider("Contact angle (°)",  0.0,  150.0, float(th_def),   step=1.0)

    st.divider()

    # ── PATIENT BIOLOGY (NEW v10) ──────────────────────────────────
    st.subheader("Patient Biology")
    patient_age = st.slider(
        "Patient age (years)", 18, 90, 45,
        help="Controls senescent cell baseline burden (S_age) by tissue. "
             "Also modulates immune clearance rate of senescent cells. "
             "Young <35y | Middle-aged 36–65y | Older >65y"
    )
    age_label = ("Young" if patient_age <= 35
                 else "Middle-aged" if patient_age <= 65
                 else "Older")
    st.caption(f"Age group: **{age_label}** | Higher age → greater SASP pre-load → elevated baseline NF-κB")

    st.divider()
    compare_mode = st.checkbox("Compare two surfaces")
    if compare_mode:
        st.subheader("Surface B")
        mat2 = st.selectbox("Material B", list(MATERIAL_PRESETS.keys()), key="mat2")
        p2   = MATERIAL_PRESETS[mat2]
        if p2: st.caption(f"📖 {p2['note']}")
        E2     = st.slider("Stiffness B (kPa)",   0.1, 200.0,
                           float(p2["E_kPa"]) if p2 else 1.0, step=0.5, key="E2")
        Ra2    = st.slider("Roughness B (µm)",    0.01,  5.0,
                           float(p2["Ra_um"]) if p2 else 0.05, step=0.01, key="Ra2")
        feat2  = st.slider("Feature size B (nm)",  5.0, 2000.0,
                           float(p2["feature_size_nm"]) if p2 else 100.0, step=5.0, key="f2")
        theta2 = st.slider("Contact angle B (°)",  0.0, 150.0,
                           float(p2["contact_angle_deg"]) if p2 else 25.0, step=1.0, key="th2")
        tissue2  = p2["tissue_context"] if p2 else tissue
        coating2 = p2["coating"]        if p2 else coating
        drug2    = p2["drug_key"]       if p2 else drug_key
    else:
        p2 = None

    st.divider()
    snap_day  = st.select_slider("View phenotype at day:",
                                  options=[d for d in SNAPSHOT_DAYS if d <= t_days],
                                  value=min(7, t_days))
    show_zoom = st.checkbox("Show early-phase zoom (days 0–7)", value=True)

dt_min = 10.0

# ═══════════════════════════════════════════════════════════════════
# RUN MODEL
# ═══════════════════════════════════════════════════════════════════
try:
    with st.spinner("Running simulation..."):
        R = run_model(E_kPa, Ra_um, feat_nm, theta,
                      tissue, coating, drug_key, mode,
                      patient_age, t_days, dt_min)
        R2 = None
        if compare_mode and p2:
            R2 = run_model(E2, Ra2, feat2, theta2,
                           tissue2, coating2, drug2, mode,
                           patient_age, t_days, dt_min)
except Exception as e:
    st.error(f"Simulation error: {e}")
    st.stop()

sc   = R["surface_scores"]
recs = generate_recommendations(R, sc)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
mat_label = (selected_material if selected_material != "— Custom (manual) —"
             else "Custom surface")
st.title("Foreign Body Response — Mechanotransduction Model")
st.caption(
    f"**{mat_label}** | Mode: **{mode.upper()}** | Tissue: **{tissue}** | "
    f"Coating: **{coating}** | Drug: **{drug_key}** | "
    f"Patient: **{age_label} ({patient_age:.0f}y)**"
)

# ── Main KPI row ──────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dominant Phenotype", R["dominant"])
c2.metric("Immuno-mod Index",   f"{R['imm_index']:.2f}",
          help="M2/M1 AUC ratio. >1.5 = M2-favorable")
risk_delta = None
if compare_mode and R2:
    risk_delta = f"{(R['fbgc_peak']-R2['fbgc_peak'])*100:+.1f}% vs B"
c3.metric("FBGC Peak Risk",  f"{R['fbgc_peak']*100:.1f}%",
          delta=risk_delta, delta_color="inverse")
c4.metric("FBGC Category",   R["fbgc_cat"])
onset_str = f"Day {R['fbgc_onset']:.1f}" if R["fbgc_onset"] > 0 else "Not reached"
c5.metric("FBGC Onset", onset_str)

# ── Senescence KPI row (NEW v10) ──────────────────────────────────
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Patient Age Group",  R["age_group"])
sc2.metric("Senescent Burden",
           f"{R['s_total']*100:.1f}%",
           help=f"S_age {R['s_age']*100:.1f}% + S_material {R['s_material']*100:.1f}%")
sc3.metric("Senescence Risk",    R["sen_risk_cat"])
sc4.metric("Peak IL-17",         f"{R['peak_il17']:.3f}",
           help="IL-17 drives the senescence feedback loop from day 7+")

if mode == "scaffold":
    score = R["traj_score"]
    msg = ("✅ Excellent productive fibrosis potential" if score > 0.7
           else "⚠️ Moderate — consider tuning drug profile" if score > 0.4
           else "❌ Poor — surface unlikely to support guided regeneration")
    st.info(f"🧬 Scaffold trajectory match: **{score:.3f} / 1.000** — {msg}")

st.divider()

# ── Body region info panel ────────────────────────────────────────
with st.expander(
    f"{BODY_REGIONS[selected_region]['icon']} Implant site: {selected_region}",
    expanded=True
):
    body_region_panel(selected_region)

# ── Recommendations ───────────────────────────────────────────────
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
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════════
n_rows = 4 if show_zoom else 3
row_h  = [0.20, 0.16, 0.35, 0.29] if show_zoom else [0.26, 0.44, 0.30]
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
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                    subplot_titles=titles, vertical_spacing=0.06,
                    row_heights=row_h)
t = R["t_days"]

# Panel 1: Signaling
for y_data, name, color, dash, op in [
    (R["yap"],   "YAP nuclear", "#E8593C", "solid", 1.0),
    (R["nfkb"],  "NF-κB",       "#C04020", "dash",  1.0),
    (R["stat6"], "STAT6",       "#1D9E75", "solid", 1.0),
    (R["ca"],    "[Ca²⁺]ᵢ",     "#7B9CDA", "dot",   0.7),
]:
    fig.add_trace(go.Scatter(x=t, y=y_data, name=name, opacity=op,
        line=dict(color=color, width=2, dash=dash),
        legendgroup="sig"), row=1, col=1)

# SASP amplification on signaling panel
fig.add_trace(go.Scatter(x=t, y=R["sasp_amp"], name="SASP→NF-κB amp",
    line=dict(color="#FFD700", width=1.5, dash="dot"), opacity=0.8,
    legendgroup="sig"), row=1, col=1)

if R["drug_ts"] is not None:
    fig.add_trace(go.Scatter(x=t, y=R["drug_ts"], name="Drug suppression",
        line=dict(color="#9B59B6", width=1.5, dash="dot"), opacity=0.8,
        legendgroup="sig"), row=1, col=1)

for day, lbl in [(7,"d7"),(14,"d14")]:
    fig.add_vline(x=day, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
    fig.add_annotation(x=day+0.2, y=0.93, xref="x", yref="y",
        text=lbl, showarrow=False, font=dict(size=9, color="gray"), row=1, col=1)

# Panel 2: Early zoom
if show_zoom:
    zm = t <= 7.0
    for y_data, name, color, dash, op in [
        (R["yap"],   "YAP nuclear", "#E8593C", "solid", 1.0),
        (R["nfkb"],  "NF-κB",       "#C04020", "dash",  1.0),
        (R["stat6"], "STAT6",       "#1D9E75", "solid", 1.0),
        (R["ca"],    "[Ca²⁺]ᵢ",     "#7B9CDA", "dot",   0.7),
    ]:
        fig.add_trace(go.Scatter(x=t[zm], y=y_data[zm], name=name,
            opacity=op, showlegend=False,
            line=dict(color=color, width=2, dash=dash),
            legendgroup="sig_zoom"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t[zm], y=R["sasp_amp"][zm],
        name="SASP amp", showlegend=False,
        line=dict(color="#FFD700", width=1.5, dash="dot"),
        legendgroup="sig_zoom"), row=2, col=1)

# Phenotype panel
pheno_row = 3 if show_zoom else 2
for label in ["Inflammatory","Transitional","Healing","FBGC-prone"]:
    key = PHENOTYPE_KEYS[label]
    fig.add_trace(go.Scatter(
        x=t, y=R[key], name=label,
        fill="tozeroy", fillcolor=PHENOTYPE_FILLS[label],
        line=dict(color=PHENOTYPE_LINES[label], width=2),
        legendgroup="pheno"), row=pheno_row, col=1)
    if compare_mode and R2:
        fig.add_trace(go.Scatter(
            x=R2["t_days"], y=R2[key], name=f"{label} (B)",
            line=dict(color=PHENOTYPE_LINES[label], width=1.5, dash="dash"),
            legendgroup="pheno_b"), row=pheno_row, col=1)

if snap_day in R["snapshots"]:
    fig.add_vline(x=snap_day, line_dash="dash",
                  line_color="white", opacity=0.5, row=pheno_row, col=1)

# FBGC risk panel
risk_row = 4 if show_zoom else 3
for y0, y1, col in [
    (0.00,0.25,"rgba(29,158,117,0.07)"),
    (0.25,0.60,"rgba(239,159,39,0.07)"),
    (0.60,0.80,"rgba(232,120,32,0.07)"),
    (0.80,1.00,"rgba(226,75,74,0.07)"),
]:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0, row=risk_row, col=1)

fig.add_trace(go.Scatter(x=t, y=R["fbgc_risk"], name="FBGC risk",
    fill="tozeroy", fillcolor="rgba(83,74,183,0.18)",
    line=dict(color="#534AB7", width=2.5),
    legendgroup="fbgc"), row=risk_row, col=1)

if compare_mode and R2:
    fig.add_trace(go.Scatter(x=R2["t_days"], y=R2["fbgc_risk"], name="FBGC risk (B)",
        line=dict(color="#534AB7", width=2, dash="dash"),
        legendgroup="fbgc"), row=risk_row, col=1)

if R["fbgc_onset"] > 0:
    fig.add_vline(x=R["fbgc_onset"], line_dash="dot",
                  line_color="#534AB7", opacity=0.8, row=risk_row, col=1)

fig.update_layout(
    height=820 if show_zoom else 700,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    margin=dict(l=55,r=40,t=100,b=40),
    title=dict(text=(f"{mat_label} | E={E_kPa} kPa | Ra={Ra_um} µm | "
                     f"θ={theta}° | {tissue} | Age {patient_age:.0f}y"),
               font=dict(size=11), x=0.5),
)
for row in range(1, n_rows+1):
    fig.update_yaxes(range=[0, 1.05], row=row, col=1)
fig.update_xaxes(title_text="Time (days)", row=n_rows, col=1)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PHENOTYPE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader(f"Phenotype Snapshot — Day {snap_day}")
st.caption("Flow cytometry–style readout at selected timepoint.")
if snap_day in R["snapshots"]:
    snap = R["snapshots"][snap_day]
    col1, col2 = st.columns([1, 1])
    with col1:
        snap_fig = go.Figure()
        labels = ["Inflammatory","Transitional","Healing","FBGC-prone"]
        values = [snap["inflammatory"],snap["transitional"],
                  snap["healing"],snap["fbgc_prone"]]
        colors = [PHENOTYPE_LINES[l] for l in labels]
        snap_fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors,
            text=[f"{v:.2f}" for v in values], textposition="outside"))
        if compare_mode and R2 and snap_day in R2.get("snapshots",{}):
            s2 = R2["snapshots"][snap_day]
            v2 = [s2["inflammatory"],s2["transitional"],s2["healing"],s2["fbgc_prone"]]
            snap_fig.add_trace(go.Bar(x=labels, y=v2, name="Surface B",
                marker_color=colors, marker_pattern_shape="/", opacity=0.6,
                text=[f"{v:.2f}" for v in v2], textposition="outside"))
        snap_fig.update_layout(template="plotly_dark", height=300,
            margin=dict(l=10,r=10,t=30,b=10),
            yaxis=dict(range=[0,1.1],title="Score"), barmode="group")
        st.plotly_chart(snap_fig, use_container_width=True)
    with col2:
        st.markdown(f"**Signaling at day {snap_day}:**")
        st.dataframe(pd.DataFrame({
            "Signal":          ["YAP nuclear","NF-κB","STAT6"],
            "Value":           [f"{snap['yap']:.3f}",f"{snap['nfkb']:.3f}",f"{snap['stat6']:.3f}"],
            "Interpretation":  [
                "High → M1 drive" if snap["yap"]>0.5 else "Low → M2-permissive",
                "Elevated → active inflammation" if snap["nfkb"]>0.15 else "Resolved",
                "Rising → M2a activation" if snap["stat6"]>0.3 else "Not yet active",
            ],
        }), use_container_width=True, hide_index=True)
        st.markdown("**Experimental correlates:**")
        st.caption(
            f"Day {snap_day} flow cytometry estimates:\n"
            f"- CD86+ (M1): ~{snap['inflammatory']*100:.0f}%\n"
            f"- CD206+ (M2): ~{snap['healing']*100:.0f}%\n"
            f"- FBGCs: {'Possible' if snap['fbgc_prone']>0.4 else 'Unlikely'} "
            f"(FBGC-prone {snap['fbgc_prone']:.2f})"
        )

# ═══════════════════════════════════════════════════════════════════
# SENESCENCE / SASP PANEL (NEW v10)
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Senescence & SASP Dynamics")
st.caption(
    "Two-component senescent burden: **S_age** (pre-existing, age-dependent, tissue-specific) "
    "+ **S_material** (surface-induced stress senescence via stiffness, hydrophobicity, and "
    "protein adsorption). SASP amplifies NF-κB above the surface-driven baseline. "
    "IL-17 feedback loop activates after day 7. "
    "Sources: Tuttle 2020 (Aging); Chung 2020 (Sci Transl Med); Mrozik 2024; Yang 2023 (IJMS)."
)

sen_col1, sen_col2 = st.columns([1, 2])

with sen_col1:
    # S_age vs S_material stacked bar
    brk_fig = go.Figure()
    brk_fig.add_trace(go.Bar(
        name="S_age (age-dependent baseline)",
        x=["Senescent Burden"],
        y=[R["s_age"] * 100],
        marker_color="rgba(239,159,39,0.85)",
        text=[f"{R['s_age']*100:.2f}%"], textposition="inside",
    ))
    brk_fig.add_trace(go.Bar(
        name="S_material (surface-induced)",
        x=["Senescent Burden"],
        y=[R["s_material"] * 100],
        marker_color="rgba(226,75,74,0.85)",
        text=[f"{R['s_material']*100:.2f}%"], textposition="inside",
    ))
    brk_fig.update_layout(
        barmode="stack", template="plotly_dark", height=300,
        margin=dict(l=10,r=10,t=30,b=10),
        yaxis_title="% stromal cells senescent",
        title=dict(text="Burden breakdown", font=dict(size=11)),
        legend=dict(orientation="h", y=-0.35),
    )
    st.plotly_chart(brk_fig, use_container_width=True)
    st.caption(
        f"**Reference (tissue: {tissue}):**  \n"
        f"Young (<35y): <1% | Middle (36–65y): 2–5% | Older (>65y): 6–15%  \n"
        f"Source: Tuttle et al. 2020 (Aging)"
    )

with sen_col2:
    # Senescence timeseries
    sen_fig = go.Figure()
    sen_fig.add_trace(go.Scatter(
        x=t, y=R["senescent_fraction"]*100,
        name="Senescent cells (%)",
        line=dict(color="#EF9F27", width=2),
        fill="tozeroy", fillcolor="rgba(239,159,39,0.12)",
    ))
    sen_fig.add_trace(go.Scatter(
        x=t, y=R["sasp_drive"],
        name="SASP drive",
        line=dict(color="#E24B4A", width=2, dash="dash"),
    ))
    sen_fig.add_trace(go.Scatter(
        x=t, y=R["il17_timeseries"],
        name="IL-17 (feedback loop)",
        line=dict(color="#7F77DD", width=2),
    ))
    sen_fig.add_vline(x=7, line_dash="dot", line_color="gray", opacity=0.5)
    sen_fig.add_annotation(x=7.5, y=0.9, xref="x", yref="paper",
        text="IL-17 onset (day 7)", showarrow=False, font=dict(size=9, color="gray"))
    sen_fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=10,r=10,t=30,b=10),
        yaxis_title="Signal [0–1] / % cells",
        xaxis_title="Time (days post-implantation)",
        title=dict(text="Senescence / SASP / IL-17 dynamics", font=dict(size=11)),
        legend=dict(orientation="h", y=-0.35),
    )
    st.plotly_chart(sen_fig, use_container_width=True)

# Senescence interpretation banner
s_pct = R["s_total"] * 100
if s_pct < 2:
    st.success(f"**Minimal senescent burden ({s_pct:.1f}%)** — SASP contribution negligible. Model behaves as young healthy patient.")
elif s_pct < 5:
    st.info(f"**Low senescent burden ({s_pct:.1f}%)** — Mild SASP pre-load. Slightly elevated baseline NF-κB. IL-17 response modest after day 7.")
elif s_pct < 10:
    st.warning(f"**Moderate senescent burden ({s_pct:.1f}%)** — SASP meaningfully amplifies NF-κB. IL-17 feedback loop active day 7–14. Consider senolytic preconditioning.")
else:
    st.error(f"**High senescent burden ({s_pct:.1f}%)** — Strong SASP pre-load. NF-κB significantly elevated above surface baseline. IL-17 loop sustains chronic inflammation. Senolytic treatment strongly recommended before implantation.")

# ═══════════════════════════════════════════════════════════════════
# SURFACE SCORES TABLE
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Surface Biological Scores")
st.caption("🟢 favorable | 🟡 borderline | 🔴 unfavorable for biocompatibility")

ref_notes = {
    "mechano_input_effective":  "Low (<0.3) → soft/M2 | High (>0.7) → stiff/M1",
    "nanotopo_modifier":        "High (>0.7) → strong immunomodulation",
    "damp_signal":              "Low (<0.2) → minimal danger signal",
    "protein_adsorption_score": "Low (<0.2) → antifouling",
    "wettability_factor":       "High (>0.6) → hydrophilic, M2-promoting",
    "mcsf_bias":                "High → M-CSF differentiation bias",
}

def score_band(key, val):
    if key not in SCORE_REFS: return str(round(val,3))
    ref = SCORE_REFS[key]; lo,hi,inv = ref["low"],ref["high"],ref["invert"]
    icon = ("🟢" if val<lo else ("🟡" if val<hi else "🔴")) if inv \
           else ("🔴" if val<lo else ("🟡" if val<hi else "🟢"))
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
table_rows = []
for label, key, has_ref in score_keys:
    val_a = sc.get(key, "—")
    disp_a = score_band(key, val_a) if has_ref and isinstance(val_a,float) \
             else str(round(val_a,3)) if isinstance(val_a,float) else str(val_a)
    row = {"Parameter": label, "Surface A": disp_a}
    if key in ref_notes: row["Reference"] = ref_notes[key]
    if compare_mode and R2:
        val_b = R2["surface_scores"].get(key,"—")
        row["Surface B"] = score_band(key,val_b) if has_ref and isinstance(val_b,float) \
                           else str(round(val_b,3)) if isinstance(val_b,float) else str(val_b)
    table_rows.append(row)
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Export")
ecol1, ecol2 = st.columns(2)
with ecol1:
    st.download_button("⬇ Download time-series CSV",
        data=build_csv(R),
        file_name=f"fbr_results_{mat_label[:25].replace(' ','_')}_age{int(patient_age)}.csv",
        mime="text/csv")
with ecol2:
    st.download_button("⬇ Download simulation report (.txt)",
        data=build_report(R, sc, recs, mat_label, mode),
        file_name=f"fbr_report_{mat_label[:25].replace(' ','_')}_age{int(patient_age)}.txt",
        mime="text/plain")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "✅ **Validated** — Hotchkiss et al. 2016 (Acta Biomaterialia) 6/6 checks · "
    "J. Nanobiotechnology 2023 2/2 checks · "
    "Senescence: Tuttle 2020 · Chung 2020 (Sci Transl Med) · Mrozik 2024 · Yang 2023 | "
    "Signaling: Bhatt 2021 (Piezo1) · Hoffmann 2002 (NF-κB) · Zanconato 2016 (YAP)"
)
