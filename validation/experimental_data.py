"""
validation/experimental_data.py
=================================
Curated experimental datasets for model validation.
 
PRIMARY SOURCE: Hotchkiss et al. 2016 (Acta Biomaterialia)
  "Titanium surface characteristics, including topography and wettability,
   alter macrophage activation"
  DOI: 10.1016/j.actabio.2015.09.03
 
  Surfaces tested on primary human macrophages, 24h and 72h timepoints.
  Seven surfaces: PT (smooth hydrophobic), plasmaPT (smooth hydrophilic),
  SLA (rough hydrophobic), plasmaSLA (rough hydrophilic),
  aged modSLA (nano+micro rough hydrophobic), modSLA (nano+micro hydrophilic)
 
SURFACE PARAMETERS mapped to our model inputs:
  PT (smooth, hydrophobic):
    Ra ~0.2 µm, contact_angle ~80°, feature_size ~5000 nm (no nano), E~110 GPa (Ti bulk)
  SLA (rough, hydrophobic):
    Ra ~3.2 µm, contact_angle ~130°, feature_size ~2000 nm, E~110 GPa
  modSLA (nano+micro rough, hydrophilic):
    Ra ~3.2 µm, contact_angle ~0° (<5°), feature_size ~100 nm, E~110 GPa
 
EXPERIMENTAL OUTCOME (relative, normalized):
  Direction of change confirmed across multiple papers:
 
  Inflammatory score proxy (M1-like):
    PT (smooth/hydrophobic):   HIGH inflammatory  (IL-1β, IL-6, TNFα elevated)
    SLA (rough/hydrophobic):   HIGHEST inflammatory (rough + hydrophobic = worst)
    modSLA (rough/hydrophilic): LOW inflammatory   (hydrophilicity dominates)
 
  Healing score proxy (M2-like):
    PT:     LOW healing    (minimal IL-4, IL-10)
    SLA:    LOW healing
    modSLA: HIGH healing   (IL-4 and IL-10 significantly elevated)
 
  M1/M2 ratio (Lee et al. 2021, in vivo rat calvarium):
    Day 1:  SLA > modSLA (SLA has higher M1/M2)
    Day 4:  SLA >> modSLA (significant difference)
    Day 7:  SLA > modSLA (difference narrows but persists)
 
  Nanotube size effect (Journal of Nanobiotechnology 2023):
    NT20 (20nm tubes): M1 dominant (CD86+ high, CD206+ low)
    NT5  (5nm tubes):  M2c dominant (CD163+, CD206+ high)
 
VALIDATION APPROACH:
  We validate that the model correctly predicts the RANK ORDER
  and DIRECTION of inflammatory vs healing scores across these surfaces.
  We do not fit to absolute cytokine concentrations (pg/mL) because
  our model outputs normalized scores [0,1], not cytokine levels.
  This is a directional/ordinal validation — scientifically appropriate
  for a first-generation mechanistic model.
"""
 
# Surface parameter mappings (our model inputs for each experimental condition)
# Ti elastic modulus is ~110 GPa — saturates Piezo1 well above 1 kPa threshold
# We cap at 200 kPa for model purposes (Piezo1 is saturated above ~100 kPa)
HOTCHKISS_SURFACES = {
    "PT (smooth, hydrophobic)": {
        "E_kPa": 200.0,        # Ti — saturates Piezo1 (stiff)
        "Ra_um": 0.2,           # smooth polished
        "feature_size_nm": 5000.0,  # no nano features
        "contact_angle_deg": 80.0,  # hydrophobic
        "tissue_context": "bone",
    },
    "SLA (rough, hydrophobic)": {
        "E_kPa": 200.0,
        "Ra_um": 3.2,
        "feature_size_nm": 2000.0,
        "contact_angle_deg": 130.0,  # strongly hydrophobic
        "tissue_context": "bone",
    },
    "modSLA (rough, hydrophilic)": {
        "E_kPa": 200.0,
        "Ra_um": 3.2,
        "feature_size_nm": 100.0,   # nanofeatures present
        "contact_angle_deg": 4.0,   # superhydrophilic (<5°)
        "tissue_context": "bone",
    },
}
 
# Expected rank order from literature (1=lowest, 3=highest)
EXPECTED_RANKS = {
    "inflammatory": {
        "PT (smooth, hydrophobic)":   2,   # moderate M1
        "SLA (rough, hydrophobic)":   3,   # highest M1
        "modSLA (rough, hydrophilic)":1,   # lowest M1
    },
    "healing": {
        "PT (smooth, hydrophobic)":   2,   # low-moderate M2
        "SLA (rough, hydrophobic)":   1,   # lowest M2
        "modSLA (rough, hydrophilic)":3,   # highest M2 (IL-4, IL-10 elevated)
    },
}
 
# Nanotube dataset (Journal of Nanobiotechnology 2023)
NANOTUBE_SURFACES = {
    "NT20 (20nm, M1-inducing)": {
        "E_kPa": 200.0,
        "Ra_um": 0.3,
        "feature_size_nm": 20.0,   # small tubes — M1 per literature
        "contact_angle_deg": 60.0,
        "tissue_context": "bone",
    },
    "NT5 (5nm, M2c-inducing)": {
        "E_kPa": 200.0,
        "Ra_um": 0.3,
        "feature_size_nm": 5.0,    # very small tubes — M2c per literature
        "contact_angle_deg": 60.0,
        "tissue_context": "bone",
    },
}
 
# NOTE on NT20 vs NT5 paradox:
# The literature shows 20nm nanotubes → M1 and 5nm → M2.
# Our nanotopo_modifier gives SMALLER features MORE immunomodulatory effect.
# 5nm < 20nm, so 5nm has higher nanotopo_modifier → lower mechano_input_effective
# → lower YAP, lower inflammatory. This matches the M2c finding for NT5. PASS.
# For NT20: larger features → lower nanotopo_modifier → higher mechano_input
# → higher YAP, higher inflammatory. This matches M1 for NT20. PASS.