"""
body_map.py — Interactive Human Body SVG Component
====================================================
Renders a clickable anatomical figure using SVG hotspots.
Each body region maps to tissue context parameters and
displays region-specific biological information.

Returns the selected region name for use in the main dashboard.
"""

import streamlit as st
import streamlit.components.v1 as components

# ── Region definitions ─────────────────────────────────────────────
# Each region: tissue_context, E_kPa_typical, description, FBR_relevance
BODY_REGIONS = {
    "Brain": {
        "tissue_context": "brain",
        "E_kPa_range": "0.1–1 kPa",
        "E_kPa_default": 0.5,
        "color": "#4FC3F7",
        "description": "Neural tissue — ultra-soft, highly immune-regulated",
        "implants": "Neural probes, DBS electrodes, cochlear implants",
        "fbr_concern": "Neuroinflammation, glial scarring, signal drift",
        "key_cells": "Microglia (brain-resident macrophages), astrocytes",
        "stiffness_note": "Most mechanosensitive context — 1 kPa implant feels 1000x stiffer than tissue",
        "icon": "🧠",
    },
    "Spine": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "1–10 kPa",
        "E_kPa_default": 5.0,
        "color": "#4DB6AC",
        "description": "Epidural/intrathecal space — soft connective tissue",
        "implants": "Spinal cord stimulators (Medtronic), intrathecal drug pumps",
        "fbr_concern": "Electrode encapsulation, stimulation threshold drift",
        "key_cells": "Macrophages, fibroblasts, mast cells",
        "stiffness_note": "Medtronic SCS drift is driven by progressive fibrotic encapsulation",
        "icon": "🦴",
    },
    "Heart": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "10–20 kPa",
        "E_kPa_default": 15.0,
        "color": "#EF5350",
        "description": "Cardiac tissue — mechanically dynamic, high vascularity",
        "implants": "Pacemaker leads, TAVR valves, ventricular assist devices",
        "fbr_concern": "Thrombosis, peel-off, calcification, fibrotic encapsulation",
        "key_cells": "Macrophages, cardiac fibroblasts, smooth muscle cells",
        "stiffness_note": "Cyclic mechanical loading amplifies mechanosensing signals",
        "icon": "❤️",
    },
    "Breast": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "0.1–5 kPa",
        "E_kPa_default": 0.3,
        "color": "#F48FB1",
        "description": "Adipose/glandular soft tissue — minimal mechanical load",
        "implants": "Silicone implants, tissue expanders",
        "fbr_concern": "Capsular contracture, ALCL (anaplastic large cell lymphoma)",
        "key_cells": "Macrophages, myofibroblasts, giant cells",
        "stiffness_note": "Capsular contracture driven by M1→myofibroblast transition",
        "icon": "⭕",
    },
    "Shoulder / Joint": {
        "tissue_context": "cartilage",
        "E_kPa_range": "20–50 kPa",
        "E_kPa_default": 30.0,
        "color": "#7986CB",
        "description": "Synovial joint — cartilage, bone interface",
        "implants": "Shoulder arthroplasty, rotator cuff anchors",
        "fbr_concern": "Wear debris, synovitis, periprosthetic loosening",
        "key_cells": "Synoviocytes, macrophages, osteoclasts",
        "stiffness_note": "Cartilage context — intermediate stiffness sensitivity",
        "icon": "🔩",
    },
    "Arm (subcutaneous)": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "1–5 kPa",
        "E_kPa_default": 2.0,
        "color": "#81C784",
        "description": "Subcutaneous tissue — low stiffness, well-vascularized",
        "implants": "CGM sensors, drug pumps, RFID implants, contraceptive rods",
        "fbr_concern": "Sensor drift, biofouling, protein corona formation",
        "key_cells": "Macrophages, adipocytes, fibroblasts",
        "stiffness_note": "Primary site for continuous glucose monitoring — sensor biofouling critical",
        "icon": "💉",
    },
    "Abdomen": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "1–8 kPa",
        "E_kPa_default": 4.0,
        "color": "#FFB74D",
        "description": "Peritoneal/abdominal — organ interfaces, mesh implants",
        "implants": "Hernia mesh, gastric bands, insulin pumps, peritoneal dialysis",
        "fbr_concern": "Mesh adhesion, peritoneal fibrosis, chronic pain",
        "key_cells": "Peritoneal macrophages, mesothelial cells, fibroblasts",
        "stiffness_note": "Hernia mesh stiffness mismatch with peritoneum drives chronic inflammation",
        "icon": "🫀",
    },
    "Hip / Pelvis": {
        "tissue_context": "bone",
        "E_kPa_range": "50–200 kPa",
        "E_kPa_default": 150.0,
        "color": "#A1887F",
        "description": "Bone-implant interface — high stiffness, osseointegration",
        "implants": "Total hip arthroplasty, acetabular cups, femoral stems",
        "fbr_concern": "Aseptic loosening, wear debris osteolysis, FBGC at interface",
        "key_cells": "Osteoclasts, macrophages, osteoblasts, giant cells",
        "stiffness_note": "Bone context — Piezo1 less sensitive; osseointegration depends on surface chemistry",
        "icon": "🦴",
    },
    "Knee": {
        "tissue_context": "cartilage",
        "E_kPa_range": "20–100 kPa",
        "E_kPa_default": 50.0,
        "color": "#90A4AE",
        "description": "Knee joint — cartilage, subchondral bone, synovium",
        "implants": "Total knee arthroplasty (Stryker Mako), ACL anchors, scaffolds",
        "fbr_concern": "Polyethylene wear particles, synovitis, implant loosening",
        "key_cells": "Chondrocytes, synoviocytes, macrophages",
        "stiffness_note": "Stryker Mako plans in bone-space — soft tissue FBR prediction is the gap",
        "icon": "🦿",
    },
    "Leg (subcutaneous)": {
        "tissue_context": "soft_tissue",
        "E_kPa_range": "1–5 kPa",
        "E_kPa_default": 2.0,
        "color": "#80CBC4",
        "description": "Subcutaneous leg — peripheral sensor and drug delivery site",
        "implants": "CGM sensors, peripheral nerve stimulators",
        "fbr_concern": "Biofouling, mechanical irritation, sensor drift",
        "key_cells": "Macrophages, fibroblasts, adipocytes",
        "stiffness_note": "Similar to arm subcutaneous — protein corona formation primary concern",
        "icon": "🦵",
    },
    "Dental / Jaw": {
        "tissue_context": "bone",
        "E_kPa_range": "100–200 kPa",
        "E_kPa_default": 200.0,
        "color": "#FFF176",
        "description": "Alveolar bone — titanium dental implant primary site",
        "implants": "Ti-SLA dental implants, zirconia abutments, PEEK spacers",
        "fbr_concern": "Peri-implantitis, osseointegration failure, FBGC at crestal bone",
        "key_cells": "Osteoblasts, osteoclasts, macrophages, gingival fibroblasts",
        "stiffness_note": "Hotchkiss 2016 validation dataset — Ti-SLA vs modSLA",
        "icon": "🦷",
    },
}

# SVG hotspot coordinates (cx, cy, rx, ry) — normalized to 300x550 viewBox
HOTSPOTS = {
    "Brain":             (150, 45,  38, 32),
    "Dental / Jaw":      (150, 90,  22, 15),
    "Spine":             (150, 200, 14, 60),
    "Heart":             (135, 190, 22, 20),
    "Breast":            (150, 205, 35, 22),
    "Abdomen":           (150, 265, 38, 35),
    "Shoulder / Joint":  (95,  170, 20, 20),
    "Hip / Pelvis":      (150, 315, 40, 25),
    "Arm (subcutaneous)":(78,  230, 16, 40),
    "Knee":              (150, 400, 35, 22),
    "Leg (subcutaneous)":(150, 455, 30, 30),
}


def render_body_selector(selected_region: str = "Knee") -> str:
    """
    Renders an interactive SVG body figure with clickable hotspots.
    Returns the name of the clicked region.
    Uses streamlit-components for HTML/JS interactivity.
    """

    # Build SVG hotspot elements
    hotspot_svg = ""
    hotspot_js_map = {}

    for region, (cx, cy, rx, ry) in HOTSPOTS.items():
        info = BODY_REGIONS[region]
        is_selected = (region == selected_region)
        fill_color  = info["color"]
        stroke_w    = 3 if is_selected else 1
        stroke_col  = "#FFFFFF" if is_selected else "rgba(255,255,255,0.4)"
        opacity     = 0.85 if is_selected else 0.5
        pulse_cls   = "pulse" if is_selected else ""
        safe_id     = region.replace(" ", "_").replace("/", "_")

        hotspot_svg += f'''
        <ellipse
            id="hs_{safe_id}"
            cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"
            fill="{fill_color}" fill-opacity="{opacity}"
            stroke="{stroke_col}" stroke-width="{stroke_w}"
            class="hotspot {pulse_cls}"
            onclick="selectRegion('{region}')"
            style="cursor:pointer; transition: all 0.2s ease;"
        />
        <text
            x="{cx}" y="{cy + 5}"
            text-anchor="middle" font-size="9"
            fill="white" font-weight="bold"
            pointer-events="none"
            style="text-shadow: 1px 1px 2px black;"
        >{info["icon"]}</text>
        '''

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0; padding: 0;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  svg {{
    filter: drop-shadow(0 0 12px rgba(79,195,247,0.3));
  }}
  .hotspot:hover {{
    fill-opacity: 1.0 !important;
    filter: drop-shadow(0 0 8px currentColor);
    transform-origin: center;
  }}
  @keyframes pulse {{
    0%   {{ stroke-opacity: 1.0; stroke-width: 3; }}
    50%  {{ stroke-opacity: 0.4; stroke-width: 5; }}
    100% {{ stroke-opacity: 1.0; stroke-width: 3; }}
  }}
  .pulse {{
    animation: pulse 2s infinite;
  }}
  .label {{
    font-size: 11px;
    color: #90CAF9;
    text-align: center;
    margin-top: 2px;
  }}
  .selected-label {{
    font-size: 13px;
    color: #4FC3F7;
    font-weight: bold;
    text-align: center;
    padding: 4px 12px;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 20px;
    margin-top: 4px;
  }}
</style>
</head>
<body>
<div class="container">

<svg viewBox="0 0 300 550" width="240" height="440"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <radialGradient id="bodyGrad" cx="50%" cy="40%" r="60%">
      <stop offset="0%" style="stop-color:#1a4a7a;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#0a1929;stop-opacity:1"/>
    </radialGradient>
  </defs>

  <!-- Body silhouette — anatomically proportioned -->
  <!-- Head -->
  <ellipse cx="150" cy="50" rx="38" ry="42"
           fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1.5" opacity="0.9"/>
  <!-- Neck -->
  <rect x="136" y="88" width="28" height="25" rx="4"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1"/>
  <!-- Torso -->
  <path d="M 90 110 Q 80 130 82 200 L 82 320 Q 82 330 88 330
           L 212 330 Q 218 330 218 320 L 218 200 Q 220 130 210 110
           Q 190 100 150 100 Q 110 100 90 110 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1.5" opacity="0.9"/>
  <!-- Left upper arm -->
  <path d="M 90 115 Q 68 120 60 145 L 55 210 Q 54 220 62 222
           L 78 218 Q 85 216 86 206 L 90 150 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Left forearm -->
  <path d="M 58 218 Q 50 225 46 260 L 44 300 Q 43 308 50 310
           L 64 308 Q 70 306 71 298 L 72 262 Q 73 240 78 220 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Left hand -->
  <ellipse cx="54" cy="318" rx="12" ry="16"
           fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right upper arm -->
  <path d="M 210 115 Q 232 120 240 145 L 245 210 Q 246 220 238 222
           L 222 218 Q 215 216 214 206 L 210 150 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right forearm -->
  <path d="M 242 218 Q 250 225 254 260 L 256 300 Q 257 308 250 310
           L 236 308 Q 230 306 229 298 L 228 262 Q 227 240 222 220 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right hand -->
  <ellipse cx="246" cy="318" rx="12" ry="16"
           fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Left thigh -->
  <path d="M 88 328 Q 82 335 84 380 L 86 420 Q 87 430 96 432
           L 130 432 Q 138 430 138 420 L 136 380 Q 136 340 140 328 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Left shin -->
  <path d="M 88 428 Q 84 440 86 490 L 88 520 Q 89 528 98 530
           L 122 530 Q 130 528 130 520 L 130 490 Q 130 455 132 432 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Left foot -->
  <ellipse cx="106" cy="536" rx="22" ry="10"
           fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right thigh -->
  <path d="M 212 328 Q 218 335 216 380 L 214 420 Q 213 430 204 432
           L 170 432 Q 162 430 162 420 L 164 380 Q 164 340 160 328 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right shin -->
  <path d="M 212 428 Q 216 440 214 490 L 212 520 Q 211 528 202 530
           L 178 530 Q 170 528 170 520 L 170 490 Q 170 455 168 432 Z"
        fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>
  <!-- Right foot -->
  <ellipse cx="194" cy="536" rx="22" ry="10"
           fill="url(#bodyGrad)" stroke="#1E88E5" stroke-width="1" opacity="0.85"/>

  <!-- Subtle body glow outline -->
  <ellipse cx="150" cy="50"  rx="39" ry="43" fill="none"
           stroke="rgba(79,195,247,0.2)" stroke-width="4"/>

  <!-- Hotspots -->
  {hotspot_svg}
</svg>

<div class="selected-label" id="selectedLabel">
  {BODY_REGIONS[selected_region]["icon"]} {selected_region}
</div>
<div class="label">Click a region to select</div>

</div>

<script>
function selectRegion(name) {{
  // Send message to Streamlit
  window.parent.postMessage({{
    type: "streamlit:setComponentValue",
    value: name
  }}, "*");

  document.getElementById("selectedLabel").textContent =
    name;
}}
</script>
</body>
</html>
"""
    return html


def body_region_panel(region: str):
    """Renders the info panel for a selected body region."""
    if region not in BODY_REGIONS:
        return
    info = BODY_REGIONS[region]

    st.markdown(f"### {info['icon']} {region}")
    st.caption(info["description"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Typical stiffness", info["E_kPa_range"])
    with col2:
        st.metric("Tissue context", info["tissue_context"].replace("_"," ").title())

    with st.expander("🔬 Implant types & FBR concerns", expanded=True):
        st.markdown(f"**Common implants:** {info['implants']}")
        st.markdown(f"**FBR concerns:** {info['fbr_concern']}")
        st.markdown(f"**Key cell types:** {info['key_cells']}")
        st.info(f"⚙️ {info['stiffness_note']}")