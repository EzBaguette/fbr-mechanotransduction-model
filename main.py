"""
main.py  (v3)
==============
FBR Mechanotransduction Model — entry point.
 
NEW IN V3 — three new CLI arguments:
  --mode        : 'suppress' (default) or 'scaffold'
  --coating     : 'none', 'peg', 'zwitterionic', 'phospholipid'
  --tissue      : 'brain', 'soft_tissue', 'cartilage', 'bone'
  --drug        : 'none', 'plga7', 'plga14', 'dual_phase', 'scaffold_guided'
 
USAGE EXAMPLES:
 
  # Standard FBR suppression (default)
  python main.py --stiffness 50 --roughness 0.5 --time 28
 
  # Neural implant context with zwitterionic coating
  python main.py --stiffness 0.5 --tissue brain --coating zwitterionic --time 28
 
  # PLGA drug-eluting implant in suppress mode
  python main.py --stiffness 50 --drug plga7 --time 28
 
  # Scaffold / guided tissue regeneration mode
  python main.py --stiffness 20 --roughness 1.0 --mode scaffold --drug scaffold_guided --time 28
 
  # Bone implant with phospholipid coating
  python main.py --stiffness 200 --tissue bone --coating phospholipid --time 28
"""
 
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
 
from inputs.surface_params import SurfaceInputs
from inputs.drug_profiles   import (plga_microsphere, dual_phase,
                                     scaffold_guided, no_drug)
from signaling.piezo1_calcium       import Piezo1CalciumModule
from signaling.yap_hippo            import YAPHippoModule
from signaling.nfkb                 import NFkBModule
from signaling.stat6                import STAT6Module
from signaling.sasp_senescence      import (SenescenceInputs,
                                             SASPSenescenceModule,
                                             sasp_to_nfkb_amplification)
from phenotype.phenotype_classifier import PhenotypeClassifier
from phenotype.fbgc_risk            import FBGCRiskModule
from outputs.results                import SimulationResults
from outputs.plotter                import plot_results
 
DRUG_OPTIONS = {
    "none":            None,
    "plga7":           lambda: plga_microsphere(half_life_days=7),
    "plga14":          lambda: plga_microsphere(half_life_days=14),
    "dual_phase":      dual_phase,
    "scaffold_guided": scaffold_guided,
}
 
 
def run_simulation(
    E_kPa:             float = 50.0,
    Ra_um:             float = 0.5,
    feature_size_nm:   float = 200.0,
    contact_angle_deg: float = 70.0,
    tissue_context:    str   = "cartilage",
    coating:           str   = "none",
    drug:              str   = "none",
    mode:              str   = "suppress",
    patient_age:       float = 45.0,   # NEW v5
    t_days:            float = 28.0,
    dt_min:            float = 1.0,
    save_outputs:      bool  = True,
    plot:              bool  = True,
) -> SimulationResults:
 
    print(f"\n{'='*62}")
    print(f"  FBR Model v5  |  mode={mode}  |  tissue={tissue_context}")
    print(f"  E={E_kPa} kPa | Ra={Ra_um} µm | age={patient_age}y | drug={drug}")
    print(f"{'='*62}\n")
 
    drug_fn = None
    if drug != "none" and drug in DRUG_OPTIONS and DRUG_OPTIONS[drug]:
        drug_fn = DRUG_OPTIONS[drug]()
 
    surface = SurfaceInputs(
        E_kPa=E_kPa, Ra_um=Ra_um, feature_size_nm=feature_size_nm,
        contact_angle_deg=contact_angle_deg, tissue_context=tissue_context,
        coating=coating, drug_release_profile=drug_fn,
    )
 
    t_minutes = np.arange(0, t_days * 24 * 60, dt_min)
 
    # ── Signaling cascade (with senescence) ───────────────────────
    ca_ts = Piezo1CalciumModule(surface).simulate(t_minutes)
 
    # Step 1: preliminary NF-kB (no SASP yet) to seed senescence ODE
    nfkb_prelim = NFkBModule(surface, ca_ts).simulate(t_minutes)
 
    # Step 2: senescence module
    sen_inputs = SenescenceInputs(
        patient_age        = patient_age,
        tissue_context     = tissue_context,
        E_kPa              = E_kPa,
        contact_angle_deg  = contact_angle_deg,
        protein_adsorption = surface.protein_adsorption_score,
    )
    print(f"[Senescence] {sen_inputs.age_label()} ({patient_age:.0f}y) | "
          f"S_age={sen_inputs.s_age:.4f} | "
          f"S_material={sen_inputs.s_material:.4f} | "
          f"S_total={sen_inputs.s_total:.4f} "
          f"({sen_inputs.risk_category()})")
 
    sasp_module  = SASPSenescenceModule(sen_inputs, nfkb_prelim, t_minutes)
    sasp_results = sasp_module.simulate(t_minutes)
 
    # Step 3: NF-kB re-run with SASP amplification
    sasp_amp = sasp_to_nfkb_amplification(
        sasp_results["sasp_drive"], sen_inputs.s_total
    )
    nfkb_ts = NFkBModule(surface, ca_ts,
                          sasp_amplification=sasp_amp).simulate(t_minutes)
 
    yap_ts   = YAPHippoModule(surface, ca_ts).simulate(t_minutes)
    stat6_ts = STAT6Module(surface, t_minutes).simulate(t_minutes)
 
    # ── Phenotype + FBGC ──────────────────────────────────────────
    classifier = PhenotypeClassifier(
        yap_nuclear=yap_ts, nfkb_active=nfkb_ts,
        stat6_active=stat6_ts, t_minutes=t_minutes,
        mcsf_bias=surface.mcsf_bias, mode=mode,
        wettability_factor=surface.wettability_factor,
    )
    phenotype_scores = classifier.compute()
 
    fbgc_risk = FBGCRiskModule(
        surface=surface, il4_signal=stat6_ts,
        nfkb_active=nfkb_ts, t_minutes=t_minutes,
    ).compute_risk()
 
    results = SimulationResults(
        surface=surface, t_minutes=t_minutes,
        ca_timeseries=ca_ts, yap_nuclear=yap_ts,
        nfkb_active=nfkb_ts, stat6_active=stat6_ts,
        phenotype_scores=phenotype_scores, fbgc_risk=fbgc_risk,
        mode=mode,
        senescence_results=sasp_results,   # NEW v5
        patient_age=patient_age,           # NEW v5
    )
    results.print_summary()
 
    if save_outputs:
        out_dir = Path("outputs_data")
        out_dir.mkdir(exist_ok=True)
        results.save_csv(out_dir / f"results_{mode}_{coating}_{drug}_age{int(patient_age)}.csv")
 
    if plot:
        plot_results(results,
                     save_path=f"outputs_data/plot_{mode}_{coating}_{drug}_age{int(patient_age)}.png")
 
    return results
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FBR Mechanotransduction Model v5")
    p.add_argument("--stiffness",     type=float, default=50.0)
    p.add_argument("--roughness",     type=float, default=0.5)
    p.add_argument("--feature_size",  type=float, default=200.0)
    p.add_argument("--contact_angle", type=float, default=70.0)
    p.add_argument("--tissue",   type=str, default="cartilage",
                   choices=["brain","soft_tissue","cartilage","bone"])
    p.add_argument("--coating",  type=str, default="none",
                   choices=["none","peg","zwitterionic","phospholipid","custom"])
    p.add_argument("--drug",     type=str, default="none",
                   choices=["none","plga7","plga14","dual_phase","scaffold_guided"])
    p.add_argument("--mode",     type=str, default="suppress",
                   choices=["suppress","scaffold"])
    p.add_argument("--age",      type=float, default=45.0,
                   help="Patient age in years (default: 45)")
    p.add_argument("--time",     type=float, default=28.0)
    p.add_argument("--dt",       type=float, default=1.0)
    p.add_argument("--no_plot",  action="store_true")
    args = p.parse_args()
 
    run_simulation(
        E_kPa=args.stiffness, Ra_um=args.roughness,
        feature_size_nm=args.feature_size,
        contact_angle_deg=args.contact_angle,
        tissue_context=args.tissue, coating=args.coating,
        drug=args.drug, mode=args.mode,
        patient_age=args.age,
        t_days=args.time, dt_min=args.dt,
        plot=not args.no_plot,
    )
