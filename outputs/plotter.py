"""
outputs/plotter.py  (v3)
=========================
NEW IN V3:
  - Shows immunomodulation index prominently on phenotype panel
  - Shows scaffold target trajectory overlay when mode='scaffold'
  - Shows drug release profile on signaling panel when active
  - Tissue context and coating shown in figure title
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from phenotype.phenotype_classifier import _scaffold_target
 
 
def plot_results(results, save_path=None):
    t   = results.t_days
    ps  = results.phenotype_scores
    fb  = results.fbgc_risk
    s   = results.surface
    mode = results.mode
 
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
 
    coating_str = f"{s.coating}" if s.coating != "none" else "uncoated"
    drug_str    = " | drug-eluting" if s.drug_release_profile else ""
    fig.suptitle(
        f"FBR Mechanotransduction Model  [{mode.upper()} mode]\n"
        f"E={s.E_kPa} kPa | Ra={s.Ra_um} µm | {s.tissue_context} | "
        f"{coating_str}{drug_str}",
        fontsize=12, fontweight='bold', y=0.99
    )
 
    # ── Panel 1: Signaling ────────────────────────────────────────
    ax1   = axes[0]
    ax1_r = ax1.twinx()
    ax1_r.plot(t, results.ca_timeseries, color='#7B9CDA', lw=1.5,
               ls='--', alpha=0.7, label='[Ca²⁺]ᵢ (µM)')
    ax1.plot(t, results.yap_nuclear,  color='#E8593C', lw=2,     label='YAP nuclear')
    ax1.plot(t, results.nfkb_active,  color='#D85A30', lw=2, ls='-.', label='NF-κB')
    ax1.plot(t, results.stat6_active, color='#1D9E75', lw=2,     label='STAT6')
 
    # NEW v3: drug release overlay
    if s.drug_release_profile is not None:
        drug_ts = np.array([s.drug_release_profile(ti*24*60) for ti in t])
        ax1.plot(t, drug_ts, color='#9B59B6', lw=1.5, ls=':',
                 alpha=0.8, label='Drug suppression')
 
    ax1.set_ylabel("Signaling fraction [0–1]", fontsize=10)
    ax1_r.set_ylabel("[Ca²⁺]ᵢ (µM)", fontsize=10, color='#7B9CDA')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left',  fontsize=8, framealpha=0.7)
    ax1_r.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax1.set_title("Intracellular signaling dynamics", fontsize=11)
    ax1.grid(True, alpha=0.3)
    for day, lbl in [(7,"d7: IL-4\nonset"), (14,"d14: peak")]:
        ax1.axvline(day, color='gray', ls=':', alpha=0.5)
        ax1.text(day+0.2, 0.93, lbl, fontsize=7, color='gray')
 
    # ── Panel 2: Phenotype spectrum ───────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(t, 0, ps.inflammatory, alpha=0.25, color='#E24B4A')
    ax2.fill_between(t, 0, ps.transitional,  alpha=0.25, color='#EF9F27')
    ax2.fill_between(t, 0, ps.healing,       alpha=0.25, color='#1D9E75')
    ax2.fill_between(t, 0, ps.fbgc_prone,    alpha=0.25, color='#7F77DD')
    ax2.plot(t, ps.inflammatory, color='#E24B4A', lw=1.8, label='Inflammatory')
    ax2.plot(t, ps.transitional,  color='#EF9F27', lw=1.8, label='Transitional')
    ax2.plot(t, ps.healing,       color='#1D9E75', lw=1.8, label='Healing (M2a)')
    ax2.plot(t, ps.fbgc_prone,    color='#7F77DD', lw=1.8, label='FBGC-prone')
 
    # NEW v3: scaffold target overlay
    if mode == "scaffold":
        tgt = _scaffold_target(results.t_minutes)
        ax2.plot(t, tgt["inflammatory"], color='#E24B4A', lw=1, ls=(0,(5,3)),
                 alpha=0.6, label='Target inflam.')
        ax2.plot(t, tgt["transitional"],  color='#EF9F27', lw=1, ls=(0,(5,3)),
                 alpha=0.6, label='Target transit.')
        ax2.plot(t, tgt["healing"],       color='#1D9E75', lw=1, ls=(0,(5,3)),
                 alpha=0.6, label='Target healing')
 
    # NEW v3: immunomodulation index annotation
    idx = ps.immunomodulation_index
    ax2.text(0.98, 0.97, f"Immuno-mod index: {idx:.2f}",
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=9, color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#aaa', alpha=0.8))
    if mode == "scaffold":
        ax2.text(0.98, 0.84, f"Scaffold match: {ps.trajectory_match_score:.2f}",
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=9, color='#555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffde7',
                           edgecolor='#aaa', alpha=0.8))
 
    ax2.set_ylabel("Phenotype score [0–1]", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.7)
    ax2.set_title("Macrophage phenotype spectrum (continuous)", fontsize=11)
    ax2.grid(True, alpha=0.3)
 
    # ── Panel 3: FBGC risk ────────────────────────────────────────
    ax3 = axes[2]
    risk_ts = fb["risk_timeseries"]
    for lo, hi, col in [(0,.25,'green'),(0.25,.60,'gold'),
                         (0.60,.80,'orange'),(0.80,1.0,'red')]:
        ax3.axhspan(lo, hi, alpha=0.07, color=col)
    ax3.plot(t, risk_ts, color='#534AB7', lw=2.5, label='FBGC risk')
    for level, col in [(0.25,'gold'),(0.60,'orange'),(0.80,'red')]:
        ax3.axhline(level, color=col, ls='--', alpha=0.5, lw=1)
    for y_pos, lbl in [(0.12,'Low'),(0.42,'Moderate'),(0.70,'High'),(0.90,'Very High')]:
        ax3.text(t[-1]*0.98, y_pos, lbl, fontsize=8, color='gray',
                 ha='right', va='center')
    if fb['onset_day'] > 0:
        ax3.axvline(fb['onset_day'], color='#534AB7', ls=':', alpha=0.8)
        ax3.text(fb['onset_day']+0.3, 0.28,
                 f"onset d{fb['onset_day']:.0f}", fontsize=7, color='#534AB7')
    ax3.set_ylabel("FBGC risk probability", fontsize=10)
    ax3.set_xlabel("Time (days post-implantation)", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.set_title(
        f"FBGC risk | Peak: {fb['peak_risk']:.2f} ({fb['risk_category']})",
        fontsize=11)
    ax3.legend(fontsize=9, framealpha=0.7)
    ax3.grid(True, alpha=0.3)
 
    plt.tight_layout(rect=[0,0,1,0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Output] Figure saved to {save_path}")
    plt.show()
    return fig