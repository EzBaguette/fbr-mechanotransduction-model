# run_sweep.py
from utils.parameter_sweep import run_sweep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = run_sweep(
    stiffness_range=[1, 5, 10, 20, 50, 100, 200],
    roughness_range=[0.05, 0.1, 0.5, 1.0, 2.0],
    contact_angle_range=[25, 50, 70, 90, 110],
    t_days=28,
    dt_min=10,       # coarser timestep keeps sweep fast
)

df.to_csv("sweep_results.csv", index=False)
print(df[["E_kPa", "Ra_um", "contact_angle_deg",
          "score_inflammatory", "score_healing",
          "fbgc_peak_risk", "dominant_phenotype"]].to_string())