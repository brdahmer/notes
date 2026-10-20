import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

FILES = [
    ("wt-acylation-pmf.log",  "WT Acylation"),
    ("c-acylation-pmf.log",    "CHIM Acylation"),
    ("wt-deacylation-pmf.log", "WT Deacylation"),
    ("c-deacylation-pmf.log",  "CHIM Deacylation"),
]

XLABELS = {"Acylation": "Forming C–O bond (Å)", "Deacylation": "Forming C–O bond (Å)"}
YLABELS = {"Acylation": "Breaking C–O bond (Å)", "Deacylation": "Breaking C–O bond (Å)"}


def load(path):
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "pmf"],
                       on_bad_lines="skip")
    # Drop inf and non-numeric — these are unexplored configurations
    data["pmf"] = pd.to_numeric(data["pmf"], errors="coerce")
    data = data[np.isfinite(data["pmf"])]
    data["pmf"] /= 4.184                # kJ/mol → kcal/mol
    return data


def build_grid(data):
    x = np.sort(data["x"].unique())
    y = np.sort(data["y"].unique())
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, np.nan)        # unexplored regions stay NaN
    xi_idx = {v: i for i, v in enumerate(x)}
    yi_idx = {v: i for i, v in enumerate(y)}
    for _, row in data.iterrows():
        Z[yi_idx[row["y"]], xi_idx[row["x"]]] = row["pmf"]
    Z -= np.nanmin(Z)
    # Smooth only over sampled cells
    mask = np.isfinite(Z)
    Zf = np.where(mask, Z, 0.0)
    Zs = gaussian_filter(Zf, sigma=1.5)
    ws = gaussian_filter(mask.astype(float), sigma=1.5)
    Z_smooth = np.where(ws > 0.01, Zs / ws, np.nan)
    Z_smooth -= np.nanmin(Z_smooth)
    return X, Y, Z_smooth


fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for ax, (path, title) in zip(axes.flat, FILES):
    step = "Acylation" if "Acylation" in title else "Deacylation"
    data = load(path)
    X, Y, Z = build_grid(data)
    Z_masked = np.ma.masked_invalid(Z)   # NaN → not plotted
    cf = ax.contourf(X, Y, Z_masked, levels=60, cmap="RdYlBu_r")
    ax.contour(X, Y, Z_masked, levels=15, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, label=r"$\Delta G$ (kcal mol$^{-1}$)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(XLABELS[step], fontsize=10)
    ax.set_ylabel(YLABELS[step], fontsize=10)
    ax.tick_params(labelsize=8)

fig.savefig("pmf_all.png", dpi=600, bbox_inches="tight")
print("Done.")
