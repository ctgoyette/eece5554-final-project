#!/usr/bin/env python3
"""
UWB Trilateration — Expected Position Error Heatmap
=====================================================
Room      : 8 m (X) x 4 m (Y)
Anchors   : id0=(0,0), id1=(0,1.372m), id2=(1.372m,0)  [4.5 ft converted to metres]
Ranging σ : 10 cm
Method    : GDOP (geometric dilution of precision) from Fisher information matrix
Output    : uwb_error_heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

# ── Constants ────────────────────────────────────────────────────────────────
ROOM_W  = 8.0       # m
ROOM_H  = 4.0       # m
SIGMA_R = 0.10      # m  (10 cm UWB ranging noise)
FT_TO_M = 0.3048

# Anchor positions converted from the spreadsheet (ft → m)
# id0 = (0 ft, 0 ft), id1 = (0 ft, 4.5 ft), id2 = (4.5 ft, 0 ft)
ANCHORS = {
    "id0": np.array([0.0       * FT_TO_M, 0.0       * FT_TO_M]),
    "id1": np.array([0.0       * FT_TO_M, 4.5       * FT_TO_M]),
    "id2": np.array([4.5       * FT_TO_M, 0.0       * FT_TO_M]),
}

# Grid resolution
NX, NY = 300, 150

# Cap displayed error for colour scale readability (cm)
ERROR_CAP_CM = 80.0

# ── GDOP calculation ─────────────────────────────────────────────────────────

def compute_gdop_field(anchors: np.ndarray, nx: int, ny: int):
    """
    Vectorised 2-D GDOP over the full room.

    Parameters
    ----------
    anchors : (n, 2) array of anchor (x, y) positions in metres
    nx, ny  : grid resolution

    Returns
    -------
    XX, YY  : (ny, nx) coordinate meshgrids
    GDOP    : (ny, nx) GDOP values (nan where geometry is degenerate)
    """
    xs = np.linspace(0.01, ROOM_W - 0.01, nx)
    ys = np.linspace(0.01, ROOM_H - 0.01, ny)
    XX, YY = np.meshgrid(xs, ys)

    # Direction unit-vectors from every grid point to every anchor
    # Shape: (ny, nx, n_anchors)
    dx = XX[..., np.newaxis] - anchors[:, 0]
    dy = YY[..., np.newaxis] - anchors[:, 1]
    r  = np.hypot(dx, dy)

    hx = dx / r
    hy = dy / r

    # Fisher information matrix elements  F = H^T H
    F11 = np.sum(hx ** 2,    axis=-1)
    F12 = np.sum(hx * hy,    axis=-1)
    F22 = np.sum(hy ** 2,    axis=-1)

    det = F11 * F22 - F12 ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        gdop = np.where(det > 1e-9,
                        np.sqrt(np.abs((F11 + F22) / det)),
                        np.nan)

    return XX, YY, gdop


# ── Build colour map (green → yellow → red) ──────────────────────────────────

GYR_CMAP = LinearSegmentedColormap.from_list(
    "gyr",
    [(0.00, "#2ecc40"),   # green  — low error
     (0.40, "#ffdc00"),   # yellow — moderate
     (0.70, "#ff851b"),   # orange
     (1.00, "#ff4136")],  # red    — high error
)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    anc_array = np.array(list(ANCHORS.values()))  # (3, 2)

    XX, YY, G = compute_gdop_field(anc_array, NX, NY)

    # Convert GDOP → expected position error in cm
    ERR_cm = SIGMA_R * G * 100.0

    # Statistics
    valid      = ERR_cm[np.isfinite(ERR_cm)]
    min_err    = float(np.nanmin(ERR_cm))
    max_err    = float(np.nanmax(ERR_cm))
    mean_err   = float(np.nanmean(ERR_cm))
    best_idx   = np.unravel_index(np.nanargmin(ERR_cm), ERR_cm.shape)
    best_x     = XX[best_idx]
    best_y     = YY[best_idx]

    # Clip for display (extreme far-corner values swamp the colour scale)
    ERR_plot = np.clip(ERR_cm, 0, ERROR_CAP_CM)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))

    img = ax.pcolormesh(XX, YY, ERR_plot,
                        cmap=GYR_CMAP,
                        vmin=min_err,
                        vmax=ERROR_CAP_CM,
                        shading="auto",
                        zorder=1)

    # Colour bar
    cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("Expected position error (cm)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(np.arange(0, ERROR_CAP_CM + 1, 10))
    cbar.ax.set_yticklabels(
        [f"{int(v)}" if v < ERROR_CAP_CM else f"≥{int(v)}"
         for v in np.arange(0, ERROR_CAP_CM + 1, 10)]
    )

    # Error iso-contours
    try:
        cs = ax.contour(XX, YY, ERR_cm,
                        levels=[10, 20, 30, 50, 70],
                        colors="white", linewidths=0.8, alpha=0.6, zorder=3)
        ax.clabel(cs, fmt="%g cm", fontsize=8, inline=True)
    except Exception:
        pass

    # Room border
    ax.add_patch(mpatches.Rectangle(
        (0, 0), ROOM_W, ROOM_H,
        lw=2, ec="black", fc="none", zorder=5))

    # Anchors
    anchor_colors = {"id0": "#1565c0", "id1": "#1976d2", "id2": "#1e88e5"}
    for name, pos in ANCHORS.items():
        ax.scatter(*pos, s=220, color=anchor_colors[name],
                   edgecolors="black", linewidths=1.2,
                   marker="^", zorder=6, label=f"{name}  ({pos[0]:.2f}, {pos[1]:.2f}) m")
        ax.annotate(name,
                    xy=pos, xytext=(pos[0] + 0.12, pos[1] + 0.12),
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="navy", alpha=0.75),
                    zorder=7)

    # Best-position star
    ax.scatter(best_x, best_y, s=300, marker="*",
               color="white", edgecolors="black", linewidths=0.8,
               zorder=7, label=f"Best position  ({best_x:.2f}, {best_y:.2f}) m  —  {min_err:.1f} cm")

    # Stats text box
    stats_txt = (
        f"Min error : {min_err:.1f} cm  at ({best_x:.2f}, {best_y:.2f}) m\n"
        f"Mean error: {mean_err:.1f} cm\n"
        f"Max error : {max_err:.0f} cm  (clipped at {ERROR_CAP_CM:.0f} cm for display)\n"
        f"σ_range = {SIGMA_R*100:.0f} cm  ·  anchors = {len(ANCHORS)}"
    )
    ax.text(0.015, 0.975, stats_txt,
            transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88),
            zorder=8)

    # Axes formatting
    ax.set_xlim(-0.2, ROOM_W + 0.2)
    ax.set_ylim(-0.2, ROOM_H + 0.2)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        "UWB Trilateration — Expected Position Error\n"
        "Room 8×4 m  ·  3 anchors at measured coordinates",
        fontsize=13, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.2, zorder=2)
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    out_path = "uwb_error_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved  {out_path}")
    plt.show()


if __name__ == "__main__":
    main()