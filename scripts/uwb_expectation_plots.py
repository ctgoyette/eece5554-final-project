#!/usr/bin/env python3
"""
UWB Trilateration GDOP Analysis
================================
Room     : 8 m x 4 m
Anchors  : constrained to top-right 1.5 x 1.5 m region
Beacons  : 0.75 x 1.5 m region - position varies per scenario
Sensors  : 7 total (any split between anchor / beacon mode)
Ranging σ: 10 cm

Outputs
-------
uwb_gdop_configurations.png  -  6-panel heatmap comparison
uwb_gdop_sweep.png           -  positional sensitivity analysis
uwb_gdop_anchor_count.png    -  anchor count vs mean error across beacon zone
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOM_W  = 8.0    # m
ROOM_H  = 4.0    # m
SIGMA_R = 0.10   # m  (10 cm UWB ranging noise)
N_TOT   = 7      # total sensors

# Anchor deployment zone - top-right corner, 1.5 x 1.5 m
AX0, AX1 = ROOM_W - 1.5, ROOM_W   # 6.5 → 8.0
AY0, AY1 = ROOM_H - 1.5, ROOM_H   # 2.5 → 4.0

# Beacon region footprint
BW, BH = 1.50, 0.75   # m

# Colour scale cap for GDOP heatmaps
GDOP_CAP = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# GDOP computation (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _fisher(anchors, XX, YY):
    """
    Return the three independent elements of the 2x2 Fisher information matrix
    F = H^T H  at every grid point (XX, YY), given anchor positions.

    Shapes: XX/YY are (ny, nx); returns F11, F12, F22 of same shape.
    """
    A  = np.array(anchors)                       # (n, 2)
    dx = XX[..., np.newaxis] - A[:, 0]           # (ny, nx, n)
    dy = YY[..., np.newaxis] - A[:, 1]
    r  = np.hypot(dx, dy)

    hx = dx / r   # unit-vector components from point to each anchor
    hy = dy / r

    F11 = np.sum(hx ** 2,      axis=-1)
    F12 = np.sum(hx * hy,      axis=-1)
    F22 = np.sum(hy ** 2,      axis=-1)
    return F11, F12, F22


def gdop_field(anchors, nx=180, ny=90):
    """
    Compute 2-D GDOP (= √trace(F⁻¹)) across the full room.
    Returns (XX, YY, GDOP) arrays of shape (ny, nx).
    """
    xs = np.linspace(0.05, ROOM_W - 0.05, nx)
    ys = np.linspace(0.05, ROOM_H - 0.05, ny)
    XX, YY = np.meshgrid(xs, ys)

    F11, F12, F22 = _fisher(anchors, XX, YY)
    det = F11 * F22 - F12 ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        trace_inv = (F11 + F22) / det
        G = np.where(det > 1e-9, np.sqrt(np.abs(trace_inv)), np.nan)

    return XX, YY, G


def zone_stats(anchors, bx0, by0, ns=24):
    """
    Mean & max *position error* (cm) sampled across the beacon zone.
    Returns (mean_cm, max_cm).
    """
    xs = np.linspace(bx0, bx0 + BW, ns)
    ys = np.linspace(by0, by0 + BH, ns * 2)
    XX, YY = np.meshgrid(xs, ys)

    F11, F12, F22 = _fisher(anchors, XX, YY)
    det = F11 * F22 - F12 ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(det > 1e-9, np.sqrt(np.abs((F11 + F22) / det)), np.nan)

    err_cm = SIGMA_R * G * 100
    valid  = err_cm[np.isfinite(err_cm)]
    return float(valid.mean()), float(valid.max())


# ─────────────────────────────────────────────────────────────────────────────
# Anchor layout presets
# ─────────────────────────────────────────────────────────────────────────────

def spread_anchors(n):
    """Maximally spread within the 1.5x1.5 m anchor region."""
    cx, cy = 0.5 * (AX0 + AX1), 0.5 * (AY0 + AY1)
    layouts = {
        3: [(AX0, AY0), (AX1, AY0), (cx, AY1)],
        4: [(AX0, AY0), (AX1, AY0), (AX0, AY1), (AX1, AY1)],
        5: [(AX0, AY0), (AX1, AY0), (AX0, AY1), (AX1, AY1), (cx, cy)],
        6: [(x, y)
            for x in np.linspace(AX0, AX1, 3)
            for y in np.linspace(AY0, AY1, 2)],
    }
    return layouts[n]


def clustered_anchors(n, jitter=0.04):
    """All anchors nearly co-located - degenerate geometry."""
    cx, cy = 0.5 * (AX0 + AX1), 0.5 * (AY0 + AY1)
    off = [(0,0),(jitter,0),(0,jitter),(-jitter,0),(0,-jitter),
           (jitter,jitter),(-jitter,jitter)]
    return [(cx + off[i][0], cy + off[i][1]) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

scenarios = [
    dict(
        title="Lowest Uncertainty",
        sub="4 spread anchors · beacons at room centre",
        note="Anchor corners maximise angular separation;\nbeacons centred minimise ranging asymmetry.",
        color="green",
        anc=spread_anchors(4), nb=3,
        bx0=3.625, by0=1.25,
    ),
    dict(
        title="Highest Uncertainty",
        sub="3 clustered anchors · beacons at far corner",
        note="Nearly co-located anchors → degenerate H matrix;\nbeacons in opposite corner compound the error.",
        color="red",
        anc=clustered_anchors(3), nb=4,
        bx0=0.0, by0=0.0,
    ),
    dict(
        title="Beacons Adjacent to Anchor Zone",
        sub="4 spread anchors · beacons 0.75 m from anchors",
        note="Short range → low noise, but all unit-vectors\npoint in nearly the same direction.",
        color="darkorange",
        anc=spread_anchors(4), nb=3,
        bx0=5.50, by0=1.25,
    ),
    dict(
        title="5 Anchors · Beacons at Far Wall",
        sub="Redundant anchor improves distant coverage",
        note="Extra sensor in anchor mode pays off most when\nbeacons are far from the anchor cluster.",
        color="purple",
        anc=spread_anchors(5), nb=2,
        bx0=0.0, by0=1.25,
    ),
    dict(
        title="3 Anchors (minimum) · Far Wall",
        sub="Minimum viable anchor count, well spread",
        note="Compare GDOP to 5-anchor case at same\nbeacon location to see redundancy benefit.",
        color="saddlebrown",
        anc=spread_anchors(3), nb=4,
        bx0=0.0, by0=1.25,
    ),
    dict(
        title="6 Anchors · 1 Beacon",
        sub="Maximum redundancy, single beacon at far wall",
        note="Diminishing returns past 4 anchors - the\nangular coverage ceiling limits improvement.",
        color="teal",
        anc=spread_anchors(6), nb=1,
        bx0=0.0, by0=1.25,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 - GDOP heatmaps for each scenario
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes = plt.subplots(3, 2, figsize=(16, 13))
axes = axes.ravel()

for ax, sc in zip(axes, scenarios):
    anc = sc['anc']
    bx0, by0 = sc['bx0'], sc['by0']

    XX, YY, G = gdop_field(anc)
    G_plot = np.clip(G, 1.0, GDOP_CAP)

    im = ax.pcolormesh(XX, YY, G_plot,
                       cmap='RdYlGn_r',
                       norm=LogNorm(vmin=1, vmax=GDOP_CAP),
                       shading='auto', zorder=1)
    cb = fig1.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label('GDOP', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # GDOP iso-contours
    try:
        cs = ax.contour(XX, YY, G, levels=[2, 4, 8, 12],
                        colors='white', linewidths=0.7, alpha=0.65, zorder=3)
        ax.clabel(cs, fmt='%.0f', fontsize=6.5, inline=True)
    except Exception:
        pass

    # Room outline
    ax.add_patch(patches.Rectangle(
        (0, 0), ROOM_W, ROOM_H,
        lw=2, ec='k', fc='none', zorder=4))

    # Anchor deployment zone
    ax.add_patch(patches.Rectangle(
        (AX0, AY0), 1.5, 1.5,
        lw=1.5, ec='royalblue', fc='lightblue',
        alpha=0.35, zorder=3, label='Anchor zone'))

    # Beacon zone
    ax.add_patch(patches.Rectangle(
        (bx0, by0), BW, BH,
        lw=2, ec='crimson', fc='lightyellow',
        alpha=0.55, zorder=3, label='Beacon zone'))

    # Anchor markers
    ax.scatter([a[0] for a in anc], [a[1] for a in anc],
               s=160, c='royalblue', marker='^',
               edgecolors='navy', lw=0.8, zorder=5,
               label=f'Anchors x{len(anc)}')

    mean_e, max_e = zone_stats(anc, bx0, by0)

    ax.set_title(f"{sc['title']}\n{sc['sub']}",
                 color=sc['color'], fontsize=9.5, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_xlim(-0.3, ROOM_W + 0.3)
    ax.set_ylim(-0.3, ROOM_H + 0.3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, zorder=2)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)

    stats_txt = (f"Anchors: {len(anc)}  ·  Beacons: {sc['nb']}\n"
                 f"Beacon zone error  -  mean: {mean_e:.1f} cm  /  max: {max_e:.1f} cm\n"
                 f"{sc['note']}")
    ax.text(0.015, 0.015, stats_txt, transform=ax.transAxes,
            fontsize=7, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.88, zorder=6))

fig1.suptitle(
    f"UWB Trilateration — GDOP Comparison\n"
    f"Room {ROOM_W:.0f}x{ROOM_H:.0f} m  ·  σ_range = {SIGMA_R*100:.0f} cm  ·  "
    f"{N_TOT} sensors total  ·  Anchors constrained to top-right 1.5x1.5 m",
    fontsize=12, fontweight='bold')
fig1.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
fig1.savefig(os.path.join(_DIR, 'uwb_gdop_configurations.png'), dpi=150, bbox_inches='tight')
print("Saved  uwb_gdop_configurations.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 - Positional sensitivity sweeps
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

colors_na = {3: 'saddlebrown', 4: 'green', 5: 'purple', 6: 'teal'}

# ── Left: horizontal sweep (beacon zone slides L↔R, vertically centred) ──────
ax = axes2[0]
by0_ctr = (ROOM_H - BH) / 2          # vertically centred
bx0_vals = np.linspace(0, ROOM_W - BW, 60)
cx_vals  = bx0_vals + BW / 2

for n_a in [3, 4, 5, 6]:
    anc   = spread_anchors(n_a)
    means = [zone_stats(anc, bx, by0_ctr)[0] for bx in bx0_vals]
    ax.plot(cx_vals, means, lw=2.2, color=colors_na[n_a],
            label=f'{n_a} anchors / {N_TOT - n_a} beacons')

ax.axvspan(AX0, AX1, alpha=0.12, color='royalblue', label='Anchor X range')
ax.set_xlabel('Beacon zone centre  X (m)', fontsize=10)
ax.set_ylabel('Mean position error (cm)', fontsize=10)
ax.set_title('Horizontal Sweep\n(beacon zone vertically centred in room)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(BW / 2, ROOM_W - BW / 2)

# ── Right: vertical sweep (beacon zone slides up/down, near left wall) ───────
ax = axes2[1]
bx0_wall = 0.10
by0_vals = np.linspace(0, ROOM_H - BH, 60)
cy_vals  = by0_vals + BH / 2

for n_a in [3, 4, 5, 6]:
    anc   = spread_anchors(n_a)
    means = [zone_stats(anc, bx0_wall, by)[0] for by in by0_vals]
    ax.plot(cy_vals, means, lw=2.2, color=colors_na[n_a],
            label=f'{n_a} anchors / {N_TOT - n_a} beacons')

ax.axhspan(AY0, AY1, alpha=0.12, color='royalblue', label='Anchor Y range')
ax.set_xlabel('Beacon zone centre  Y (m)', fontsize=10)
ax.set_ylabel('Mean position error (cm)', fontsize=10)
ax.set_title('Vertical Sweep\n(beacon zone near left wall,  x ≈ 0)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(BH / 2, ROOM_H - BH / 2)

fig2.suptitle(
    'UWB Positioning Error vs Beacon Zone Location\n'
    '(anchors maximally spread within top-right 1.5x1.5 m region)',
    fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(_DIR, 'uwb_gdop_sweep.png'), dpi=150, bbox_inches='tight')
print("Saved  uwb_gdop_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 - 2-D heatmap: error as a function of beacon-zone position
#            (one panel per anchor count, anchors maximally spread)
# ─────────────────────────────────────────────────────────────────────────────

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))
axes3 = axes3.ravel()

# Sweep beacon zone origin on a coarse grid
grid_n   = 35
bx0_grid = np.linspace(0, ROOM_W - BW, grid_n)
by0_grid = np.linspace(0, ROOM_H - BH, grid_n)
BX, BY   = np.meshgrid(bx0_grid, by0_grid)

for idx, n_a in enumerate([3, 4, 5, 6]):
    ax  = axes3[idx]
    anc = spread_anchors(n_a)

    ERR = np.zeros_like(BX)
    for i in range(grid_n):
        for j in range(grid_n):
            ERR[i, j] = zone_stats(anc, BX[i, j], BY[i, j])[0]

    # Centres of beacon zone
    CX = BX + BW / 2
    CY = BY + BH / 2

    im = ax.pcolormesh(CX, CY, ERR, cmap='plasma_r', shading='auto', zorder=1)
    cb = fig3.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label('Mean error (cm)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Anchor zone overlay
    ax.add_patch(patches.Rectangle(
        (AX0, AY0), 1.5, 1.5,
        lw=1.5, ec='white', fc='none',
        linestyle='--', zorder=3, label='Anchor zone'))

    # Mark best position
    best_idx = np.unravel_index(np.argmin(ERR), ERR.shape)
    bx_best  = BX[best_idx] + BW / 2
    by_best  = BY[best_idx] + BH / 2
    ax.scatter(bx_best, by_best, s=200, c='lime', marker='*',
               zorder=5, label=f'Best: ({bx_best:.1f}, {by_best:.1f}) m')

    ax.set_title(f'{n_a} anchors · {N_TOT - n_a} beacons\n'
                 f'Min error in zone: {ERR.min():.1f} cm  /  Max: {ERR.max():.1f} cm',
                 color=colors_na[n_a], fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Beacon zone centre  X (m)', fontsize=8)
    ax.set_ylabel('Beacon zone centre  Y (m)', fontsize=8)
    ax.set_xlim(0, ROOM_W)
    ax.set_ylim(0, ROOM_H)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, color='white', zorder=2)
    ax.legend(fontsize=7, loc='lower left', framealpha=0.85)

fig3.suptitle(
    'Mean Position Error vs Beacon Zone Placement\n'
    f'(anchors maximally spread in top-right corner  ·  σ_range = {SIGMA_R*100:.0f} cm)',
    fontsize=12, fontweight='bold')
fig3.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore
fig3.savefig(os.path.join(_DIR, 'uwb_gdop_anchor_count.png'), dpi=150, bbox_inches='tight')
print("Saved  uwb_gdop_anchor_count.png")

plt.show()
print("\nDone.")
