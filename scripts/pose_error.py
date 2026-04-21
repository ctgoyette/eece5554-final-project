#!/usr/bin/env python3
"""
UWB Robot Pose Error Estimation
=================================
For each test pose (A–E) this script:

  1. Derives a per-anchor calibration model (linear fit: raw_cm → true_ft)
  2. Applies the calibration to the raw corner measurements
  3. Trilaterates each robot corner's world position from calibrated distances
  4. Fits a rigid-body transform (SVD) to get the estimated robot pose (x, y, θ)
  5. Computes predicted pose uncertainty via GDOP-based error propagation
     using the Fisher information matrix at each corner location
  6. Plots predicted vs actual errors for all 5 test poses

Units: metres internally; feet / degrees for display where noted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import least_squares

# ─────────────────────────────────────────────────────────────────────────────
# Constants & raw data from spreadsheet
# ─────────────────────────────────────────────────────────────────────────────

FT_TO_M  = 0.3048
IN_TO_M  = 0.0254
CM_TO_M  = 0.01
SIGMA_R  = 0.10          # UWB ranging noise, metres

ROOM_W, ROOM_H = 8.0, 4.0   # metres

# Fixed anchor positions (ft → m)
ANCHORS_FT = {
    "id0": np.array([0.0, 0.0]),
    "id1": np.array([0.0, 4.5]),
    "id2": np.array([4.5, 0.0]),
}
ANCHORS = {k: v * FT_TO_M for k, v in ANCHORS_FT.items()}
ANC = np.array([ANCHORS["id0"], ANCHORS["id1"], ANCHORS["id2"]])  # (3,2)

# Robot corner offsets in body frame (inches → m), origin = robot centre-ish
# Corner 1=(20,-7), 2=(20,7), 3=(0,7), 4=(0,-7)  inches
CORNERS_BODY_IN = np.array([
    [20, -7],
    [20,  7],
    [ 0,  7],
    [ 0, -7],
], dtype=float)
CORNERS_BODY = CORNERS_BODY_IN * IN_TO_M   # (4,2) in metres

# Calibration data: ground truth (ft) vs raw readings per anchor (cm)
CALIB_GT_FT  = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
CALIB_RAW_CM = {
    "id0": np.array([48, 111, 165, 228, 288, 356, 420, 483, 552, 650], dtype=float),
    "id1": np.array([48, 110, 174, 228, 288, 352, 427, 463, 564, 630], dtype=float),
    "id2": np.array([43, 105, 177, 225, 306, 393, 424, 491, 590, 611], dtype=float),
}

# Ground truth poses: robot origin (ft) + heading (degrees, CCW from +X)
POSES_GT = {
    "A": {"x_ft": 6.0,  "y_ft": 5.0,  "theta_deg":   0.0},
    "B": {"x_ft": 13.5, "y_ft": 7.0,  "theta_deg": 145.15},
    "C": {"x_ft": 18.5, "y_ft": 6.0,  "theta_deg": 286.6},
    "D": {"x_ft": 23.5, "y_ft": 10.0, "theta_deg": 270.0},
    "E": {"x_ft": 12.0, "y_ft": 1.0,  "theta_deg": 180.0},
}

# Raw measured distances at each corner, cm  [corner_index, anchor_index]
# columns: id0, id1, id2
RAW_MEAS_CM = {
    "A": np.array([[217, 142, 150],
                   [202, 151, 131],
                   [223, 176, 174],
                   [251, 189, 183]], dtype=float),
    "B": np.array([[523, 480, 401],
                   [487, 476, 393],
                   [502, 367, 322],
                   [472, 428, 356]], dtype=float),
    "C": np.array([[567, 581, 418],
                   [555, 578, 446],
                   [624, 611, 489],
                   [600, 617, 461]], dtype=float),
    "D": np.array([[753, 708, 635],
                   [811, 746, 703],
                   [808, 836, 705],
                   [825, 712, 648]], dtype=float),
    "E": np.array([[423, 466, 307],
                   [412, 427, 296],
                   [356, 375, 220],
                   [251, 403, 213]], dtype=float),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-anchor calibration  (linear fit: raw_cm → true_m)
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibration(anchor_name):
    """Return (slope, intercept) mapping raw_cm → true_metres."""
    raw = CALIB_RAW_CM[anchor_name]
    gt  = CALIB_GT_FT * FT_TO_M
    slope, intercept = np.polyfit(raw, gt, 1)
    return slope, intercept


CALIB = {name: fit_calibration(name) for name in ("id0", "id1", "id2")}


def calibrate(raw_cm_row):
    """Convert a (3,) raw [id0, id1, id2] cm reading to calibrated metres."""
    return np.array([
        CALIB["id0"][0] * raw_cm_row[0] + CALIB["id0"][1],
        CALIB["id1"][0] * raw_cm_row[1] + CALIB["id1"][1],
        CALIB["id2"][0] * raw_cm_row[2] + CALIB["id2"][1],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Trilateration — estimate 2-D position from 3 range measurements
# ─────────────────────────────────────────────────────────────────────────────

def trilaterate(ranges_m, anchors=ANC):
    """
    Nonlinear least-squares trilateration.
    ranges_m : (3,) calibrated distances in metres
    anchors  : (3, 2) anchor positions in metres
    Returns estimated (x, y) in metres.
    """
    def residuals(p):
        return np.hypot(p[0] - anchors[:, 0], p[1] - anchors[:, 1]) - ranges_m

    # Initial guess: weighted centroid
    w   = 1.0 / (ranges_m + 1e-6)
    x0  = np.average(anchors, axis=0, weights=w)
    res = least_squares(residuals, x0, method="lm")
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Rigid-body pose fit (SVD)
# ─────────────────────────────────────────────────────────────────────────────

def fit_rigid_body(world_pts, body_pts):
    """
    Fit rotation R and translation t such that  world ≈ R @ body + t.
    Uses SVD (Kabsch algorithm).

    world_pts, body_pts : (N, 2)
    Returns (x, y, theta_rad).
    """
    c_w = world_pts.mean(axis=0)
    c_b = body_pts.mean(axis=0)
    W   = (world_pts - c_w).T @ (body_pts - c_b)
    U, _, Vt = np.linalg.svd(W)
    R   = U @ np.diag([1, np.linalg.det(U @ Vt)]) @ Vt
    t   = c_w - R @ c_b
    theta = np.arctan2(R[1, 0], R[0, 0])
    return t[0], t[1], theta


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GDOP-based predicted pose uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def corner_covariance(pos_m, anchors=ANC, sigma_r=SIGMA_R):
    """
    Return 2×2 position covariance at pos_m given anchor geometry.
    Σ = σ² · (H^T H)^{-1}
    """
    dp  = pos_m - anchors                       # (3, 2)
    r   = np.linalg.norm(dp, axis=1, keepdims=True)
    H   = dp / r                                 # unit vectors  (3, 2)
    F   = H.T @ H                                # Fisher matrix (2, 2)
    det = np.linalg.det(F)
    if det < 1e-9:
        return np.eye(2) * 1e6
    return sigma_r ** 2 * np.linalg.inv(F)


def predicted_pose_uncertainty(corner_world_pts, body_pts=CORNERS_BODY,
                               anchors=ANC, sigma_r=SIGMA_R):
    """
    Propagate per-corner position covariances into pose (x, y, θ) covariance.

    Uses first-order (Jacobian) propagation through the rigid-body model.
    Returns Σ_pose (3×3) and per-corner σ_pos (m).
    """
    N = len(corner_world_pts)

    # Per-corner covariances
    Sigma_corners = [corner_covariance(p, anchors, sigma_r) for p in corner_world_pts]
    sigma_corner  = [np.sqrt(np.trace(S) / 2) for S in Sigma_corners]

    # Fit pose to get current R, t estimate
    x, y, theta = fit_rigid_body(corner_world_pts, body_pts)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Jacobian of each corner world position w.r.t. pose (x, y, θ)
    # p_i = R @ b_i + t
    # ∂p_i/∂x = [1, 0]
    # ∂p_i/∂y = [0, 1]
    # ∂p_i/∂θ = dR/dθ @ b_i
    dR = np.array([[-np.sin(theta), -np.cos(theta)],
                   [ np.cos(theta), -np.sin(theta)]])

    # Build full Jacobian (2N × 3) and block-diagonal weight matrix
    J   = np.zeros((2 * N, 3))
    W_blocks = []
    for i, b in enumerate(body_pts):
        row = 2 * i
        J[row:row+2, 0] = [1, 0]          # ∂/∂x
        J[row:row+2, 1] = [0, 1]          # ∂/∂y
        J[row:row+2, 2] = dR @ b          # ∂/∂θ
        W_blocks.append(np.linalg.inv(Sigma_corners[i]))

    W = np.block([[W_blocks[i] if i == j else np.zeros((2, 2))
                   for j in range(N)] for i in range(N)])

    JtWJ = J.T @ W @ J
    det  = np.linalg.det(JtWJ)
    if det < 1e-12:
        return np.eye(3) * 1e6, sigma_corner

    Sigma_pose = np.linalg.inv(JtWJ)
    return Sigma_pose, sigma_corner


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run all test poses
# ─────────────────────────────────────────────────────────────────────────────

def world_corners_gt(pose):
    """Compute ground-truth world positions of all 4 corners."""
    x  = pose["x_ft"]  * FT_TO_M
    y  = pose["y_ft"]  * FT_TO_M
    th = np.radians(pose["theta_deg"])
    R  = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th),  np.cos(th)]])
    return (R @ CORNERS_BODY.T).T + np.array([x, y])


results = {}

for test_id, pose_gt in POSES_GT.items():
    raw = RAW_MEAS_CM[test_id]              # (4, 3) raw cm readings

    # Calibrate and trilaterate each corner
    est_corners = []
    for c in range(4):
        cal_m = calibrate(raw[c])           # (3,) calibrated metres
        pos   = trilaterate(cal_m)          # (2,) estimated world position
        est_corners.append(pos)
    est_corners = np.array(est_corners)     # (4, 2)

    # Ground-truth corners
    gt_corners = world_corners_gt(pose_gt)  # (4, 2)

    # Fit robot pose from estimated corners
    x_est, y_est, th_est = fit_rigid_body(est_corners, CORNERS_BODY)

    # Ground-truth pose in metres
    x_gt  = pose_gt["x_ft"]  * FT_TO_M
    y_gt  = pose_gt["y_ft"]  * FT_TO_M
    th_gt = np.radians(pose_gt["theta_deg"])

    # Actual errors
    err_x     = abs(x_est - x_gt) * 100        # cm
    err_y     = abs(y_est - y_gt) * 100         # cm
    err_pos   = np.hypot(x_est - x_gt, y_est - y_gt) * 100  # cm
    err_theta = abs(np.degrees(np.arctan2(
        np.sin(th_est - th_gt), np.cos(th_est - th_gt))))    # deg

    # Per-corner actual position errors
    corner_errs_cm = np.linalg.norm(est_corners - gt_corners, axis=1) * 100

    # Predicted uncertainty
    Sigma_pose, sigma_corner_pred = predicted_pose_uncertainty(gt_corners)
    pred_sigma_x     = np.sqrt(Sigma_pose[0, 0]) * 100   # cm
    pred_sigma_y     = np.sqrt(Sigma_pose[1, 1]) * 100   # cm
    pred_sigma_pos   = np.sqrt((pred_sigma_x**2 + pred_sigma_y**2) / 2)
    pred_sigma_theta = np.degrees(np.sqrt(Sigma_pose[2, 2]))  # deg

    results[test_id] = {
        "x_gt": x_gt, "y_gt": y_gt, "th_gt": np.degrees(th_gt),
        "x_est": x_est * FT_TO_M / FT_TO_M,    # keep in m
        "y_est": y_est,
        "th_est": np.degrees(th_est),
        "err_x_cm":     err_x,
        "err_y_cm":     err_y,
        "err_pos_cm":   err_pos,
        "err_theta_deg": err_theta,
        "pred_sigma_x_cm":     pred_sigma_x,
        "pred_sigma_y_cm":     pred_sigma_y,
        "pred_sigma_pos_cm":   pred_sigma_pos,
        "pred_sigma_theta_deg": pred_sigma_theta,
        "corner_errs_cm":      corner_errs_cm,
        "sigma_corner_pred_cm": np.array(sigma_corner_pred) * 100,
        "est_corners": est_corners,
        "gt_corners":  gt_corners,
        "x_est_m": x_est,
        "y_est_m": y_est,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

TEST_IDS      = list(results.keys())
x_pos         = np.arange(len(TEST_IDS))
WIDTH         = 0.35
COLORS        = {"actual": "#e24b4a", "predicted": "#378add"}
corner_colors = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0"]

SUBTITLE = "3 fixed anchors · 4-corner robot · σ_range = 10 cm"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Overall error summary (position + heading + scatter)
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle(f"UWB Pose Error — Summary\n{SUBTITLE}",
              fontsize=13, fontweight="bold")

# Position error bar chart
ax = axes1[0]
actual_pos = [results[t]["err_pos_cm"]        for t in TEST_IDS]
pred_pos   = [results[t]["pred_sigma_pos_cm"] for t in TEST_IDS]
ax.bar(x_pos - WIDTH/2, actual_pos, WIDTH, label="Actual error",
       color=COLORS["actual"],    alpha=0.85)
ax.bar(x_pos + WIDTH/2, pred_pos,  WIDTH, label="Predicted 1σ (GDOP)",
       color=COLORS["predicted"], alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(TEST_IDS, fontsize=11)
ax.set_ylabel("Position error (cm)", fontsize=10)
ax.set_title("Overall position error", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# Heading error bar chart
ax = axes1[1]
actual_th = [results[t]["err_theta_deg"]        for t in TEST_IDS]
pred_th   = [results[t]["pred_sigma_theta_deg"] for t in TEST_IDS]
ax.bar(x_pos - WIDTH/2, actual_th, WIDTH, label="Actual",
       color=COLORS["actual"],    alpha=0.85)
ax.bar(x_pos + WIDTH/2, pred_th,   WIDTH, label="Pred 1σ",
       color=COLORS["predicted"], alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(TEST_IDS, fontsize=11)
ax.set_ylabel("Heading error (°)", fontsize=10)
ax.set_title("Heading error", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# Predicted vs actual scatter
ax = axes1[2]
all_actual = [results[t]["err_pos_cm"]        for t in TEST_IDS]
all_pred   = [results[t]["pred_sigma_pos_cm"] for t in TEST_IDS]
lim = max(max(all_actual), max(all_pred)) * 1.15
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="1:1 line")
for t in TEST_IDS:
    ax.scatter(results[t]["pred_sigma_pos_cm"], results[t]["err_pos_cm"],
               s=100, zorder=5)
    ax.annotate(t, (results[t]["pred_sigma_pos_cm"], results[t]["err_pos_cm"]),
                textcoords="offset points", xytext=(5, 3), fontsize=9)
ax.set_xlabel("Predicted 1σ (cm)", fontsize=10)
ax.set_ylabel("Actual error (cm)", fontsize=10)
ax.set_title("Predicted vs actual\n(above line = worse than expected)",
             fontsize=10, fontweight="bold")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig1.tight_layout()
fig1.savefig("uwb_pose_error_summary.png", dpi=150, bbox_inches="tight")
print("Saved  uwb_pose_error_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-corner error breakdown
# ─────────────────────────────────────────────────────────────────────────────

fig2, ax = plt.subplots(figsize=(14, 6))
fig2.suptitle(f"UWB Pose Error — Per-corner breakdown\n{SUBTITLE}",
              fontsize=13, fontweight="bold")

cx      = np.arange(len(TEST_IDS))
offsets = [-1.5, -0.5, 0.5, 1.5]
bar_w   = 0.18

for ci in range(4):
    actual_c = [results[t]["corner_errs_cm"][ci]       for t in TEST_IDS]
    pred_c   = [results[t]["sigma_corner_pred_cm"][ci] for t in TEST_IDS]
    ax.bar(cx + offsets[ci] * bar_w, actual_c, bar_w,
           color=corner_colors[ci], alpha=0.85)
    ax.bar(cx + offsets[ci] * bar_w, pred_c,   bar_w,
           color=corner_colors[ci], alpha=0.35,
           hatch="///", edgecolor="gray", linewidth=0.5)

ax.set_xticks(cx); ax.set_xticklabels(TEST_IDS, fontsize=12)
ax.set_ylabel("Position error (cm)", fontsize=11)
ax.set_title("Solid = actual error  ·  hatched = predicted 1σ",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
handles = [mpatches.Patch(color=corner_colors[i], label=f"Corner {i+1}")
           for i in range(4)]
ax.legend(handles=handles, fontsize=10, ncol=4)

fig2.tight_layout()
fig2.savefig("uwb_pose_error_corners.png", dpi=150, bbox_inches="tight")
print("Saved  uwb_pose_error_corners.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Floor-plan overlays (one subplot per test pose)
# ─────────────────────────────────────────────────────────────────────────────

fig3, axes3 = plt.subplots(1, 5, figsize=(22, 5))
fig3.suptitle(f"UWB Pose Error — Floor-plan overlays\n{SUBTITLE}",
              fontsize=13, fontweight="bold")

def draw_robot(ax, corners, color, lw, label, ls="-"):
    pts = np.vstack([corners, corners[0]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls,
            label=label, zorder=4)
    ax.scatter(corners[:, 0], corners[:, 1], s=25, color=color, zorder=5)

for idx, test_id in enumerate(TEST_IDS):
    r  = results[test_id]
    ax = axes3[idx]

    ax.set_xlim(-0.3, ROOM_W + 0.3)
    ax.set_ylim(-0.3, ROOM_H + 0.3)
    ax.add_patch(mpatches.Rectangle((0, 0), ROOM_W, ROOM_H,
                 lw=1.5, ec="black", fc="#f9f9f9", zorder=1))
    ax.set_aspect("equal")
    ax.set_title(f"Test {test_id}\npos {r['err_pos_cm']:.1f} cm  "
                 f"θ {r['err_theta_deg']:.1f}°",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)

    # Fixed anchors
    ax.scatter(ANC[:, 0], ANC[:, 1], s=90, marker="^",
               color="#1565c0", edgecolors="navy", lw=0.8, zorder=5)
    for name, pos in ANCHORS.items():
        ax.annotate(name, pos, fontsize=7, xytext=(4, 4),
                    textcoords="offset points", color="navy")

    draw_robot(ax, r["gt_corners"],  "#2ecc40", 1.8, "Ground truth", ls="-")
    draw_robot(ax, r["est_corners"], "#e24b4a", 1.8, "Estimated",    ls="--")

    # Error vectors corner → estimated
    for c in range(4):
        ax.annotate("", xy=r["est_corners"][c], xytext=r["gt_corners"][c],
                    arrowprops=dict(arrowstyle="-|>", color="darkorange",
                                   lw=1.2, mutation_scale=8), zorder=6)

    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

fig3.tight_layout()
fig3.savefig("uwb_pose_error_floorplans.png", dpi=150, bbox_inches="tight")
print("Saved  uwb_pose_error_floorplans.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'Test':<6} {'GT x,y (m)':<20} {'Est x,y (m)':<20} "
      f"{'Err pos (cm)':<14} {'Pred σ (cm)':<13} {'Err θ (°)':<11} {'Pred σθ (°)'}")
print("─" * 98)
for t in TEST_IDS:
    r = results[t]
    print(f"{t:<6} "
          f"({r['x_gt']:.2f}, {r['y_gt']:.2f})              "
          f"({r['x_est_m']:.2f}, {r['y_est_m']:.2f})              "
          f"{r['err_pos_cm']:<14.1f}"
          f"{r['pred_sigma_pos_cm']:<13.1f}"
          f"{r['err_theta_deg']:<11.1f}"
          f"{r['pred_sigma_theta_deg']:.1f}")

plt.show()