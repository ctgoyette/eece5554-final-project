import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

CHASSIS_LENGTH_CM = 20 * 2.54
CHASSIS_WIDTH_CM = 14 * 2.54
FEET_TO_CM = 12 * 2.54

ANCHORS = [
    (0, 0),
    (0, 4.5 * FEET_TO_CM),
    (4.5 * FEET_TO_CM, 0)
]

DATA_FILE_DIR = "../data/formatted/"
OUTPUT_FILE_DIR = "../results/plots/"

def parse_measurements(filepath):
    origin = None
    angle = None
    corner_distances = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            label, values = line.split(":")
            nums = [float(v) for v in values.split(",")]

            if label == "Actual":
                if len(nums) != 2:
                    raise ValueError("Actual must have 2 values (x, y)")
                origin = (nums[0] * FEET_TO_CM, nums[1] * FEET_TO_CM)
            
            elif label == "Angle":
                if len(nums) != 1:
                    raise ValueError("Angle must have 1 value (theta)")
                angle = math.radians(nums[0])

            elif label.startswith("C"):
                corner_distances[label] = nums

            else:
                raise ValueError(f"Unknown label: {label}")

    if origin is None:
        raise ValueError("Missing 'Actual' line")

    return origin, angle, corner_distances

def parse_calibration_single(file):
    reference = None
    measured = None

    with open(file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            label, values = line.split(":")
            nums = [float(v) for v in values.split(",")]

            if label == "Reference":
                if len(nums) != 10:
                    raise ValueError("Reference must have 10 values")
                reference = np.array([num * FEET_TO_CM for num in nums])
            
            elif label == "Measured":
                if len(nums) != 10:
                    raise ValueError("Measured must have 10 values")
                measured = nums

            else:
                raise ValueError(f"Unknown label: {label}")

    if reference is None:
        raise ValueError("Missing 'Reference' line")
    if measured is None:
        raise ValueError("Missing 'Measured' line")
    
    m,b = np.polyfit(reference, measured, 1)
    print(m, b)

    return (m, b)

def parse_calibration(a0_cal_file, a1_cal_file, a2_cal_file):
    params = []
    params.append(parse_calibration_single(a0_cal_file))
    params.append(parse_calibration_single(a1_cal_file))
    params.append(parse_calibration_single(a2_cal_file))

    return params

def load_time_series(filepath):
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#!"):
                continue

            try:
                data.append(float(line))
            except ValueError:
                continue

    return np.array(data)

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def residuals(pos, anchors, distances):
    x, y = pos
    res = []

    for (xi, yi), di in zip(anchors, distances):
        predicted = np.sqrt((x - xi)**2 + (y - yi)**2)
        res.append(predicted - di)

    return res

def robot_residuals(params, anchors, measured_distances):
    x, y, theta = params

    local_corners = [
        (0, -CHASSIS_WIDTH_CM/2),   # bottom-left
        (0,  CHASSIS_WIDTH_CM/2),   # top-left
        (CHASSIS_LENGTH_CM, -CHASSIS_WIDTH_CM/2),  # bottom-right
        (CHASSIS_LENGTH_CM,  CHASSIS_WIDTH_CM/2)    # top-right
    ]

    residuals = []

    for i, (cx, cy) in enumerate(local_corners):
        rx = cx * np.cos(theta) - cy * np.sin(theta) + x
        ry = cx * np.sin(theta) + cy * np.cos(theta) + y

        for (ax, ay), d_meas in zip(anchors, measured_distances[i]):
            d_pred = np.sqrt((rx - ax)**2 + (ry - ay)**2)
            residuals.append(d_pred - d_meas)

    return residuals

def estimate_position(anchors, distances, initial_guess=(0, 0), cal_params=None):
    """
    anchors: list of (x,y), length = A
    distances: list of lists -> shape (C corners, A anchors)
    cal_params: list of (m,b) per anchor OR per sensor channel
    """

    n_anchors = len(anchors)

    # ---- 1. Validate input shape ----
    for i, d in enumerate(distances):
        if len(d) != n_anchors:
            raise ValueError(
                f"Corner {i}: expected {n_anchors} distances, got {len(d)}"
            )

    # ---- 2. Apply calibration safely ----
    if cal_params is not None:
        if len(cal_params) != n_anchors:
            raise ValueError(
                f"Calibration mismatch: {len(cal_params)} cal params but {n_anchors} anchors"
            )

        calibrated = []
        for corner_dists in distances:
            calibrated_corner = [
                (d - b) / m
                for d, (m, b) in zip(corner_dists, cal_params)
            ]
            calibrated.append(calibrated_corner)

        distances = calibrated

    # ---- 3. DO NOT convert to numpy (important) ----
    # Keep as pure Python lists to avoid shape corruption

    # ---- 4. Solve ----
    result = least_squares(
        robot_residuals,
        x0=[initial_guess[0], initial_guess[1], 0],
        args=(anchors, distances)
    )

    return result.x

def estimate_centroid():
    pass

def rotate_point(x, y, px, py, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x -= px
    y -= py

    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    return (x_new + px, y_new + py)

def actual_position(origin, angle):
    rotation_point = (0, CHASSIS_WIDTH_CM/2)

    centroid = (CHASSIS_LENGTH_CM / 2, CHASSIS_WIDTH_CM / 2)

    points = [
        (0, 0),
        (0, CHASSIS_WIDTH_CM),
        (CHASSIS_LENGTH_CM, 0),
        (CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM)
    ]

    rotated = [
        rotate_point(x, y, rotation_point[0], rotation_point[1], angle)
        for x, y in points
    ]
    rotated = [(x + origin[0], y + origin[1])for (x, y) in rotated]

    centroid = rotate_point(centroid[0], centroid[1], rotation_point[0], rotation_point[1], angle)
    centroid = (centroid[0] + origin[0], centroid[1] + origin[1])

    return centroid, rotated

def remove_anchor_data(anchors, distances_list, removed_indices):
    if removed_indices is None:
        return anchors, distances_list

    removed_set = set(removed_indices)

    new_anchors = [
        a for i, a in enumerate(anchors)
        if i not in removed_set
    ]

    new_distances = []
    for d in distances_list:
        new_distances.append([
            val for i, val in enumerate(d)
            if i not in removed_set
        ])

    return new_anchors, new_distances

def compute_gdop(anchors, est_pos):
    H = []
    x, y = est_pos

    for xi, yi in anchors:
        r = np.sqrt((x - xi)**2 + (y - yi)**2)
        H.append([(x - xi)/r, (y - yi)/r])

    H = np.array(H)
    Q = np.linalg.inv(H.T @ H)
    gdop = np.sqrt(np.trace(Q))
    return gdop

def allan_deviation(data, max_m=None):
    N = len(data)
    if max_m is None:
        max_m = N // 10

    taus = []
    adev = []

    for m in np.logspace(0, np.log10(max_m), 50).astype(int):
        if m < 1 or 2*m >= N:
            continue

        diffs = data[2*m:] - 2*data[m:-m] + data[:-2*m]
        sigma2 = np.mean(diffs**2) / (2 * (m**2))

        taus.append(m)
        adev.append(np.sqrt(sigma2))

    return np.array(taus), np.array(adev)

def analyze_test(input_file, plot_title, remove_anchors=None, remove_corners=None, use_cal=False):
    origin, angle, corners = parse_measurements(f"{DATA_FILE_DIR}{input_file}")

    centroid_true, corners_true = actual_position(origin, angle)

    corner_labels = ["C1", "C2", "C3", "C4"]
    if remove_corners is not None:
        if isinstance(remove_corners, int):
            remove_corners = [remove_corners]
        for c in sorted(remove_corners, reverse=True):
            del corner_labels[c]
    corner_data = [corners[c] for c in corner_labels if c in corners]

    anchors = ANCHORS

    if remove_anchors is not None:
        if isinstance(remove_anchors, int):
            remove_anchors = [remove_anchors]

        anchors, corner_data = remove_anchor_data(ANCHORS, corner_data, remove_anchors)

    cal_params = None
    if use_cal:
        cal_params = parse_calibration(f"{DATA_FILE_DIR}cal_id0.txt", f"{DATA_FILE_DIR}cal_id1.txt", f"{DATA_FILE_DIR}cal_id2.txt")

    x, y, theta = estimate_position(
        anchors,
        corner_data,
        initial_guess=(centroid_true[0], centroid_true[1]),
        cal_params=cal_params
    )

    x_anchors = [point[0] for point in anchors]
    y_anchors = [point[1] for point in anchors]

    x_true_corners = [point[0] for point in corners_true]
    y_true_corners = [point[1] for point in corners_true]

    local_corners = [
        (0, -CHASSIS_WIDTH_CM/2),   # bottom-left
        (0,  CHASSIS_WIDTH_CM/2),   # top-left
        (CHASSIS_LENGTH_CM, -CHASSIS_WIDTH_CM/2),  # bottom-right
        (CHASSIS_LENGTH_CM,  CHASSIS_WIDTH_CM/2)    # top-right
    ]

    x_est_corners = []
    y_est_corners = []

    for cx, cy in local_corners:
        rx = cx*np.cos(theta) - cy*np.sin(theta) + x
        ry = cx*np.sin(theta) + cy*np.cos(theta) + y

        x_est_corners.append(rx)
        y_est_corners.append(ry)
        
    centroid_local = (CHASSIS_LENGTH_CM/2, 0)

    x_est_centroid = (
        centroid_local[0] * np.cos(theta)
        - centroid_local[1] * np.sin(theta)
        + x
    )

    y_est_centroid = (
        centroid_local[0] * np.sin(theta)
        + centroid_local[1] * np.cos(theta)
        + y
)

    centroid_err = distance((x_est_centroid, y_est_centroid), centroid_true)
    res = robot_residuals([x, y, theta], anchors, corner_data)
    rmse = np.sqrt(np.mean(np.square(res)))
    gdop = compute_gdop(anchors, (x_est_centroid, y_est_centroid))
    output_stats = f"Error: {centroid_err:.2f} cm | RMSE: {rmse:.2f} cm | GDOP: {gdop:.2f}"
    print(output_stats)

    plt.scatter(x_true_corners, y_true_corners, color="green", label="Actual Corner Positions")
    plt.scatter(centroid_true[0], centroid_true[1], color='purple', label="Actual Centroid Position")

    plt.scatter(x_anchors, y_anchors, color="red", label="Anchors")

    plt.scatter(x_est_corners, y_est_corners, color='blue', label="Estimated Positions")
    plt.scatter(x_est_centroid, y_est_centroid, color='yellow', label="Estimated Centroid Position")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    plt.title(f"{plot_title}\n{output_stats}")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{OUTPUT_FILE_DIR}{plot_title}.png")
    plt.close()

def analyze_time_series(los_file, nlos_file, plot_title):

    TRUE_DIST_CM = 5 * 12 * 2.54  # 152.4 cm
    d_los = load_time_series(los_file)
    d_nlos = load_time_series(nlos_file)

    err_los = d_los - TRUE_DIST_CM
    err_nlos = d_nlos - TRUE_DIST_CM

    t1, a1 = allan_deviation(err_los)
    t2, a2 = allan_deviation(err_nlos)

    min_idx1 = np.argmin(a1)
    min_idx2 = np.argmin(a2)

    print("Best Allan deviation (LOS):", a1[min_idx1], "cm")
    print("Optimal averaging window (LOS):", t1[min_idx1], "samples")

    print("Best Allan deviation (NLOS):", a2[min_idx2], "cm")
    print("Optimal averaging window (NLOS):", t2[min_idx2], "samples")

    plt.loglog(t1, a1, label="LOS")
    plt.loglog(t2, a2, label="NLOS")

    plt.xlabel("Averaging Time (samples)")
    plt.ylabel("Allan Deviation (cm)")
    plt.legend()
    plt.title("Allan Deviation - UWB Distance Noise")
    plt.grid(True, which="both")
    plt.savefig(f"{OUTPUT_FILE_DIR}{plot_title}.png")
    plt.close()

print("########### Stationary Stats ###########")
analyze_test("test_a.txt", "Test A All")
analyze_test("test_a.txt", "Test A Removed Anchor 0", 0)
analyze_test("test_a.txt", "Test A Removed Anchor 1", 1)
analyze_test("test_a.txt", "Test A Removed Anchor 2", 2)

analyze_test("test_b.txt", "Test B All")
analyze_test("test_b.txt", "Test B All (Calibrated)", use_cal=True)
analyze_test("test_b.txt", "Test B Removed Anchor 0", 0)
analyze_test("test_b.txt", "Test B Removed Anchor 1", 1)
analyze_test("test_b.txt", "Test B Removed Anchor 2", 2)

analyze_test("test_c.txt", "Test C All")
analyze_test("test_c.txt", "Test C Removed Anchor 0", 0)

analyze_test("test_d.txt", "Test D All")
analyze_test("test_d.txt", "Test D Removed Anchor 0", 0)

analyze_test("test_e.txt", "Test E All")
analyze_test("test_e.txt", "Test E Removed Anchor 0", 0)

print("\n########### Allan Deviation ###########")
analyze_time_series("../data/raw/5' Sounding/Allan_Variance_Unobstructed.csv", "../data/raw/5' Sounding/Allan_Variance_Obstructed.csv", "Allan Variance")