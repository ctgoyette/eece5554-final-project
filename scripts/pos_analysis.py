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


def estimate_position(anchors, distances, initial_guess=(0, 0)):
    anchors = np.array(anchors)
    distances = np.array(distances)

    result = least_squares(
        residuals,
        x0=np.array(initial_guess),
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

def remove_anchor_data(anchors, distances_list, removed_index):
    new_anchors = [a for i, a in enumerate(anchors) if i != removed_index]

    new_distances = []
    for d in distances_list:
        new_distances.append([val for i, val in enumerate(d) if i != removed_index])

    return new_anchors, new_distances

def analyze_test(input_file, plot_title, remove_anchor=None):
    origin, angle, corners = parse_measurements(f"{DATA_FILE_DIR}{input_file}")

    centroid_true, corners_true = actual_position(origin, angle)

    corner_labels = ["C1", "C2", "C3", "C4"]
    corner_data = [corners[c] for c in corner_labels if c in corners]

    anchors = ANCHORS

    if remove_anchor is not None:
        anchors, corner_data = remove_anchor_data(ANCHORS, corner_data, remove_anchor)

    corner_dist = []

    for d in corner_data:
        corner_dist.append(
            estimate_position(anchors, d, initial_guess=(centroid_true[0], centroid_true[1]))
        )

    x_anchors = [point[0] for point in anchors]
    y_anchors = [point[1] for point in anchors]

    x_true_corners = [point[0] for point in corners_true]
    y_true_corners = [point[1] for point in corners_true]

    x_est_corners = [point[0] for point in corner_dist]
    y_est_corners = [point[1] for point in corner_dist]
    x_est_centroid = np.mean(x_est_corners)
    y_est_centroid = np.mean(y_est_corners)

    centroid_err = distance((x_est_centroid, y_est_centroid), centroid_true)
    print(f"{plot_title} Centroid Error: {centroid_err} cm")

    plt.scatter(x_true_corners, y_true_corners, color="green", label="Actual Corner Positions")
    plt.scatter(centroid_true[0], centroid_true[1], color='purple', label="Actual Centroid Position")

    plt.scatter(x_anchors, y_anchors, color="red", label="Anchors")

    plt.scatter(x_est_corners, y_est_corners, color='blue', label="Estimated Positions")
    plt.scatter(x_est_centroid, y_est_centroid, color='yellow', label="Estimated Centroid Position")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    plt.title(plot_title)
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{OUTPUT_FILE_DIR}{plot_title}.png")
    plt.close()

analyze_test("test_a.txt", "Test A All")
analyze_test("test_b.txt", "Test B All")
analyze_test("test_c.txt", "Test C All")
analyze_test("test_d.txt", "Test D All")
analyze_test("test_e.txt", "Test E All")

analyze_test("test_d.txt", "Test D Removed 0", 0)