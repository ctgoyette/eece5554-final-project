import numpy as np
from scipy.optimize import least_squares

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

anchors = [
    (0, 0),
    (0, 4.5*12),
    (4.5*12, 0)
]

distances = [3.6, 2.8, 3.2, 2.5]

pos = estimate_position(anchors, distances, initial_guess=(2, 2))

print("Estimated position:", pos)