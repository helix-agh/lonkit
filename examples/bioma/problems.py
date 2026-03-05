import numpy as np


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2 / 4000.0) - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def schwefel2_26(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def ackley4(x: np.ndarray) -> float:
    f = 0.0
    for i in range(len(x) - 1):
        f += np.exp(-0.2) * np.sqrt(x[i] ** 2 + x[i + 1] ** 2) + 3 * (
            np.cos(2 * x[i]) + np.sin(2 * x[i + 1])
        )
    return f


def spread_spectrum_radar_polly_phase(x: np.ndarray) -> float:
    var = 2 * len(x) - 1
    hsum = np.zeros(2 * var)
    for kk in range(2 * var):
        if (kk + 1) % 2 != 0:
            i = int((kk + 2) / 2)
            hsum[kk] = 0
            for j in range((i - 1), len(x)):
                summ = np.sum(x[abs(2 * i - (j + 1) - 1) : j + 1])
                hsum[kk] = np.cos(summ) + hsum[kk]
        else:
            i = int((kk + 1) / 2)
            hsum[kk] = 0
            for j in range(i, len(x)):
                summ = np.sum(x[abs(2 * i - (j + 1)) : j + 1])
                hsum[kk] = np.cos(summ) + hsum[kk]
            hsum[kk] = hsum[kk] + 0.5
    return np.max(hsum)


def sum_of_squares(center: np.ndarray, x: np.ndarray) -> float:
    # x: (N, D), center: (K, D) -> dists: (N, K)
    dists = np.sum((x[:, np.newaxis, :] - center[np.newaxis, :, :]) ** 2, axis=2)
    return float(np.sum(np.min(dists, axis=1)))


_RUSPINI_DATA = np.array(
    [
        (4, 53),
        (5, 63),
        (10, 59),
        (9, 77),
        (13, 49),
        (13, 69),
        (12, 88),
        (15, 75),
        (18, 61),
        (19, 65),
        (22, 74),
        (27, 72),
        (28, 76),
        (24, 58),
        (27, 55),
        (28, 60),
        (30, 52),
        (31, 60),
        (32, 61),
        (36, 72),
        (28, 147),
        (32, 149),
        (35, 153),
        (33, 154),
        (38, 151),
        (41, 150),
        (38, 145),
        (38, 143),
        (32, 143),
        (34, 141),
        (44, 156),
        (44, 149),
        (44, 143),
        (46, 142),
        (47, 149),
        (49, 152),
        (50, 142),
        (53, 144),
        (52, 152),
        (55, 155),
        (54, 124),
        (60, 136),
        (63, 139),
        (86, 132),
        (85, 115),
        (85, 96),
        (78, 94),
        (74, 96),
        (97, 122),
        (98, 116),
        (98, 124),
        (99, 119),
        (99, 128),
        (101, 115),
        (108, 111),
        (110, 111),
        (108, 116),
        (111, 126),
        (115, 117),
        (117, 115),
        (70, 4),
        (77, 12),
        (83, 21),
        (61, 15),
        (69, 15),
        (78, 16),
        (66, 18),
        (58, 13),
        (64, 20),
        (69, 21),
        (66, 23),
        (61, 25),
        (76, 27),
        (72, 31),
        (64, 30),
    ],
    dtype=float,
)


def ssc_ruspini(c: np.ndarray) -> float:
    center = c.reshape(-1, 2)
    return sum_of_squares(center, _RUSPINI_DATA)
