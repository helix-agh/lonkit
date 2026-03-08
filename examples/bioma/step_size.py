from problems import ackley4, griewank, schwefel2_26

from lonpy import StepSizeEstimator, StepSizeEstimatorConfig

FUNCTIONS = {
    "Ackley 4": {
        "func": ackley4,
        "bounds": (-35, 35),
        "dimensions": [3, 5, 8],
    },
    "Griewank": {
        "func": griewank,
        "bounds": (-600, 600),
        "dimensions": [3, 5, 8],
    },
    "Schwefel 2.26": {
        "func": schwefel2_26,
        "bounds": (-500, 500),
        "dimensions": [3, 5, 8],
    },
}

CONFIG = StepSizeEstimatorConfig(
    n_samples=100,
    n_perturbations=30,
    target_escape_rate=0.5,
    search_precision=4,
    coordinate_precision=4,
)


def main() -> None:
    estimator = StepSizeEstimator(CONFIG)

    for func_name, cfg in FUNCTIONS.items():
        lb, ub = cfg["bounds"]
        for d in cfg["dimensions"]:
            domain = [(lb, ub)] * d
            print(f"\n  {func_name}  D = {d}")

            result = estimator.estimate(cfg["func"], domain)

            step_abs = result.step_size * abs(ub - lb)
            print(f"  percentage  : {result.step_size:.8f}")
            print(f"  escape rate : {result.escape_rate:.6f}")
            print(f"  error       : {result.error:.6f}")
            print(f"  step        : {step_abs:.4f}")


if __name__ == "__main__":
    main()
