from __future__ import annotations

import os

LONKIT_N_JOBS_ENV = "LONKIT_N_JOBS"


def max_workers_from_env() -> int | None:
    value = os.environ.get(LONKIT_N_JOBS_ENV)
    return int(value) if value is not None else None
