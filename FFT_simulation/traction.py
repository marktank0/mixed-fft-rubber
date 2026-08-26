# -*- coding: utf-8 -*-
"""Foreground YAML runner for traction-style simulations."""

import _bootstrap  # noqa: F401  (puts the repo root and FFT_simulation on sys.path)
from project_paths import run_config_path

from batch_run import main


if __name__ == "__main__":
    main(
        default_config=run_config_path("1_test_run.yaml"),
        default_mode="cases",
        default_max_workers=1,
        default_log_to_file=False,
        default_on_existing="overwrite",
    )