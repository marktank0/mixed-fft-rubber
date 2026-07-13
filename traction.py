# -*- coding: utf-8 -*-
"""Foreground YAML runner for traction-style simulations."""

from batch_run import main


if __name__ == "__main__":
    main(
        default_config="Run_configs/1_test_run.yaml",
        default_mode="cases",
        default_max_workers=1,
        default_log_to_file=False,
        default_on_existing="overwrite",
    )