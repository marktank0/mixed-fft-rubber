# -*- coding: utf-8 -*-
"""YAML-driven runner for one or many independent FFT simulations."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap  # noqa: F401  (puts the repo root and FFT_simulation on sys.path)
from project_paths import run_config_path

from fg.io_paths import output_run_path
from simulation_config import (
    SUPPORTED_MODES,
    apply_thread_env,
    build_cases,
    get_execution_settings,
    load_config,
    resolved_config_path,
    write_resolved_config,
)


DEFAULT_CONFIG = run_config_path("1_test_run.yaml")


def _run_case_worker(case):
    from run_case import run_case

    return run_case(case)


def _case_output_path(case):
    return output_run_path(case["structure_path"], case.get("output_path"), case.get("output_name"))


def _prepare_cases(cases, on_existing):
    runnable = []
    skipped = []
    for case in cases:
        output_path = _case_output_path(case)
        if os.path.exists(output_path):
            if on_existing == "error":
                raise RuntimeError(
                    "Output path already exists for case {!r}: {}"
                    .format(case.get("case_name", case["structure_path"]), output_path)
                )
            if on_existing == "skip":
                skipped.append({
                    "status": "skipped",
                    "structure_path": case["structure_path"],
                    "output_path": output_path,
                })
                print("skipping existing {} -> {}".format(case["structure_path"], output_path))
                continue
        runnable.append(case)
    return runnable, skipped


def run_batch(cases, max_workers, on_existing="error"):
    runnable, results = _prepare_cases(cases, on_existing)
    if not runnable:
        print("No cases to run.")
        return results

    actual_workers = min(max_workers, len(runnable))
    print("Running {} cases with {} workers.".format(len(runnable), actual_workers))

    if actual_workers == 1:
        for case in runnable:
            result = _run_case_worker(case)
            results.append(result)
            print("finished {} -> {}".format(result["structure_path"], result["output_path"]))
        return results

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(_run_case_worker, case) for case in runnable]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("finished {} -> {}".format(result["structure_path"], result["output_path"]))
    return results


def print_dry_run(run_plan):
    print("mode: {}".format(run_plan["mode"]))
    print("base_path: {}".format(run_plan["base_path"]))
    print("max_workers: {}".format(run_plan["execution"]["max_workers"]))
    print("on_existing: {}".format(run_plan["execution"]["on_existing"]))
    print("cases: {}".format(len(run_plan["cases"])))
    for case in run_plan["cases"]:
        print("- {} -> {}".format(case["structure_path"], _case_output_path(case)))


def run_from_config(
    config_path,
    base_path_override=None,
    mode_override=None,
    max_workers_override=None,
    log_to_file_override=None,
    on_existing_override=None,
    dry_run=False,
    config=None,
):
    """Run every case in a YAML config.

    `config` lets a caller pass an already-loaded (and possibly expanded)
    config dictionary - contrast_sweep.py builds its cases in memory and hands
    them here, so the sweep gets the same defaults merging, path resolution,
    resolved-config dump and worker pool as a plain batch run.
    """
    if config is None:
        config = load_config(config_path)
    apply_thread_env(config)
    run_plan = build_cases(config, base_path_override=base_path_override, mode_override=mode_override)
    if max_workers_override is not None:
        execution = get_execution_settings(config, max_workers_override=max_workers_override)
        run_plan["execution"].update(execution)
    if log_to_file_override is not None:
        run_plan["execution"]["log_to_file"] = bool(log_to_file_override)
        for case in run_plan["cases"]:
            case["log_to_file"] = bool(log_to_file_override)
    if on_existing_override is not None:
        run_plan["execution"]["on_existing"] = on_existing_override

    if dry_run:
        print_dry_run(run_plan)
        return []

    if run_plan["execution"].get("save_resolved_config", True):
        resolved_dir = resolved_config_path(config, run_plan["base_path"])
        resolved_file = write_resolved_config(resolved_dir, config, run_plan)
        if resolved_file:
            print("resolved config saved to {}".format(resolved_file))

    return run_batch(
        run_plan["cases"],
        run_plan["execution"]["max_workers"],
        on_existing=run_plan["execution"]["on_existing"],
    )


def parse_args(default_config=DEFAULT_CONFIG, default_mode=None, default_max_workers=None):
    parser = argparse.ArgumentParser(description="Run FFT simulations from a YAML config.")
    parser.add_argument("config", nargs="?", default=default_config, help="Path to a YAML run config.")
    parser.add_argument("--base-path", help="Override config base_path, useful on servers.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=default_mode, help="Override run.mode.")
    parser.add_argument("--max-workers", type=int, default=default_max_workers, help="Override execution.max_workers.")
    parser.add_argument(
        "--terminal-output",
        action="store_true",
        help="Print solver progress to this terminal instead of redirecting each case to run.log.",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Redirect each case's solver progress to run.log.",
    )
    parser.add_argument("--on-existing", choices=("skip", "overwrite", "error"), help="Override execution.on_existing.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved cases without running them.")
    return parser.parse_args()


def main(
    default_config=DEFAULT_CONFIG,
    default_mode=None,
    default_max_workers=None,
    default_log_to_file=None,
    default_on_existing=None,
):
    args = parse_args(
        default_config=default_config,
        default_mode=default_mode,
        default_max_workers=default_max_workers,
    )
    log_to_file_override = default_log_to_file
    if args.terminal_output:
        log_to_file_override = False
    if args.log_to_file:
        log_to_file_override = True
    on_existing_override = args.on_existing if args.on_existing is not None else default_on_existing
    run_from_config(
        args.config,
        base_path_override=args.base_path,
        mode_override=args.mode,
        max_workers_override=args.max_workers,
        log_to_file_override=log_to_file_override,
        on_existing_override=on_existing_override,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
