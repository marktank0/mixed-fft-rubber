# -*- coding: utf-8 -*-
"""YAML configuration helpers for FFT simulation launchers."""

import copy
import fnmatch
import glob
import json
import os

from project_paths import PROJECT_ROOT

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "FFT_WORKERS": "1",
}

SUPPORTED_MODES = ("cases", "batch")
SUPPORTED_ON_EXISTING = ("skip", "overwrite", "error")


class ConfigError(ValueError):
    """Raised when a simulation YAML file is invalid."""


def load_config(path):
    """Load a YAML configuration file."""
    with open(path) as file:
        text = file.read()

    if yaml is not None:
        config = yaml.safe_load(text)
    else:
        config = _load_simple_yaml(text)

    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    version = config.get("version")
    if version != 1:
        raise ConfigError("Unsupported config version {!r}; expected version 1.".format(version))

    return config


def apply_thread_env(config):
    """Apply numerical library thread limits before importing solver modules."""
    execution = config.get("execution", {}) or {}
    thread_env = execution.get("thread_env", THREAD_ENV_DEFAULTS)
    if thread_env is None:
        return {}
    if not isinstance(thread_env, dict):
        raise ConfigError("execution.thread_env must be a mapping or null.")

    applied = {}
    for key, value in thread_env.items():
        applied[str(key)] = str(value)
        os.environ[str(key)] = str(value)
    return applied


def resolve_base_path(config, base_path_override=None):
    """Resolve the run base path from CLI override or YAML.

    A relative ``base_path`` in the YAML is resolved against the repository
    root, not the working directory: the configs live in
    FFT_simulation/Run_configs/ but their structure, charge and output paths
    are written relative to the repository root, and a run must resolve to the
    same files whether it was started from the root, from FFT_simulation/ or
    from a server job directory.

    A ``--base-path`` given on the command line is resolved against the working
    directory instead, which is what a human typing a path expects.
    """
    if base_path_override is not None:
        raw_base_path = os.path.expanduser(os.path.expandvars(str(base_path_override)))
        return os.path.abspath(raw_base_path)

    raw_base_path = os.path.expanduser(os.path.expandvars(str(config.get("base_path", "."))))
    if os.path.isabs(raw_base_path):
        return os.path.normpath(raw_base_path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, raw_base_path))


def resolve_path(path, base_path):
    """Resolve relative project paths against base_path."""
    if path is None:
        return None
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_path, path))


def get_run_mode(config, mode_override=None):
    """Return the selected run mode: 'cases' or 'batch'."""
    if mode_override:
        mode = mode_override
    else:
        run = config.get("run", {}) or {}
        mode = run.get("mode")
        if mode is None and "run_mode" in config:
            mode = config["run_mode"]
        if mode is None and "single_or_batch" in config:
            legacy = config["single_or_batch"]
            mode = "cases" if int(legacy) == 0 else "batch"
        if mode is None:
            mode = "cases"

    if mode not in SUPPORTED_MODES:
        raise ConfigError("Unsupported run mode {!r}; use one of {}.".format(mode, SUPPORTED_MODES))
    return mode


def get_execution_settings(config, max_workers_override=None):
    """Return normalized execution settings."""
    execution = config.get("execution", {}) or {}
    max_workers = max_workers_override
    if max_workers is None:
        max_workers = execution.get("max_workers", 1)
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ConfigError("execution.max_workers must be at least 1.")

    on_existing = execution.get("on_existing", "error")
    if on_existing not in SUPPORTED_ON_EXISTING:
        raise ConfigError(
            "execution.on_existing must be one of {}; got {!r}."
            .format(SUPPORTED_ON_EXISTING, on_existing)
        )

    return {
        "max_workers": max_workers,
        "log_to_file": bool(execution.get("log_to_file", False)),
        "on_existing": on_existing,
        "save_resolved_config": bool(execution.get("save_resolved_config", True)),
    }


def build_cases(config, base_path_override=None, mode_override=None):
    """Build resolved run_case dictionaries from YAML config."""
    base_path = resolve_base_path(config, base_path_override=base_path_override)
    mode = get_run_mode(config, mode_override=mode_override)
    execution = get_execution_settings(config)

    if mode == "cases":
        raw_cases = config.get("cases", [])
    else:
        raw_cases = _build_batch_cases(config, base_path)

    if not isinstance(raw_cases, list):
        raise ConfigError("cases must be a list.")

    cases = [
        _normalize_case(config, raw_case, base_path, execution)
        for raw_case in raw_cases
    ]

    return {
        "base_path": base_path,
        "mode": mode,
        "execution": execution,
        "cases": cases,
    }


def resolved_config_path(config, base_path):
    """Return the experiment directory used for archived resolved configs."""
    output_root = _experiment_output_root(config)
    if output_root is None:
        return None
    return resolve_path(output_root, base_path)


def write_resolved_config(path, config, run_plan):
    """Write the expanded case plan used for a run."""
    if path is None:
        return None

    os.makedirs(path, exist_ok=True)
    outfile = os.path.join(path, "resolved_config.yaml")
    payload = {
        "version": config.get("version"),
        "base_path": run_plan["base_path"],
        "mode": run_plan["mode"],
        "execution": run_plan["execution"],
        "cases": run_plan["cases"],
    }
    with open(outfile, "w") as file:
        if yaml is not None:
            yaml.safe_dump(payload, file, sort_keys=False)
        else:
            json.dump(payload, file, indent=2)
            file.write("\n")
    return outfile


def _build_batch_cases(config, base_path):
    batch = config.get("batch", {}) or {}
    structures = batch.get("structures", {}) or {}
    pattern = structures.get("glob")
    if not pattern:
        raise ConfigError("batch.structures.glob is required when run.mode is 'batch'.")

    resolved_pattern = resolve_path(pattern, base_path)
    paths = [path for path in glob.glob(resolved_pattern) if os.path.isfile(path)]
    paths = _filter_paths(paths, structures.get("include"), keep_matches=True)
    paths = _filter_paths(paths, structures.get("exclude"), keep_matches=False)

    sort_key = structures.get("sort", "name")
    if sort_key == "name":
        paths = sorted(paths, key=lambda item: os.path.basename(item).lower())
    elif sort_key == "path":
        paths = sorted(paths)
    elif sort_key in (None, "none"):
        pass
    else:
        raise ConfigError("batch.structures.sort must be 'name', 'path', or 'none'.")

    return [
        {
            "name": os.path.splitext(os.path.basename(path))[0],
            "structure_path": path,
        }
        for path in paths
    ]


def _filter_paths(paths, patterns, keep_matches):
    if patterns is None:
        return paths
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list):
        raise ConfigError("batch include/exclude patterns must be a string or list.")

    filtered = []
    for path in paths:
        name = os.path.basename(path)
        matched = any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        if matched == keep_matches:
            filtered.append(path)
    return filtered


def _normalize_case(config, raw_case, base_path, execution):
    if not isinstance(raw_case, dict):
        raise ConfigError("Each case must be a mapping.")

    defaults = config.get("defaults", {}) or {}
    merged = _deep_merge(defaults, raw_case)

    solver = merged.get("solver", {}) or {}
    phases = merged.get("phases", {}) or {}
    outputs = merged.get("outputs", {}) or {}
    charge = merged.get("charge", {}) or {}

    solver_type = solver.get("type", "mixed")
    if solver_type != "mixed":
        raise ConfigError("Only solver.type 'mixed' is currently supported.")

    structure_path = merged.get("structure_path")
    if structure_path is None:
        raise ConfigError("Each case requires structure_path.")

    charge_path = charge.get("path", merged.get("charge_path"))
    output_root = merged.get("output_path") or outputs.get("output_path") or _experiment_output_root(config)

    case = {
        "case_name": merged.get("name"),
        "structure_path": resolve_path(structure_path, base_path),
        "charge_path": resolve_path(charge_path, base_path),
        "output_path": resolve_path(output_root, base_path),
        "N": int(solver.get("N", 31)),
        "incre_list": _normalize_increments(solver.get("increments", solver.get("incre_list", [0.1]*10))),
        "preconditioner": solver.get("preconditioner", "green"),
        "diagnostics": bool(solver.get("diagnostics", False)),
        "max_gmres_iter": int(solver.get("max_gmres_iter", 1000)),
        "min_substep_ratio": float(solver.get("min_substep_ratio", 1.0/16.0)),
        "tol_rel": float(solver.get("tol_rel", 1.e-5)),
        "gmres_restart": solver.get("gmres_restart"),
        "reference": solver.get("reference", "mean"),
        "discretization": solver.get("discretization", "fourier"),
        "precond_restrict": bool(solver.get("precond_restrict", True)),
        "forcing": solver.get("forcing", "eisenstat_walker"),
        "inner_rtol": float(solver.get("inner_rtol", 1.e-6)),
        "eta_max": float(solver.get("eta_max", 1.e-2)),
        "eta_min": float(solver.get("eta_min", 1.e-3)),
        "matrix_phase": int(phases.get("matrix_phase", 0)),
        "filler_phase": int(phases.get("filler_phase", 1)),
        "phase_key": phases.get("phase_key", "phase"),
        "phase_path": resolve_path(phases.get("phase_path"), base_path),
        "save_plots": bool(outputs.get("save_plots", True)),
        "plot_dpi": int(outputs.get("plot_dpi", 200)),
        "save_fields": bool(outputs.get("save_fields", False)),
        "field_filename": outputs.get("field_filename", "fields.vti"),
        "log_to_file": bool(merged.get("log_to_file", execution["log_to_file"])),
    }

    output_name = merged.get("output_name") or outputs.get("output_name")
    if output_name is None:
        output_name = _format_case_template(config, case)
    if output_name:
        case["output_name"] = output_name

    if case["charge_path"] is None:
        case.pop("charge_path")
    if case["output_path"] is None:
        case.pop("output_path")
    if case["phase_path"] is None:
        case.pop("phase_path")
    if case["case_name"] is None:
        case.pop("case_name")

    return case


def _normalize_increments(increments):
    if isinstance(increments, dict):
        repeat = int(increments.get("repeat", 0))
        value = increments.get("value")
        if repeat < 1 or value is None:
            raise ConfigError("solver.increments mapping requires repeat >= 1 and value.")
        return [float(value)] * repeat

    if isinstance(increments, list):
        if not increments:
            raise ConfigError("solver.increments list must not be empty.")
        return [float(item) for item in increments]

    raise ConfigError("solver.increments must be a list or a {repeat, value} mapping.")


def _format_case_template(config, case):
    naming = config.get("naming", {}) or {}
    template = naming.get("case_template")
    if not template:
        return None

    structure_stem = os.path.splitext(os.path.basename(case["structure_path"]))[0]
    charge_path = case.get("charge_path")
    charge_stem = os.path.splitext(os.path.basename(charge_path))[0] if charge_path else "default_charge"
    context = {
        "case_name": case.get("case_name") or structure_stem,
        "structure_stem": structure_stem,
        "charge_stem": charge_stem,
        "N": case["N"],
        "preconditioner": case["preconditioner"],
    }
    return template.format(**context)


def _experiment_output_root(config):
    experiment = config.get("experiment", {}) or {}
    return experiment.get("output_root")


def _deep_merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def _load_simple_yaml(text):
    """Parse the small YAML subset used by the run config when PyYAML is absent."""
    tokens = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        raw_line = raw_line.rstrip()
        if not raw_line.strip():
            continue
        stripped_line = _strip_yaml_comment(raw_line)
        if not stripped_line.strip():
            continue
        indent = len(stripped_line) - len(stripped_line.lstrip(" "))
        tokens.append((indent, stripped_line.strip(), line_number))

    if not tokens:
        return {}

    result, index = _parse_yaml_block(tokens, 0, tokens[0][0])
    if index != len(tokens):
        raise ConfigError("Could not parse YAML near line {}.".format(tokens[index][2]))
    return result


def _parse_yaml_block(tokens, index, indent):
    if index >= len(tokens):
        return {}, index
    token_indent, content, line_number = tokens[index]
    if token_indent < indent:
        return {}, index
    if token_indent != indent:
        raise ConfigError("Unexpected indentation near line {}.".format(line_number))
    if content.startswith("- "):
        return _parse_yaml_sequence(tokens, index, indent)
    return _parse_yaml_mapping(tokens, index, indent)


def _parse_yaml_mapping(tokens, index, indent):
    result = {}
    while index < len(tokens):
        token_indent, content, line_number = tokens[index]
        if token_indent < indent:
            break
        if token_indent > indent:
            raise ConfigError("Unexpected indentation near line {}.".format(line_number))
        if content.startswith("- "):
            break

        key, value = _split_yaml_key_value(content, line_number)
        index += 1
        if value == "":
            if index < len(tokens) and tokens[index][0] > indent:
                result[key], index = _parse_yaml_block(tokens, index, tokens[index][0])
            else:
                result[key] = {}
        else:
            result[key] = _parse_yaml_scalar(value)
    return result, index


def _parse_yaml_sequence(tokens, index, indent):
    result = []
    while index < len(tokens):
        token_indent, content, line_number = tokens[index]
        if token_indent < indent:
            break
        if token_indent != indent or not content.startswith("- "):
            break

        value = content[2:].strip()
        index += 1
        if value == "":
            if index < len(tokens) and tokens[index][0] > indent:
                item, index = _parse_yaml_block(tokens, index, tokens[index][0])
            else:
                item = None
        elif _looks_like_yaml_key_value(value):
            key, scalar_value = _split_yaml_key_value(value, line_number)
            item = {key: _parse_yaml_scalar(scalar_value)}
            if index < len(tokens) and tokens[index][0] > indent:
                extra, index = _parse_yaml_block(tokens, index, tokens[index][0])
                if not isinstance(extra, dict):
                    raise ConfigError("Expected mapping after sequence item near line {}.".format(line_number))
                item.update(extra)
        else:
            item = _parse_yaml_scalar(value)
        result.append(item)
    return result, index


def _strip_yaml_comment(line):
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line


def _split_yaml_key_value(content, line_number):
    if ":" not in content:
        raise ConfigError("Expected key/value pair near line {}.".format(line_number))
    key, value = content.split(":", 1)
    key = key.strip()
    if not key:
        raise ConfigError("Empty key near line {}.".format(line_number))
    return key, value.strip()


def _looks_like_yaml_key_value(value):
    if ":" not in value:
        return False
    key, _ = value.split(":", 1)
    return bool(key.strip()) and " " not in key.strip()


def _parse_yaml_scalar(value):
    value = value.strip()
    if value == "":
        return ""
    if value[0] in ("'", '"') and value[-1:] == value[0]:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part) for part in _split_yaml_inline_list(inner)]

    lowered = value.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _split_yaml_inline_list(value):
    parts = []
    current = []
    in_single = False
    in_double = False
    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "," and not in_single and not in_double:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return parts
