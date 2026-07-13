# -*- coding: utf-8 -*-
"""YAML configuration helpers for the microstructure generation runner.

Loads a generation config (see generation_phr_sweep.yaml) and expands it into a
flat list of per-structure parameter dicts ("specs") that
combined_particle_models.generate_and_save can consume directly.
"""

import copy
import itertools
import math

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


SUPPORTED_MODES = ("sweep", "cases")
SUPPORTED_SEED_STRATEGIES = ("sequential", "fixed", "random")
SUPPORTED_ON_EXISTING = ("skip", "overwrite", "error")


class ConfigError(ValueError):
    """Raised when a generation YAML file is invalid."""


def load_config(path):
    """Load and validate a generation YAML config."""
    if yaml is None:
        raise ConfigError("PyYAML is required to read generation configs. Install pyyaml.")

    with open(path) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")
    if config.get("version") != 1:
        raise ConfigError("Unsupported config version {!r}; expected version 1.".format(config.get("version")))
    if not config.get("output_dir"):
        raise ConfigError("Config requires an 'output_dir'.")
    return config


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
        "on_existing": on_existing,
        "save_manifest": bool(execution.get("save_manifest", True)),
    }


def get_phr_settings(config):
    """Return densities used for the recorded PHR."""
    phr = config.get("phr", {}) or {}
    return {
        "filler_density": float(phr.get("filler_density", 1.8)),
        "rubber_density": float(phr.get("rubber_density", 0.92)),
    }


def build_specs(config, mode_override=None):
    """Expand the config into a list of per-structure parameter dicts."""
    mode = mode_override or (config.get("run", {}) or {}).get("mode", "sweep")
    if mode not in SUPPORTED_MODES:
        raise ConfigError("run.mode must be one of {}; got {!r}.".format(SUPPORTED_MODES, mode))

    defaults = config.get("defaults", {}) or {}
    output_dir = config["output_dir"]

    if mode == "sweep":
        specs = _build_sweep_specs(config, defaults)
    else:
        specs = _build_case_specs(config, defaults)

    for spec in specs:
        spec["output_dir"] = output_dir

    _check_unique_names(specs)
    return specs


# ---------------------------------------------------------------------------
# sweep expansion
# ---------------------------------------------------------------------------
def _build_sweep_specs(config, defaults):
    sweep = config.get("sweep", {}) or {}
    axes = sweep.get("axes", {}) or {}
    if not axes:
        raise ConfigError("sweep.axes must define at least one axis.")

    axis_keys = list(axes.keys())
    axis_values = [_expand_axis(key, axes[key]) for key in axis_keys]

    replicates = int(sweep.get("replicates", 1))
    if replicates < 1:
        raise ConfigError("sweep.replicates must be at least 1.")

    seed_cfg = sweep.get("seed", {}) or {}
    strategy = seed_cfg.get("strategy", "sequential")
    if strategy not in SUPPORTED_SEED_STRATEGIES:
        raise ConfigError(
            "sweep.seed.strategy must be one of {}; got {!r}."
            .format(SUPPORTED_SEED_STRATEGIES, strategy)
        )
    start_seed = int(seed_cfg.get("start_seed", 0))
    seed_rng = _make_seed_rng()

    template = ((sweep.get("naming", {}) or {}).get("template")) or "{index:04d}"

    specs = []
    global_index = 0
    for combo in itertools.product(*axis_values):
        combo_map = dict(zip(axis_keys, combo))
        for sample in range(replicates):
            params = copy.deepcopy(defaults)
            for key, value in combo_map.items():
                _set_dotted(params, key, value)

            seed = _resolve_seed(strategy, start_seed, global_index, seed_rng)
            params["seed"] = seed
            # The final name is resolved after generation (it may embed {phr}).
            params["name_template"] = template
            params["name_context"] = _name_context(combo_map, sample, global_index, seed)
            specs.append(params)
            global_index += 1
    return specs


def _build_case_specs(config, defaults):
    cases = config.get("cases", []) or []
    if not isinstance(cases, list):
        raise ConfigError("cases must be a list.")

    specs = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ConfigError("Each case must be a mapping.")
        params = _deep_merge(defaults, case)
        params.setdefault("name", "case_{:04d}".format(index))
        params.setdefault("seed", None)
        specs.append(params)
    return specs


# ---------------------------------------------------------------------------
# axis / seed / naming helpers
# ---------------------------------------------------------------------------
def _expand_axis(key, value):
    """Turn an axis spec into a concrete list of values."""
    if isinstance(value, dict):
        missing = [k for k in ("start", "stop", "step") if k not in value]
        if missing:
            raise ConfigError("Axis {!r} range is missing {}.".format(key, missing))
        return _expand_range(value["start"], value["stop"], value["step"], key)
    if isinstance(value, list):
        if not value:
            raise ConfigError("Axis {!r} list must not be empty.".format(key))
        return [_clean_number(v) for v in value]
    raise ConfigError(
        "Axis {!r} must be a list or a {{start, stop, step}} mapping.".format(key)
    )


def _expand_range(start, stop, step, key):
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        raise ConfigError("Axis {!r} step must be non-zero.".format(key))
    if (stop - start) * step < 0:
        raise ConfigError("Axis {!r} step points away from stop.".format(key))

    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [_clean_number(start + i * step) for i in range(count)]


def _clean_number(value):
    """Round float drift and collapse integral floats to int for tidy names."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    rounded = round(value, 10)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _make_seed_rng():
    import random

    return random.Random()


def _resolve_seed(strategy, start_seed, global_index, seed_rng):
    if strategy == "sequential":
        return start_seed + global_index
    if strategy == "fixed":
        return start_seed
    return seed_rng.randint(0, 2 ** 31 - 1)


def _name_context(combo_map, sample, global_index, seed):
    """Placeholders available to naming.template (except {phr}, added at save time)."""
    context = {
        "sample": sample + 1,
        "index": global_index + 1,
        "seed": seed,
    }
    if combo_map:
        first_key = next(iter(combo_map))
        context["value"] = combo_map[first_key]
    for key, value in combo_map.items():
        context[key.replace(".", "_")] = value
    return context


# ---------------------------------------------------------------------------
# generic dict helpers
# ---------------------------------------------------------------------------
def _set_dotted(target, dotted_key, value):
    parts = dotted_key.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _deep_merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _check_unique_names(specs):
    """Ensure names are unique ignoring {phr} (unknown until after generation).

    {phr} is treated as a wildcard, so the non-PHR part of every name must be
    unique on its own -- otherwise two structures with the same rounded PHR
    would collide.
    """
    seen = set()
    for spec in specs:
        key = _name_check_key(spec)
        if key in seen:
            raise ConfigError(
                "Structure name {!r} is not unique once {{phr}} is ignored. Make "
                "naming.template unique (e.g. add {{index}} or {{sample}}).".format(key)
            )
        seen.add(key)


def preview_name(spec):
    """Human-readable name for dry runs (PHR shown as <phr>, resolved at run time)."""
    template = spec.get("name_template")
    if template is None:
        return spec.get("name")
    context = dict(spec.get("name_context", {}))
    context["phr"] = "<phr>"
    return template.format(**context)


def _name_check_key(spec):
    template = spec.get("name_template")
    if template is None:
        return spec.get("name")
    context = dict(spec.get("name_context", {}))
    context["phr"] = "{phr}"  # leave the PHR placeholder intact as a wildcard
    try:
        return template.format(**context)
    except (KeyError, IndexError) as exc:
        raise ConfigError(
            "naming.template references unknown placeholder {}. Available: {} (plus phr)."
            .format(exc, sorted(k for k in context if k != "phr"))
        )
