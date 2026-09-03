# -*- coding: utf-8 -*-
"""Run one independent training per GPU, in parallel.

With ~119 samples this uses a multi-GPU node far better than splitting a single
run across it: every GPU trains a whole model on a different split seed, and
the spread across seeds tells you how much of the validation score is real and
how much is the luck of a 19-case validation set.

    python CNN/sweep.py --gpus 0,1,2,3 --seeds 0,1,2,3
    python CNN/sweep.py --gpus 0,1 --seeds 0,1,2,3      # 2 at a time, queued

Each run lands in CNN/runs/<tag>_seed<N>/ exactly as a normal training would.
"""

import argparse
import os
import subprocess
import sys
import time

import config


def launch(seed, gpu, tag, extra):
    """Start one training pinned to a single GPU.

    CUDA_VISIBLE_DEVICES remaps that GPU to cuda:0 inside the child, so
    train.py needs no notion of which physical device it got.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    run_name = "%s_seed%d" % (tag, seed)
    cmd = [sys.executable, os.path.join(config.CNN_DIR, "train.py"),
           "--seed", str(seed), "--run-name", run_name,
           "--device", "cuda"] + extra
    log_path = os.path.join(config.RUN_DIR, run_name + ".log")
    os.makedirs(config.RUN_DIR, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    print("gpu %s -> seed %d  (%s)" % (gpu, seed, log_path))
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"proc": proc, "log": log, "seed": seed, "gpu": gpu,
            "name": run_name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0",
                        help="comma-separated physical GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3",
                        help="comma-separated split seeds, one run each")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="everything after -- is passed to train.py")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    extra = [a for a in args.rest if a != "--"]

    print("%d runs over %d GPUs" % (len(seeds), len(gpus)))
    pending = list(seeds)
    running, failed = [], []

    # Keep every GPU busy: start a new seed as soon as one frees up.
    while pending or running:
        free = [g for g in gpus if g not in {r["gpu"] for r in running}]
        while pending and free:
            running.append(launch(pending.pop(0), free.pop(0), args.tag, extra))

        time.sleep(5)
        for run in list(running):
            if run["proc"].poll() is None:
                continue
            run["log"].close()
            running.remove(run)
            code = run["proc"].returncode
            print("%s finished with exit code %d" % (run["name"], code))
            if code != 0:
                failed.append(run["name"])

    print("\nall runs done")
    if failed:
        print("failed: %s" % ", ".join(failed))
        raise SystemExit(1)
    print("best.pt and history.csv per run are under %s" % config.RUN_DIR)


if __name__ == "__main__":
    main()
