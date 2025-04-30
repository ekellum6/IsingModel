#!/usr/bin/env python
"""
ising_pool_driver.py – run Ising simulations in parallel *without* MPI.

Each temperature is handled by one worker process in a multiprocessing Pool.
Results are gathered into a pandas DataFrame and written to CSV.

Usage (inside a SLURM job with e.g. 32 CPUs on one node):

    python ising_pool_driver.py \
        --dim 3 --L 100 \
        --tmin 1.0 --tmax 8.0 --nT 200 \
        --steps 8000 --eq 3000 \
        --outfile results_cpu.csv

If you omit --outfile the script prints the DataFrame to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import isingModelCPU as isingModel

# -----------------------------------------------------------------------------


def _simulate_one_temperature(args: tuple) -> tuple[float, float, float, float]:
    """
    Worker function executed in each process.

    Returns:
      (T, E_per_spin, C_per_spin, |M|_per_spin)
    """
    T, model_kwargs, sim_kwargs = args
    # create a fresh model *inside* the worker
    model = isingModel.IsingModel(**model_kwargs, T=T)
    energies, mags = model.run(**sim_kwargs)

    N = model.L ** model.dim
    beta = 1.0 / T

    e_avg = np.mean(energies) / N
    c = (beta**2 / N) * (np.mean(energies**2) - np.mean(energies) ** 2)
    m_avg = np.mean(np.abs(mags)) / N

    return T, e_avg, c, m_avg


# -----------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiprocessing Ising sweep")
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--tmin", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=5.0)
    p.add_argument("--nT", type=int, default=20)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eq", type=int, default=1000)
    p.add_argument("--outfile", default="")
    p.add_argument(
        "--nprocs",
        type=int,
        default=cpu_count(),
        help="worker processes (default = all CPUs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    temperatures = np.linspace(args.tmin, args.tmax, args.nT)

    model_kwargs = dict(L=args.L, dim=args.dim, J=args.J, seed=None)
    sim_kwargs = dict(n_steps=args.steps, equilibration_steps=args.eq)

    work = [(float(T), model_kwargs, sim_kwargs) for T in temperatures]

    print(
        f"[{time.ctime()}] launching pool: {args.nprocs} workers, "
        f"{len(work)} temperatures"
    )

    with Pool(processes=args.nprocs) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(_simulate_one_temperature, work), 1):
            results.append(result)
            if i % max(1, args.nT // 50) == 0 or i == args.nT:
                print(f"[{time.ctime()}] Progress: {i}/{args.nT} temperatures completed", flush=True)

    df = pd.DataFrame(
        results, columns=["T", "E_per_spin", "C_per_spin", "M_abs_per_spin"]
    ).sort_values("T")

    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        df.to_csv(args.outfile, index=False)
        print(f"[{time.ctime()}] wrote results → {args.outfile}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
