#!/usr/bin/env bash
# Runs every comparison on one machine; each script writes its timed runs to results/,
# from which plot.jl draws the figures. Needs Julia, the Python environment of
# requirements.txt, and for the GPU runs a CUDA GPU and the CUDA build of jax
# (pip install "jax[cuda12]"). On a machine with several GPUs, choose one with
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index in nvidia-smi>.
#
# Usage:  PYTHON=venv/bin/python benchmark/compare/run_all.sh [cpu|gpu|all]
#
# CPU (default 2) is the core of every single-core run and CORES (default 0-7) the eight
# cores of the threaded runs. The QDax runs on the CPU take about two hours in all.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
CPU=${CPU:-2}
CORES=${CORES:-0-7}
what=${1:-all}

one=(taskset -c "$CPU")
eight=(taskset -c "$CORES")
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
xla_one_core="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

if [[ $what == cpu || $what == all ]]; then
    "${one[@]}" julia --threads=1 compare.jl
    "${eight[@]}" julia --threads=8 compare.jl
    "${one[@]}" julia --threads=1 compare_cma.jl
    "${one[@]}" julia --threads=1 compare_mome.jl
    "${one[@]}" "$PYTHON" compare.py
    "${one[@]}" "$PYTHON" compare_cma.py
    # JAX_PLATFORMS=cpu keeps a CUDA build of jax on the CPU.
    for x64 in 0 1; do
        JAX_PLATFORMS=cpu JAX_ENABLE_X64=$x64 XLA_FLAGS="$xla_one_core" QDAX_GRID_LOOKUP=1 \
            "${one[@]}" "$PYTHON" compare_qdax.py
        JAX_PLATFORMS=cpu JAX_ENABLE_X64=$x64 XLA_FLAGS="$xla_one_core" \
            "${one[@]}" "$PYTHON" compare_cma_qdax.py
    done
    JAX_PLATFORMS=cpu XLA_FLAGS="$xla_one_core" REPEATS=3 \
        "${one[@]}" "$PYTHON" compare_qdax.py                  # centroid search, about 15 minutes
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS="$xla_one_core" REPEATS=3 \
        "${one[@]}" "$PYTHON" compare_mome_qdax.py             # about 45 minutes
fi

if [[ $what == gpu || $what == all ]]; then
    "${one[@]}" julia gpu/compare_gpu.jl
    "${eight[@]}" julia --threads=8 gpu/compare_cma_gpu.jl
    for x64 in 0 1; do
        for grid in 0 1; do
            XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=$x64 QDAX_GRID_LOOKUP=$grid \
                "${one[@]}" "$PYTHON" compare_qdax.py
        done
    done
fi
