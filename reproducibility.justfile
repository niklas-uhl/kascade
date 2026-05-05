# -*- mode: just-ts; -*-
# Reproducibility recipes for the paper experiments.  
# See REPRODUCING.md for step-by-step instructions.

# Defaults run locally on a shared machine.
# To run on a Slurm cluster, override machine/cores/min/max-cores and provide both templates, e.g.:
#   just machine=generic-job-file cores=node-size-pow2 min-cores=64 max-cores=2048 \
#        extra_args="--sbatch-template experiments/reproducibility/sbatch-template-example.txt \
#                    --command-template experiments/reproducibility/command-template-example.txt" \
#        run-all
# Adapt both template files to your cluster (MPI launcher, partition, constraints, etc.).
machine := "shared"
cores := "pow2"
min-cores := "1"
max-cores := "64"
experiment-out := "experiment-out"
extra_args := ""

run-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/reproducibility/ --machine {{machine}} --cores {{cores}} --min-cores {{min-cores}} --max-cores {{max-cores}} {{experiment}} --experiment-data-dir {{experiment-out}} --no-date-suffix {{extra_args}} {{args}}

run-locality: (run-experiment "sparse-ruling-set-locality")
run-scalability: (run-experiment "pointer-doubling") (run-experiment "sparse-ruling-set")
run-indirection: (run-experiment "sparse-ruling-set-indirection")
run-all: run-locality run-scalability run-indirection

julia := "julia --project=eval"

plot-instantiate:
    {{julia}} -e "using Pkg; Pkg.instantiate()"

plot-locality output="locality_plot.pdf":
    {{julia}} eval/locality_plot.jl {{experiment-out}}/sparse-ruling-set-locality {{output}}

plot-scalability output="scalability_plot.pdf":
    {{julia}} eval/scalability_plot.jl {{experiment-out}} {{output}}

plot-indirection-scatter output="indirection_scatter_plot.pdf":
    {{julia}} eval/indirection_scatter_plot.jl {{experiment-out}} {{output}}

plot-indirection-bar output="indirection_bar_plot.pdf":
    {{julia}} eval/indirection_bar_plot.jl {{experiment-out}} {{output}}

plot: plot-locality plot-scalability plot-indirection-scatter plot-indirection-bar

configure:
    cmake --preset experiments

build:
    cmake --build --preset experiments --parallel

setup: configure build plot-instantiate
