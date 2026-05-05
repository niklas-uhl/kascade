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
experiment-out := justfile_directory() / "repro-out/data"
time-limit := "5"
extra_args := ""

run-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/reproducibility/ --machine {{machine}} --cores {{cores}} --min-cores {{min-cores}} --max-cores {{max-cores}} --time-limit {{time-limit}} {{experiment}} --experiment-data-dir {{experiment-out}} --no-date-suffix {{extra_args}} {{args}}

run-locality: (run-experiment "sparse-ruling-set-locality")
run-scalability: (run-experiment "pointer-doubling") (run-experiment "sparse-ruling-set")
run-indirection: (run-experiment "sparse-ruling-set-indirection")
run-all: run-locality run-scalability run-indirection

julia := "julia --project=eval"

plot-instantiate:
    {{julia}} -e "using Pkg; Pkg.instantiate()"

plot-locality output="repro-out/plots/locality_plot.pdf":
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/locality_plot.jl {{experiment-out}}/sparse-ruling-set-locality --output {{output}}

plot-scalability output="repro-out/plots/scalability_plot.pdf":
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/scalability_plot.jl {{experiment-out}}/sparse-ruling-set {{experiment-out}}/pointer-doubling --output {{output}}

plot-indirection-scatter output="repro-out/plots/indirection_scatter_plot.pdf":
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/indirection_scatter_plot.jl {{experiment-out}}/sparse-ruling-set-indirection --output {{output}}

plot-indirection-bar output="repro-out/plots/indirection_bar_plot.pdf":
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/indirection_bar_plot.jl {{experiment-out}}/sparse-ruling-set-indirection --output {{output}}

plot: plot-locality plot-scalability plot-indirection-scatter plot-indirection-bar

configure:
    cmake --preset experiments

build:
    cmake --build --preset experiments --parallel

setup: configure build plot-instantiate
