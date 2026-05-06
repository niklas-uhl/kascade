# -*- mode: just-ts; -*-
# Reproducibility recipes for the paper experiments.
# See the artifact overview document for step-by-step instructions.
#
# Default targets a shared-memory machine (machine=shared, up to 64 cores).
# Edit the variables below or override them on the command line, e.g.:
#   just max-cores=16 run-all
#
# For SLURM clusters, set machine=generic-job-file, provide adapted copies of the
# example templates in experiments/reproducibility/, and set extra_args accordingly.
# kaval ships HoreKa and SuperMUC-NG templates under kaval/sbatch-templates/ and
# kaval/command-templates/ as a starting point.
# See the artifact overview document (sec. "Running on a Custom System") for details.
machine := "shared"
cores := "pow2"
min-cores := "1"
max-cores := "64"
experiment-out := justfile_directory() / "repro-out/data"
plots-out := experiment-out / "../plots"
locality-plot-out := plots-out / "locality_plot.pdf"
scalability-plot-out := plots-out / "scalability_plot.pdf"
indirection-line-plot-out := plots-out / "indirection_line_plot.pdf"
indirection-bar-plot-out := plots-out / "indirection_bar_plot.pdf"
time-limit := "5" # time limit per config in minutes
extra_args := ""

run-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/reproducibility/ --machine {{machine}} --cores {{cores}} --min-cores {{min-cores}} --max-cores {{max-cores}} --time-limit {{time-limit}} {{experiment}} --experiment-data-dir {{experiment-out}} --no-date-suffix {{extra_args}} {{args}}

run-locality: (run-experiment "sparse-ruling-set-locality")
run-scalability: (run-experiment "pointer-doubling") (run-experiment "sparse-ruling-set")
run-indirection: (run-experiment "sparse-ruling-set-indirection")
run-all: run-locality run-scalability run-indirection

julia := "julia --project=eval"

# The plotting targets assume that the run targets have been executed and the output lives in {{experiment-out}}
plot-instantiate:
    {{julia}} -e "using Pkg; Pkg.instantiate()"

plot-locality output=locality-plot-out:
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/locality_plot.jl {{experiment-out}}/sparse-ruling-set-locality --output {{output}}

plot-scalability output=scalability-plot-out:
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/scalability_plot.jl {{experiment-out}}/sparse-ruling-set {{experiment-out}}/pointer-doubling --output {{output}}

plot-indirection-line output=indirection-line-plot-out:
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/indirection_line_plot.jl {{experiment-out}}/sparse-ruling-set-indirection --output {{output}}

plot-indirection-bar output=indirection-bar-plot-out:
    mkdir -p $(dirname {{output}})
    {{julia}} eval/reproducibility/indirection_bar_plot.jl {{experiment-out}}/sparse-ruling-set-indirection --output {{output}}

plot: plot-locality plot-scalability plot-indirection-line plot-indirection-bar

configure:
    cmake --preset experiments

build:
    cmake --build --preset experiments --parallel

uv-sync:
    uv sync

# Run on the experiment machine: compile the benchmark binary and sync Python deps
setup-experiments: configure build uv-sync

# Run on the eval machine: install Julia plot dependencies
setup-eval: plot-instantiate

# Convenience: run both setup steps (when experiment and eval machine are the same)
setup: setup-experiments setup-eval
