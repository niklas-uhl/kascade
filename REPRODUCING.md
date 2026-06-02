This artifact for reproducing the results of the paper "Engineering Scalable Distributed List Ranking" is also available as a source archive (GitHub repository:
<https://github.com/niklas-uhl/kascade>) at

::: center
<https://github.com/niklas-uhl/kascade/releases/tag/europar-artifact>
:::

The archive contains the full source of the benchmark suite including
all reproducibility scripts. Alternatively, clone the repository
directly (see [1.3](#sec:setup){reference-type="ref+label"
reference="sec:setup"}).

This paper has one artifact, the `kascade` benchmark suite, consisting
of the list ranking library, containing all algorithm variants described
in the paper, and the corresponding benchmark runner. The benchmark
binary `kascade` allows selecting an algorithm and input configuration
to execute algorithm on, and reports running time, which is the main
experimental output of our work. The artifact is implemented in C++. For
executing the benchmarks using different algorithm configurations and
inputs, we use `kaval`[^1], a Python-based experiment runner for
distributed algorithm engineering; it is fetched automatically into the
`kaval/` directory during the CMake configure step. `kaval` reads YAML
experiment suite descriptions (located in
`experiments/reproducibility/`), which specify the benchmark binary,
input graphs, core counts, and other parameters. From these descriptions
it generates and executes the benchmark jobs, writing output logs to the
output directory. On a shared-memory machine jobs are run directly; on
an HPC cluster, `kaval` can be configured to emit SLURM job files from
user-provided templates.

For parsing the output files and generating plots, we use a Julia-based
evaluation pipeline.

Since we evaluate the scalability of the algorithms on up to 24 576
cores, for fully reproducing the results, access to an HPC system is
required. All steps for reproducing allow adjusting the maximum number
of cores used, so the experiments can be easily downscaled to either run
on a large shared memory machine, or an HPC system. We suggest using a
system with at least 64 physical cores. At this scale, correctness and
the relative ordering of algorithm variants (Figures 2 and 3) can be
verified. For observing the indirection benefits (Figures 4 and 5) and
the full scalability range (Figure 3), at least 1 000 cores on an HPC
system are required; these effects are driven by network latencies only
present in multi-node HPC environments.

# Getting Started Guide {#sec:gett-start-guide}

## Platform

The experiments in the paper have been performed on SuperMUC-NG, running
SUSE Linux Enterprise Server (SLES), but we successfully executed
down-scaled experiments on HoreKa[^4] (running RHEL 9.4) and a shared
memory machine using an AMD EPYC 9684X 96-Core with 2 threads each
(running Rocky Linux 9.4).

The artifact runs as standard user-level software; no root access is
required. The required system dependencies (compiler, MPI, CMake) are
standard packages available on any Linux HPC environment. Python and
Julia dependencies are fully isolated via `uv` and the Julia project
environment respectively, requiring no system-wide installation.

## Dependencies

The following dependencies must be installed manually on the system
where experiments are executed. Most C++ library dependencies are
fetched automatically at build time via CMake's FetchContent; see
[3](#sec:full-deps){reference-type="ref+label"
reference="sec:full-deps"} for a complete list. Outbound network access
is therefore required on the machine where `cmake --preset experiments`
is run.

- GNU `g++` 15.1.0 (minimum required 14)

- Intel MPI 2021.15.0 (or another MPI implementation supporting MPI-3)

- CMake 3.25 (or higher)

- Python 3.13 (minimum required 3.10), with `pyyaml` 6.0.3. We recommend
  `uv`[^5], a fast Python package manager. Install it with:

      curl -LsSf https://astral.sh/uv/install.sh | sh

  Run `uv sync` once during setup (see below) to install Python
  dependencies into an isolated environment; experiment commands use
  `uv run` to invoke scripts within that environment. Alternatively,
  install `pyyaml` manually (`pip install pyyaml`) and invoke scripts
  with `python` instead of `uv run`.

Generating plots additionally requires Julia 1.11 on the machine used
for evaluation. We recommend installing Julia via `juliaup`[^6], a Julia
version manager:

    curl -fsSL https://install.julialang.org | sh

Julia package dependencies are managed via the project environment in
`eval/` and can be installed automatically (see
[1.3](#sec:setup){reference-type="ref+label" reference="sec:setup"}).

## Setup {#sec:setup}

All commands should be run from the repository root. We provide
`reproducibility.justfile` with recipes for all steps below; default
parameters such as core counts can be adjusted at the top of that file.
Install `just`[^7] via your system's package manager; it is also
available via `uv run just` once `uv sync` has been run. Alternatively,
copy the commands from the file directly without installing `just`. Each
step in [2](#sec:step-step-instr){reference-type="ref+label"
reference="sec:step-step-instr"} lists the corresponding `just` recipe
in [blue]{style="color: blue"} as a shorthand.

1.  Install the compiler, MPI, and CMake using the system's package
    manager.

2.  Optionally install `uv` and `just` as described in this section.

3.  Clone the repository:

        git clone https://github.com/niklas-uhl/kascade
        cd kascade

4.  On the *experiment machine*, configure and build the benchmark
    binary and sync Python dependencies:

    ``` {style="just"}
    just setup-experiments
    ```

    Or manually:

        cmake --preset experiments
        cmake --build --preset experiments --parallel
        uv sync

5.  On the *eval machine* (may be the same), instantiate the Julia plot
    environment:

    ``` {style="just"}
    just setup-eval
    ```

    Or manually:

        julia --project=eval -e "using Pkg; Pkg.instantiate()"

    If both machines are the same, run `just setup` instead of the two
    steps above.

6.  Verify the build and MPI setup:

        build/bench/kascade --help
        mpiexec -n 2 ./build/bench/kascade \
          --kagen_option_string="type=path;N=12" \
          --input-processing none \
          --algorithm SparseRulingSet \
          --verify-level 1 --output-file out.json

By default, experiments run on a shared-memory machine using up to 64
cores. Adjust `max-cores` to match the available hardware, e.g.:

    just max-cores=16 run-all

For running on a SLURM cluster, see
[4](#sec:custom-system){reference-type="ref+label"
reference="sec:custom-system"}.

# Step-by-Step Instructions {#sec:step-step-instr}

All inputs are generated synthetically on-the-fly by KaGen[^8], a
communication-free distributed graph generator for different graph
families, bundled with the benchmark binary; no external datasets need
to be downloaded. Experiment parameters (input sizes, algorithm
configurations, iteration counts) are defined in the YAML suite files in
`experiments/reproducibility/`; refer to the paper for the rationale
behind each choice. The `--no-date-suffix` flag passed to `kaval`
ensures output is written to a fixed directory name; without it, `kaval`
appends a timestamp, which would cause the plot scripts to not find the
data.

On a shared-memory machine with up to 64 cores, all four experiments
complete in approximately 40 minutes when run one after another. On
HoreKa with up to 1 216 cores (`cores=node-size-pow2`), all four
experiments submitted a total of 85 SLURM jobs; individual jobs
completed in under 5 minutes, and from the first job starting to the
last finishing took approximately 23 minutes. Queue wait time adds to
this and depends on cluster load.

All experiments write JSON result files to
`repro-out/data/<experiment-name>/` and plots to `repro-out/plots/`. The
plotting scripts only require the JSON files, so experiments and
plotting can be performed on different machines (e.g., running
benchmarks on an HPC cluster and generating plots locally after
transferring the `repro-out/data/` directory). The Julia plot
environment only needs to be instantiated once; if `just setup-eval` was
not run on the plotting machine, run `just setup-eval` first (installs
Julia package dependencies).

Each subsection below lists the *run* step (experiment machine) and the
*plot* step (eval machine) separately. The manual commands shown use
`--machine shared` and `--max-cores 64`; adjust these to match your
system as described in
[4](#sec:custom-system){reference-type="ref+label"
reference="sec:custom-system"}. Run the plot step only after the
corresponding experiment data has been collected and transferred to the
eval machine.

All steps can be run together with:

``` {style="just"}
just run-all
just plot
```

or individually as described below.

## Locality (Figure 2) {#sec:locality}

Evaluates locality-aware techniques (Plain, LocalChasing,
LocalContraction) on randomly permuted path graphs with varying locality
parameter $\gamma$.

``` {style="just"}
just run-locality
just plot-locality
```

    uv run kaval/run-experiments.py \
      --search-dirs ./experiments/reproducibility/ \
      --machine shared --cores pow2 \
      --min-cores 1 --max-cores 64 \
      sparse-ruling-set-locality \
      --experiment-data-dir repro-out/data --no-date-suffix
    julia --project=eval eval/reproducibility/locality_plot.jl \
      repro-out/data/sparse-ruling-set-locality \
      --output repro-out/plots/locality_plot.pdf

*Output:* `repro-out/plots/locality_plot.pdf`. The plot shows four
panels for $\gamma \in \{0, 0.01, 0.1, 1.0\}$, where $\gamma$ is the
permutation probability ($\gamma = 0$: fully local, $\gamma = 1$: fully
random). At high locality ($\gamma = 0$), the ordering Plain $>$
LocalChasing $>$ LocalContraction (slowest to fastest in terms of
running time) should be clearly visible; the gap between variants
decreases with increasing $\gamma$, until all variants perform similarly
at $\gamma = 1.0$ (fully random). Absolute times will differ from the
paper (which uses up to 24 576 cores on SuperMUC-NG), but the relative
ordering of variants should be preserved.

## Scalability (Figure 3) {#sec:scalability}

Compares pointer doubling (PD) and sparse ruling-set (SRS) with direct
communication and topology-aware indirect communication (+Ind) across
list sizes $2^{16}$--$2^{22}$ and two Euler tour instances (GNM, RGG2D)
constructed from random graphs.

``` {style="just"}
just run-scalability
just plot-scalability
```

    uv run kaval/run-experiments.py \
      --search-dirs ./experiments/reproducibility/ \
      --machine shared --cores pow2 \
      --min-cores 1 --max-cores 64 \
      pointer-doubling \
      --experiment-data-dir repro-out/data --no-date-suffix
    uv run kaval/run-experiments.py \
      --search-dirs ./experiments/reproducibility/ \
      --machine shared --cores pow2 \
      --min-cores 1 --max-cores 64 \
      sparse-ruling-set \
      --experiment-data-dir repro-out/data --no-date-suffix
    julia --project=eval eval/reproducibility/scalability_plot.jl \
      repro-out/data/sparse-ruling-set \
      repro-out/data/pointer-doubling \
      --output repro-out/plots/scalability_plot.pdf

*Output:* `repro-out/plots/scalability_plot.pdf`. SRS variants should be
consistently faster than PD at larger core counts; the gap widens with
list size. At low core counts, direct SRS should outperform SRS+Ind, as
indirection introduces overhead; with increasing core count, as network
latencies dominate, SRS+Ind should catch up and eventually surpass
direct SRS. This switching point is related to message startup overheads
and is therefore system dependent; it might not be observable on shared
memory machines and with fewer than 1 000 cores. The switching point
should happen at higher core counts with increasing list size.

## Indirection (Figures 4 and 5) {#sec:indirection}

Evaluates the impact of message indirection schemes (Direct, 2D-grid,
topology-aware) on a large randomly permuted path ($2^{22}$ elements per
rank), including a phase breakdown.

``` {style="just"}
just run-indirection
just plot-indirection-line
just plot-indirection-bar
```

    uv run kaval/run-experiments.py \
      --search-dirs ./experiments/reproducibility/ \
      --machine shared --cores pow2 \
      --min-cores 1 --max-cores 64 \
      sparse-ruling-set-indirection \
      --experiment-data-dir repro-out/data --no-date-suffix
    julia --project=eval \
      eval/reproducibility/indirection_line_plot.jl \
      repro-out/data/sparse-ruling-set-indirection \
      --output repro-out/plots/indirection_line_plot.pdf
    julia --project=eval \
      eval/reproducibility/indirection_bar_plot.jl \
      repro-out/data/sparse-ruling-set-indirection \
      --output repro-out/plots/indirection_bar_plot.pdf

*Output:* `repro-out/plots/indirection_line_plot.pdf` and
`indirection_bar_plot.pdf`. At small core counts, direct communication
should outperform indirection variants, as indirection introduces
overhead; based on our experiments on different systems, topology-aware
and 2D-grid indirection outperform direct communication only on
distributed HPC systems with at least 1 000 cores. The bar plot should
show that ruler propagation and ruler chasing are the dominant phases;
on large core counts indirect communication improves the running time of
these phases.

# Full Dependency List {#sec:full-deps}

## C++ (fetched via CMake FetchContent)

#### library dependencies

- [fmtlib](https://github.com/fmtlib/fmt) v11.0.2

- [spdlog](https://github.com/gabime/spdlog) v1.16.0

- [kassert](https://github.com/kamping-site/kassert) v1.0.0

- [KaMPIng](https://github.com/kamping-site/kamping) v0.2.0

- [Abseil](https://github.com/abseil/abseil-cpp) 20250814.0

- [KaGen](https://github.com/KarlsruheGraphGeneration/KaGen) tag
  `kascade-partial-path-permutation-v1.0`

- [BriefKAsten](https://github.com/niklas-uhl/briefkasten) v0.2.1

#### CLI dependencies

- [cmake_git_version_tracking](https://github.com/niklas-uhl/cmake-git-version-tracking)
  v1.0.0

- [nlohmann_json](https://github.com/nlohmann/json) v3.12.0

- [CLI11](https://github.com/CLIUtils/CLI11) v2.5.0

- [kamping-spdlog-adapter](https://github.com/kamping-site/kamping-spdlog-adapter)
  v1.0.0

- [kamping-nlohmann-json-adapter](https://github.com/kamping-site/kamping-nlohmann-json-adapter)
  v1.0.0

## Experiment Runner

[kaval](https://github.com/niklas-uhl/kaval) 20260505.0 (fetched
automatically via CMake)

## Python

Python 3.13 (minimum required 3.10)

- [pyyaml](https://pypi.org/project/PyYAML/) 6.0.3

## Julia

Julia 1.11.6

- [AlgebraOfGraphics](https://juliapackages.com/p/AlgebraOfGraphics)
  v0.12.1

- [ArgParse](https://juliapackages.com/p/ArgParse) v1.2.0

- [CSV](https://juliapackages.com/p/CSV) v0.10.16

- [CairoMakie](https://juliapackages.com/p/Makie) v0.15.8

- [CategoricalArrays](https://juliapackages.com/p/CategoricalArrays)
  v1.0.2

- [DataFrames](https://juliapackages.com/p/DataFrames) v1.8.1

- [DataFramesMeta](https://juliapackages.com/p/DataFramesMeta) v0.15.6

- [Glob](https://juliapackages.com/p/Glob) v1.4.0

- [JSON](https://juliapackages.com/p/JSON) v1.4.0

- [LaTeXStrings](https://juliapackages.com/p/LaTeXStrings) v1.4.0

- [Makie](https://juliapackages.com/p/Makie) v0.24.8

# Running on a Custom System {#sec:custom-system}

By default, `reproducibility.justfile` targets a shared-memory machine
(`machine=shared`) and scales core counts as powers of two from 1 to 64.
The variables shown in [1](#tab:vars){reference-type="ref+label"
reference="tab:vars"} are defined at the top of the file and can be
edited there directly, or overridden on the command line.

::: {#tab:vars}
  Variable                                                                     Description
  ---------------------------------------------------------------------------- ------------------------------------------------------
  `machine`                                                                    `shared` (direct) or `generic-job-file` (SLURM)
  `cores`                                                                      
  (use $2^0, 2^1, \ldots$ or $k \times 2^0, k \times 2^1, \ldots$ as #cores)   
                                                                               
  `max-cores`                                                                  
  the inclusive core count range                                               
  `extra_args`                                                                 extra arguments forwarded to `kaval`
  `time-limit`                                                                 per-configuration time limit in minutes (default: 5)

  : Variables for controlling execution in `reproducibility.justfile`.
:::

The required compiler, MPI, and CMake are typically provided as
environment modules on HPC systems; load them before building. For
example, on SuperMUC-NG:

    module load cmake
    module load gcc/15.1.0
    module load intel-mpi

The exact module names vary by system.

For a SLURM cluster, adapt the two template files in
`experiments/reproducibility/`: `sbatch-template-example.txt` controls
the job script header (partitions, modules, constraints) and
`command-template-example.txt` controls how each individual run is
launched (MPI launcher, flags). As a starting point, `kaval` ships
ready-to-use templates for HoreKa and SuperMUC-NG under
`kaval/sbatch-templates/` and `kaval/command-templates/`; copy and adapt
the closest match to your system. Then set the following variables at
the top of `reproducibility.justfile`:

    machine    := "generic-job-file"
    cores      := "node-size-pow2"
    max-cores  := "2048"
    # helper variables for readability
    sbatch-tmpl  := "experiments/reproducibility/sbatch-template.txt"
    command-tmpl := "experiments/reproducibility/command-template.txt"
    extra_args   := "--sbatch-template " + sbatch-tmpl \
                  + " --command-template " + command-tmpl \
                  + " --tasks-per-node <N>"

Replace `<N>` with the number of physical cores per node. This is
required when using `cores=node-size-pow2`, which generates core counts
as multiples of the node size ($N, 2N, 4N, \ldots$). Variables can also
be overridden on the command line without editing the file, e.g.:

    just machine=generic-job-file max-cores=2048 run-all

[^1]: <https://github.com/niklas-uhl/kaval>

[^2]: <https://doku.lrz.de/supermuc-ng-10745965.html>

[^4]: <https://www.nhr.kit.edu/userdocs/horeka/>

[^5]: <https://astral.sh/uv>

[^6]: <https://github.com/JuliaLang/juliaup>

[^7]: <https://just.systems>

[^8]: <https://github.com/KarlsruheGraphGeneration/KaGen>
