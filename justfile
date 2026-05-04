set dotenv-load := true

run-supermuc-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine supermuc --module-config kascade --cores node-size-pow2 --min-cores 48 --max-cores 2048 {{ experiment }} {{ args }}

run-horeka-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine horeka --module-config kascade --command-template ./kaval/command-templates/horeka-IntelMPI.txt --cores node-size-pow2 --min-cores 64 --max-cores 2048  {{ experiment }} {{ args }}

min-cores := "1"
max-cores := "64"

run-locality-experiment experiment="sparse-ruling-set-locality" +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/reproducibility/ --cores pow2 --min-cores {{min-cores}} --max-cores {{max-cores}}  {{ experiment }} {{ args }} --experiment-data-dir experiment-out --no-date-suffix

run-scalability-experiment experiment="pointer-doubling sparse-ruling-set" +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/reproducibility/ --cores pow2 --min-cores {{min-cores}} --max-cores {{max-cores}}  {{ experiment }} {{ args }} --experiment-data-dir experiment-out --no-date-suffix

plot-locality experiment=sparse-ruling-set-locality output="eval/locality_plot.pdf":
    julia --project=eval eval/locality_plot.jl eval/experiment-out/{{experiment}} {{output}}

configure:
    cmake --preset experiments

build:
    cmake --build --preset experiments --parallel

