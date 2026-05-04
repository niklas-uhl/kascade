set dotenv-load := true

run-supermuc-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine supermuc --module-config kascade --cores node-size-pow2 --min-cores 48 --max-cores 2048 {{ experiment }} {{ args }}

run-horeka-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine horeka --module-config kascade --command-template ./kaval/command-templates/horeka-IntelMPI.txt --cores node-size-pow2 --min-cores 64 --max-cores 2048  {{ experiment }} {{ args }}

min-cores := 1
max-cores := 64

run-locality-experiment experiment=sparse-ruling-set-locality +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments/submission/ --machine shared --cores node-size-pow2 --min-cores {{min-cores}} --max-cores {{max-cores}}  {{ experiment }} {{ args }}

configure:
    cmake --preset experiments

build:
    cmake --build --preset experiments --parallel

