set dotenv-load := true

run-supermuc-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine supermuc --module-config kascade --cores node-size-pow2 --min-cores 48 --max-cores 2048 {{ experiment }} {{ args }}

run-horeka-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine horeka --module-config kascade --command-template ./kaval/command-templates/horeka-IntelMPI.txt --cores node-size-pow2 --min-cores 64 --max-cores 2048  {{ experiment }} {{ args }}
