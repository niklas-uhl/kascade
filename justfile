set dotenv-load := true

run-supermuc-experiment experiment +args="":
    uv run kaval/run-experiments.py --search-dirs ./experiments --machine supermuc --module-config kaccv2 {{ experiment }} {{ args }}
