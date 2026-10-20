# Cynapse CCTV query agent

Query simulated CCTV frame records with Python, LangGraph and SQLite.

## Setup

Requires Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,openai]'
cynapse seed
```

## Chat

Without AI or an API key:

```sh
cynapse chat --nonai
```

With OpenAI:

```sh
export OPENAI_API_KEY='your-api-key'
export CYNAPSE_MODEL='openai:gpt-5-mini'
cynapse chat --ai
```

If credentials are already in your local `.env`, load them first:

```sh
set -a
source .env
set +a
cynapse chat --ai
```

Try `Show CTE frames yesterday`, then `Only between 8 AM and 10 AM`.
Use `next`, `previous`, `reset` and `exit`. Context expires after 30 idle minutes.

Use a different database:

```sh
cynapse seed --db sample.db
cynapse chat --nonai --db sample.db
```

## Tests and evaluation

Run these from the repository root with the virtual environment active:

```sh
# Local tests; no paid calls
pytest -q

# Nonai assignment examples
pytest -q tests/test_assignment_inputs.py -k nonai

# All live AI cases; requires exported credentials and makes paid calls
CYNAPSE_LIVE_TESTS=1 pytest -q tests/test_assignment_inputs.py -k openai

# Local timing benchmark
python -m evaluations.evaluate_local

# Live full-agent or extraction-only reports; paid calls
python -m evaluations.evaluate_agent --model openai:gpt-5-mini
python -m evaluations.evaluate_extraction --model openai:gpt-5-mini
```

The database is generated locally. Never commit `.env` or API keys.
See [submission.md](submission.md) for the design, results and file walkthrough.
