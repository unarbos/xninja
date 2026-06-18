# xninja

`xninja` is a small PyPI-ready CLI package that runs the public
[`unarbos/ninja`](https://github.com/unarbos/ninja) agent like a local coding
assistant.

Subnet 66 harnesses are now multi-file bundles: an `agent.py` entrypoint plus a
stdlib-only `agent/` package (up to 32 `*.py` files), with `tau_agent_files.json`
listing every file. `xninja` bundles the latest such snapshot at build time,
loads it exactly the way the validator does (the bundle root goes on `sys.path`
so the entrypoint's `from agent.* import ...` resolves), and calls its
validator-compatible contract:

```python
solve(repo_path, issue, model, api_base, api_key)
```

Single-file agents (just `agent.py`) remain fully supported.

## Install

### Prerequisites

- Python 3.11 or newer
- `pip` (bundled with Python)
- An [OpenRouter](https://openrouter.ai) API key

### From PyPI (recommended)

```bash
pip install xninja
```

Verify the installation:

```bash
xninja --version
```

### From source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/unarbos/xninja.git
cd xninja
python -m pip install -e .
```

Verify the installation:

```bash
xninja --version
```

## Configure

```bash
xninja config
```

This stores your OpenRouter API key and default model in the OS user config
directory, such as `~/.config/xninja/config.json` on Linux. The file is written
with user-only permissions.

Environment variables override stored config:

```bash
OPENROUTER_API_KEY=...
XNINJA_MODEL=anthropic/claude-sonnet-4.6
```

## Use

Open an interactive session in the current git repo:

```bash
xninja
```

Run a one-shot task:

```bash
xninja "fix the failing parser test"
```

Run explicitly:

```bash
xninja run --repo . --model anthropic/claude-sonnet-4.6 "add validation for empty input"
```

By default, `xninja` previews the returned patch and asks before applying it.
Use `--apply` to request application after the run.

## Agent Sources

Show bundled metadata:

```bash
xninja agent info
```

Cache and use a specific `unarbos/ninja` ref. The whole bundle (every file in
`tau_agent_files.json`) is fetched, not just `agent.py`:

```bash
xninja agent update --ref cec561e45192042687c053237c1db503cd7d3ae0
xninja run --agent-ref cec561e45192042687c053237c1db503cd7d3ae0 "fix the issue"
```

Use your own compatible local agent. `--agent-path` accepts either a single
`agent.py` or a bundle directory containing `agent.py` (and, optionally, an
`agent/` package and a `tau_agent_files.json` manifest):

```bash
xninja run --agent-path ./agent.py "fix the issue"
xninja run --agent-path ./my_bundle "fix the issue"
xninja --agent-path ~/agents/my_agent.py "fix the issue"
```

Custom agents must expose the same callable contract as the bundled agent:

```python
def solve(repo_path, issue, model, api_base, api_key):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
```

For a multi-file bundle, `agent.py` is the entrypoint and may import its
supporting modules (`from agent.foo import bar`); list every file, including
`agent.py`, in `tau_agent_files.json`.

## Models

```bash
xninja models
```

Recommended defaults are intentionally simple. You can pass any OpenRouter model
id with `--model`.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```
