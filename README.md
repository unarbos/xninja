# xninja

`xninja` is a small PyPI-ready CLI package that runs the public
[`unarbos/ninja`](https://github.com/unarbos/ninja) agent like a local coding
assistant.

It bundles the latest `ninja` `agent.py` snapshot at package build time and
calls its validator-compatible contract:

```python
solve(repo_path, issue, model, api_base, api_key)
```

## Install

From this repo:

```bash
python -m pip install -e .
```

From PyPI, once published:

```bash
python -m pip install xninja
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

Cache and use a specific `unarbos/ninja` ref:

```bash
xninja agent update --ref cec561e45192042687c053237c1db503cd7d3ae0
xninja run --agent-ref cec561e45192042687c053237c1db503cd7d3ae0 "fix the issue"
```

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
