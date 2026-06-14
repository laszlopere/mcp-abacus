# Contributing to mcp-abacus

Thanks for your interest in improving **mcp-abacus**! Contributions of all kinds
are welcome — bug reports, feature requests, docs, and code.

## Where things live

- Source: [`github.com/laszlopere/mcp-abacus`](https://github.com/laszlopere/mcp-abacus)
- Issues & feature requests: <https://github.com/laszlopere/mcp-abacus/issues>
- Package source lives under `src/mcp_abacus/`; tests under `tests/`.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[dev]"
```

## Checks before opening a PR

The CI runs these on Python 3.10–3.13; please run them locally first:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # type-check (src/)
pytest                  # unit tests (or ./run-tests.sh for a per-failure summary)
```

## Guidelines

- **Type-faithfulness matters.** Every result must be bit-for-bit what the
  selected numeric regime (fixed-width int, IEEE-754 float, fixed-point, decimal,
  rational) actually produces — never an approximation that "looks right."
- **`Value` is the single chokepoint for type behavior.** Adding a numeric type
  is a local, two-step change — add its member to the mode enum, then implement
  that member's branch in each operation in `value.py`. The rest of the engine is
  generic over the enum and never branches on type itself; keep it that way.
- Match the surrounding code style — type hints, no stray `type: ignore`.
- Update the README when you add or change a tool, mode, operator, or function.

## Reporting bugs

Use the issue templates. Please include your mcp-abacus version, Python version,
the numeric mode(s) involved, and a minimal expression that reproduces the issue.

## License

By contributing, you agree that your contributions are licensed under the
project's [GNU General Public License v3.0 or later](LICENSE).
