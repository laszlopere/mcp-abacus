# mcp-abacus

[![CI](https://github.com/laszlopere/mcp-abacus/actions/workflows/ci.yml/badge.svg)](https://github.com/laszlopere/mcp-abacus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mcp-abacus.svg)](https://pypi.org/project/mcp-abacus/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2.svg)](https://github.com/sponsors/laszlopere)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org/)
[![Last commit](https://img.shields.io/github/last-commit/laszlopere/mcp-abacus.svg)](https://github.com/laszlopere/mcp-abacus/commits)

**A calculator for the artificial minds — because we know their needs are
different.**

People reach for a calculator to get a number. A language model reaches for one
to get a number it can *trust* and reason about: Was this exact, or rounded? At
what scale? Would a wider type have held more digits? Does this overflow the way
the production code will? A floating-point answer that merely *looks* precise is
worse than no answer — it launders a rounding error into a confident claim.

mcp-abacus is built for that caller. It does **type-faithful calculation**: you
pick a numeric type/mode (fixed-point, IEEE-754 double, exact rational) and the
*whole* expression behaves exactly as that type would in real code — it rounds
where the type rounds, stays exact where the type is exact, and carries the
result onward bit-for-bit. Every answer comes back labelled with its own
precision verdict (`exact` vs `inexact, rounded to N decimals`), so the model
never has to guess whether a result is the true value. It does not approximate a
type; it calculates *using* that type.

## What it gives you

- **`calculate`** — evaluate one expression in one numeric type. Modes:
  - `fixed-point` *(default)* — exact scaled integer; money / ERC-20-safe
  - `floating-point` — IEEE-754 double (aliases `float64`, `double`)
  - `rational` — exact numerator/denominator; no silent rounding
- **`analyze`** — evaluate an expression and return its whole parse tree, each
  node annotated with the value it computed, so you can see *where* a surprising
  answer rounded or overflowed (e.g. `(1 + 1/2) * 3` is `3` in fixed-point — the
  tree shows the `1/2 = 0` leaf that explains it)
- **`solver`** — find the value of one variable that drives an expression to a
  target over a bracket: *find-root* (`x**2 - 2` over `[0, 2]` → √2) or
  *find-minimum* / *find-maximum*, in the same numeric type and expression
  language (constants come from `name = expr` assignment lines)
- **`help`** — the grammar and type reference, on tap for the model
- **`info`** — server version and environment

Each `calculate` result is self-describing: a rendered `value` string with its
precision verdict baked in, plus structured `exact` / `precision` fields. An
inexact fixed-point result even previews what a few more decimals would reveal,
so the caller is steered toward more precision rather than toward a misleading
float.

## Install and register for Claude Code

Install the server as a [uv](https://docs.astral.sh/uv/) tool from this
checkout:

```sh
uv tool install .
```

This puts an `mcp-abacus` executable on your PATH. Register it with Claude Code
(user scope, so it's available in every project):

```sh
claude mcp add abacus -- mcp-abacus
```

Then start (or `/mcp` reconnect) a Claude Code session — the `abacus` tools will
be available. Verify the server is up with:

```sh
claude mcp list
```

> **Upgrading from source:** the version is pinned, so a plain reinstall can
> reuse a cached wheel and silently install stale code. Force a clean rebuild:
> ```sh
> uv cache clean mcp-abacus
> uv tool install --force --no-cache .
> ```
> A long-lived Claude session keeps the old server subprocess until you `/mcp`
> reconnect or start a fresh session.

## Development

```sh
uv sync
uv run pytest
```

## Sponsoring

mcp-abacus is free, open-source software developed in my spare time.
Sponsorships are what keep the project alive and actively maintained — they fund
new numeric modes, bug fixes, and ongoing support, and they're a direct signal
that the work is worth continuing.

If the project is useful to you, please consider sponsoring it through
**[GitHub Sponsors](https://github.com/sponsors/laszlopere)**. Click the
**Sponsor** button at the top of the repository, or visit the link directly, and
pick a one-time or recurring tier. Every contribution, large or small, is hugely
appreciated and goes straight back into keeping mcp-abacus healthy.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later). See [LICENSE](LICENSE).
