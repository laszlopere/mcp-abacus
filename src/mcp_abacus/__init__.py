# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""mcp-abacus — type-faithful calculation MCP server."""

import sys
from importlib.metadata import PackageNotFoundError, version

# Lift CPython's default 4300-digit int<->str cap (PEP 7000 / CVE-2020-10735 guard). abacus
# is an EXACT bignum calculator: its own DoS-bounded operations already produce integers far
# past 4300 digits — the sum/product fold caps at 100000 terms, so product(i, 1, 100000, i) is
# 100000! at ~456574 digits, and an exact rational fit can hold a comparably large numerator/
# denominator. Those results compute fine but then CRASH when rendered (Value.to_string ->
# str(int)), so the cap is simply wrong for this application. DoS is bounded at the operation
# level instead (factorial's 1000 cap, the fold's 100000-term cap, the solver's wall-clock);
# a generous finite ceiling here keeps a backstop while letting every legitimately-bounded
# result render. Set once at import so the server, CLI and tests all behave identically.
sys.set_int_max_str_digits(1_000_000)

try:
    __version__ = version("mcp-abacus")
except PackageNotFoundError:  # running from an unbuilt source tree
    __version__ = "0.0.0+unknown"
