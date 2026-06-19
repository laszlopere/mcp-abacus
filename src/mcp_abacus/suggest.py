# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Nearest-match "did you mean" hints for unknown names (TODO 43.5).

The expression-layer analog of the actionable JSON errors in errors.py (43.2):
when a tool call names a function, mode, or algorithm that does not exist, the
error appends the closest valid spelling so a model self-corrects in one turn
(``sqr`` -> ``sqrt``, ``arcsin`` -> ``asin``) instead of retrying blindly.

Backed by the stdlib ``difflib`` so there is no new dependency. The match runs
only AFTER the caller's own alias resolution has already failed, over the union
of canonical names and aliases, and a deliberately high cutoff means a genuine
typo gets a suggestion while an unrelated word gets none — a wrong "did you mean"
is worse than silence. It catches typos only, never semantic misses (``ln`` for
``log`` is not a near-spelling; such pairs belong in the alias tables instead).
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

# Minimum similarity (0..1) for a candidate to be offered. difflib's own default;
# at this height "sqr"/"sqrt" (~0.86) and "arcsin"/"asin" (~0.8) clear it while an
# unrelated token does not, so a suggestion never points the caller the wrong way.
_CUTOFF = 0.6


def did_you_mean(word: str, candidates: Iterable[str]) -> str:
    """Return ``" Did you mean 'X'?"`` for the closest candidate, or ``""`` if none fits.

    A trailing fragment meant to be appended directly to an "unknown ..." message:
    it leads with a space and is empty when no candidate clears ``_CUTOFF``, so the
    caller can splice it unconditionally. ``candidates`` is the full set of accepted
    spellings (canonical names plus aliases); the single best match is offered.
    """
    matches = difflib.get_close_matches(word, list(candidates), n=1, cutoff=_CUTOFF)
    if not matches:
        return ""
    return f" Did you mean {matches[0]!r}?"
