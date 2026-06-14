#!/usr/bin/env bash
# Run the whole unit-test suite; print which tests failed at the end.
# Extra arguments are passed straight to pytest, except:
#   --verbose / -v   one test per line AND the tests narrate what they do —
#                    expressions, token streams, parse trees and computed
#                    values in readable form (see tests/conftest.py).
#                    Translated to `pytest -v -s` (verbose, capture off).
#   --human-readable  frame each solver call as a Code / Result block
#                    (see tests/conftest.py). Translated to
#                    `pytest --human-readable -s` (capture off so it prints).

set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

pytest_args=()
for arg in "$@"; do
    case "$arg" in
        --verbose|-v) pytest_args+=(-v -s) ;;
        --human-readable) pytest_args+=(--human-readable -s) ;;
        *) pytest_args+=("$arg") ;;
    esac
done

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# -rfE adds end-of-run summary lines for failures and errors (parsed below).
uv run pytest -rfE "${pytest_args[@]}" 2>&1 | tee "$tmp"
status="${PIPESTATUS[0]}"

echo
if [ "$status" -eq 0 ]; then
    echo "All tests passed."
else
    failures="$(grep -E '^(FAILED|ERROR) ' "$tmp" || true)"
    if [ -n "$failures" ]; then
        echo "Failed tests:"
        printf '%s\n' "$failures" | sed 's/^/  /'
    else
        echo "pytest exited with status $status without reporting individual failures (see output above)."
    fi
fi
exit "$status"
