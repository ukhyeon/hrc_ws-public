#!/usr/bin/env bash
# Make chumpy 0.70 (PyPI) importable with numpy >= 1.24 (SETUP.md 6.1).
#
# chumpy/__init__.py line 11 reads
#     from numpy import bool, int, float, complex, object, unicode, str, nan, inf
# and those numpy aliases were removed in numpy 1.24, so `import chumpy` fails with numpy 1.26.4.
# This comments the line out -- the same single change found in the reference PC's ~/.local copy
# (2026-11-18). chumpy is needed to unpickle the SMPL .pkl models.
#
#   tools/patch_chumpy.sh              # uses `python3` from PATH (activate the venv first)
#   PYTHON=/path/to/python tools/patch_chumpy.sh
#
# Safe to run more than once: an already commented line is left alone.
# The unpatched file is kept once as __init__.py.orig next to it (never overwritten).
set -euo pipefail

PY="${PYTHON:-python3}"
LINE='from numpy import bool, int, float, complex, object, unicode, str, nan, inf'

# locate chumpy WITHOUT importing it (the import is what fails)
read -r VER INIT < <("$PY" - <<'EOF'
import importlib.metadata as md, importlib.util, sys
spec = importlib.util.find_spec("chumpy")
if spec is None or not spec.origin:
    sys.exit("chumpy is not installed for this python")
print(md.version("chumpy"), spec.origin)
EOF
)
echo "python : $(command -v "$PY") ($("$PY" -c 'import sys; print(sys.prefix)'))"
echo "chumpy : $VER  $INIT"

if [ "$VER" != "0.70" ]; then
    echo "stop: expected chumpy 0.70, found $VER (not patching)" >&2
    exit 1
fi
[ -w "$INIT" ] || { echo "stop: $INIT is not writable" >&2; exit 1; }

if grep -qxF "$LINE" "$INIT"; then
    [ -e "$INIT.orig" ] || cp -p "$INIT" "$INIT.orig"
    # comment out exactly that line (fixed-string match on the whole line)
    "$PY" - "$INIT" "$LINE" <<'EOF'
import sys
path, line = sys.argv[1], sys.argv[2]
src = open(path).read().split("\n")
src = ["# " + l + "  # numpy >= 1.24 removed these aliases (hrc_ws tools/patch_chumpy.sh)" if l == line else l
       for l in src]
open(path, "w").write("\n".join(src))
EOF
    echo "patched: commented out the numpy alias import (original kept as $INIT.orig)"
elif grep -qE "^#\s*${LINE//./\\.}" "$INIT"; then
    echo "already patched: nothing to do"
else
    echo "stop: the expected line is not in $INIT (unknown chumpy contents, not patching)" >&2
    exit 1
fi

# quick check (no __pycache__ written into site-packages)
"$PY" -B -c "import chumpy; print('import chumpy OK')"
