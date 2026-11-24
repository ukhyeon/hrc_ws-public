"""Filesystem locations used by hrc_body_tracking.

Every directory comes from an environment variable with a home-relative
default, so another PC only needs the variables set (or the same layout):

    HRC_WS_DIR          default ~/hrc_ws
    REALTIME_SMPL_DIR   default ~/realtime_smpl

HRC_WS_DIR cannot be derived from __file__ because colcon installs a copy of
this package under install/. Paths are only user-expanded (no resolve()), so
they stay identical to the previous hard-coded strings on this machine.
"""
import os
from string import Template

_DEFAULTS = {
    "HRC_WS_DIR": "~/hrc_ws",
    "REALTIME_SMPL_DIR": "~/realtime_smpl",
}


def _get(name):
    return os.path.expanduser(os.environ.get(name) or _DEFAULTS[name])


HRC_WS_DIR = _get("HRC_WS_DIR")
REALTIME_SMPL_DIR = _get("REALTIME_SMPL_DIR")


def expand(value):
    """Resolve ${VAR} (with the defaults above) and ~ in a config string."""
    mapping = {name: _get(name) for name in _DEFAULTS}
    return os.path.expanduser(Template(value).safe_substitute(mapping))


def ws(*parts):
    """Path inside the hrc_ws workspace (source tree, not install/)."""
    return os.path.join(HRC_WS_DIR, *parts)

