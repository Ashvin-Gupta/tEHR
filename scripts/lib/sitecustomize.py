"""Point Triton/gcc at a user-extracted Python.h when system headers are missing.

RHEL's python3.11-libs package leaves /usr/include/python3.11 without Python.h.
Triton compiles cuda_utils.c with -I$(sysconfig include), so Unsloth kernels fail
unless that path is patched before Triton imports.
"""
from __future__ import annotations

import os
import sysconfig


def _apply() -> None:
    inc = os.environ.get("PYTHON_INCLUDE_DIR", "")
    if not inc or not os.path.isfile(os.path.join(inc, "Python.h")):
        return
    sys_inc = sysconfig.get_paths().get("include") or ""
    if os.path.isfile(os.path.join(sys_inc, "Python.h")):
        return

    orig_get_paths = sysconfig.get_paths
    orig_get_config_var = sysconfig.get_config_var

    def get_paths(*args, **kwargs):
        paths = dict(orig_get_paths(*args, **kwargs))
        paths["include"] = inc
        paths["platinclude"] = inc
        return paths

    def get_config_var(name):
        if name in ("INCLUDEPY", "CONFINCLUDEPY"):
            return inc
        return orig_get_config_var(name)

    sysconfig.get_paths = get_paths  # type: ignore[method-assign]
    sysconfig.get_config_var = get_config_var  # type: ignore[method-assign]


_apply()
