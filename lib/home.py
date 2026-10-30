"""Install root for GJ V2.

GJ_HOME wins, then FARQUAD_HOME, then the clone that contains this file.
Scripts keep working when the repo is checked out somewhere other than ~/farquad.
"""
import os
import pathlib


def gj_home() -> str:
    for key in ("GJ_HOME", "FARQUAD_HOME"):
        val = os.environ.get(key)
        if val:
            return os.path.expanduser(val)
    return str(pathlib.Path(__file__).resolve().parents[1])
