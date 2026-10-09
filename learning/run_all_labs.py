"""Run every module's lab and report which ones pass.

Usage:  python learning/run_all_labs.py

Each lab ends with assertions that check its numbers against theory, so this is
also the test suite for the learning material.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parent
labs = sorted(root.glob("[0-9][0-9]_*/lab.py"))
failed = []
for lab in labs:
    start = time.time()
    proc = subprocess.run([sys.executable, str(lab)], capture_output=True, text=True)
    status = "ok" if proc.returncode == 0 else "FAILED"
    print(f"{status:6s} {lab.parent.name:32s} {time.time() - start:5.1f}s")
    if proc.returncode != 0:
        failed.append(lab)
        print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
print(f"\n{len(labs) - len(failed)}/{len(labs)} labs passed")
sys.exit(1 if failed else 0)
