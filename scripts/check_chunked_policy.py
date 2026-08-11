#!/usr/bin/env python3
"""Exercise ACT-style action chunks against every joint simulator."""

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunked_policy import replay_recorded_chunks  # noqa: E402
from sim_tasks import TASK_SPECS  # noqa: E402


def main():
    results = [replay_recorded_chunks(task) for task in TASK_SPECS]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
