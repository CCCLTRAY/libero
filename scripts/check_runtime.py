#!/usr/bin/env python3
"""Print and validate the runtime contract used by released Scene-2 data."""

import json

from libero_scene2.runtime import validate_runtime


if __name__ == "__main__":
    print(json.dumps(validate_runtime(strict=True), indent=2))
