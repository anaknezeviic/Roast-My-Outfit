#!/usr/bin/env python
"""CLI wrapper around :mod:`rmo.data.download`."""

from __future__ import annotations

import logging
import sys

from rmo.data.download import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(main())
