#!/usr/bin/env python
"""CLI wrapper around :mod:`rmo.perception.train_cnn`."""

from __future__ import annotations

import logging
import sys

from rmo.perception.train_cnn import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(main())
