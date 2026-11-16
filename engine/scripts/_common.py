"""Make the ``kvrsu`` package importable when a script is run from anywhere."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
