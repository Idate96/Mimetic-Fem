"""Path hack to make tests work."""

import os
import sys

modpath = os.path.realpath('.') + '/src'
sys.path.insert(0, modpath)
