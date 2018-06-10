"""
Adding the library path to the syspath for imports of packages
"""
#adding project path to the system path, if unix os, export recommended
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
