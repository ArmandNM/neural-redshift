import sys
from os.path import dirname
from os.path import abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
