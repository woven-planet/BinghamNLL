from os.path import splitext, basename
from setuptools import setup
from setuptools import find_packages
import glob

# def setup(**kwargs):
#   print(kwargs)

def get_pyname(fname):
  return splitext(basename(fname))[0]

src_dir = "src"
setup(
    name="BinghamNLL",
    version="1.0.0",
    package_dir={'': src_dir},
    packages=find_packages(where=src_dir),
    py_modules=[get_pyname(fname) for fname in glob.glob('{}/*.py'.format(src_dir)) if not get_pyname(fname) == '__init__'],
)
