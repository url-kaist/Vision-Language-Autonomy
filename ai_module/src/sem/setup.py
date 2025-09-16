"""Catkin setup for semantic_segment."""

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["sem"],
    package_dir={"": "src"},
    # scripts=["app/semantic_segment_node"],
)
setup(**setup_args)
