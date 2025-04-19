#!/usr/bin/env python
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="TensorFlow_Decision_Forests",
    version="0.0.1",
    author="Inaki Madinabeitia",
    description="TensorFlow Decision Forests",
    license="GNU",
    keywords="",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Alpha",
    ],
)