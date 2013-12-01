import os, shutil
from setuptools import setup

with open('README.md') as f:
    description = f.read()
with open('LICENSE.md') as f:
    license = f.read()

setup(
    name = "openautomata",
    version = "0.1",
    author = "Miguel Flores",
    author_email = "miguel.frde@gmail.com",
    description = ("Python automata theory library."),
    long_description = description,
    license = license,
    keywords = "automata automaton grammar turing dfa nfa context sensitive free",
    url = "https://github.com/miguelfrde/openautomata",
    packages = ['openautomata', ],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
    ],
)

shutil.rmtree("dist/")
shutil.rmtree("build/")
shutil.rmtree("openautomata.egg-info/")