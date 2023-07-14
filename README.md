# Compute emotion expression-related voice features

[![github license badge](https://img.shields.io/github/license/mexca/emvoice)](git@github.com:mexca/emvoice)
[![RSD](https://img.shields.io/badge/rsd-emvoice-00a3e3.svg)](https://www.research-software.nl/software/emvoice)
[![workflow pypi badge](https://img.shields.io/pypi/v/emvoice.svg?colorB=blue)](https://pypi.python.org/project/emvoice/)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_emvoice&metric=alert_status)](https://sonarcloud.io/dashboard?id=mexca_emvoice)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_emvoice&metric=coverage)](https://sonarcloud.io/dashboard?id=mexca_emvoice)
[![Documentation Status](https://readthedocs.org/projects/emvoice/badge/?version=latest)](https://emvoice.readthedocs.io/en/latest/?badge=latest)
[![build](git@github.com:mexca/emvoice/actions/workflows/build.yml/badge.svg)](git@github.com:mexca/emvoice/actions/workflows/build.yml)
[![cffconvert](git@github.com:mexca/emvoice/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:mexca/emvoice/actions/workflows/cffconvert.yml)
[![sonarcloud](git@github.com:mexca/emvoice/actions/workflows/sonarcloud.yml/badge.svg)](git@github.com:mexca/emvoice/actions/workflows/sonarcloud.yml)
[![markdown-link-check](git@github.com:mexca/emvoice/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:mexca/emvoice/actions/workflows/markdown-link-check.yml)

## How to use emvoice

**emvoice** is a pure Python package for computing emotion expression-related features from speech signals. It uses similar algorithms as in [Praat](https://www.praat.org) and [openSMILE](https://github.com/audeering/opensmile/) but also includes more recent methods from the [librosa](https://librosa.org/doc/latest/index.html) package. Currently, most features from the extended Geneva Minimalistica Acousting Parameter Set ([eGeMAPS](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7160715)) are implemented.

Given that it is entirely written in Python, it is easier to include emvoice in other Python-based applications, especially if they use numpy or scipy.

## Installation

emvoice requires Python >=3.7 and can be installed via `pip`:

```console
pip install emvoice
```

To install the lastet development version from GitHub repository, do:

```console
git clone git@github.com:mexca/emvoice.git
cd emvoice
python -m pip install .
```

## Documentation

The documentation of emvoice can be found on [Read the Docs](https://emvoice.readthedocs.io/en/latest/index.html).

## Contributing

If you want to contribute to the development of emvoice,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
