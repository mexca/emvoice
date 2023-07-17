# Compute emotion expression-related voice features

[![github license badge](https://img.shields.io/github/license/mexca/emvoice)](https://github.com/mexca/emvoice)
[![RSD](https://img.shields.io/badge/rsd-emvoice-00a3e3.svg)](https://www.research-software.nl/software/emvoice)
[![DOI](https://zenodo.org/badge/665993095.svg)](https://zenodo.org/badge/latestdoi/665993095)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_emvoice&metric=alert_status)](https://sonarcloud.io/dashboard?id=mexca_emvoice)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_emvoice&metric=coverage)](https://sonarcloud.io/dashboard?id=mexca_emvoice)
[![Documentation Status](https://readthedocs.org/projects/emvoice/badge/?version=latest)](https://emvoice.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/mexca/emvoice/actions/workflows/build.yml/badge.svg)](https://github.com/mexca/emvoice/actions/workflows/build.yml)
[![cffconvert](https://github.com/mexca/emvoice/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/mexca/emvoice/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/mexca/emvoice/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/mexca/emvoice/actions/workflows/sonarcloud.yml)
[![markdown-link-check](https://github.com/mexca/emvoice/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/mexca/emvoice/actions/workflows/markdown-link-check.yml)
[![black code style badge](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## How to use emvoice

**emvoice** is a pure Python package for computing emotion expression-related features from speech signals. It uses similar algorithms as in [Praat](https://www.praat.org) and [openSMILE](https://github.com/audeering/opensmile/) but also includes more recent methods from the [librosa](https://librosa.org/doc/latest/index.html) package. Currently, most low-level descriptor features from the extended Geneva Minimalistic Acousting Parameter Set ([eGeMAPS](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7160715)) are implemented.

Given that it is entirely written in Python, it is easier to include emvoice in other Python-based applications, especially if they use numpy or scipy.

## Getting started

Take a look at the [examples](https://emvoice.readthedocs.io/en/latest/examples.html) to get started with emvoice.

## Installation

emvoice requires Python >=3.7 and can be installed via `pip`:

```console
pip install emvoice
```

To install the lastet development version from GitHub repository, do:

```console
git clone https://github.com/mexca/emvoice.git
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
