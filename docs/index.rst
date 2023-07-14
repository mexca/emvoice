.. emvoice documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to emvoice's documentation!
==========================================================

**emvoice** is a pure Python package for computing emotion expression-related features from speech signals.
It uses similar algorithms as in `Praat <https://www.praat.org>`_ and `openSMILE <https://github.com/audeering/opensmile/>`_ but
also includes more recent methods from the `librosa <https://librosa.org/doc/latest/index.html>`_ package.
Currently, most low-level descriptor features from the
extended Geneva Minimalistic Acousting Parameter Set (`eGeMAPS <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7160715>`_)
are implemented.

Given that it is entirely written in Python, it is easier to include emvoice in other Python-based applications,
especially if they use numpy or scipy.

Installation
------------

emvoice requires Python >=3.7 and can be installed via `pip`:

.. code-block:: console

  pip install emvoice

To install the lastet development version from GitHub repository, do:

.. code-block:: console

  git clone https://github.com/mexca/emvoice.git
  cd emvoice
  python -m pip install .


.. toctree::
  :maxdepth: 2
  :caption: Contents:

  algorithms
  examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
