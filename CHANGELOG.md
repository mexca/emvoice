# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2023-07-17

Minor fix with correct URLs in `setup.cfg`.


## [0.1.0] - 2023-07-17

First release since creation of stand-alone package. Previously code was part of [mexca](https://github.com/mexca/mexca) package. See mexca's [changelog](https://github.com/mexca/mexca/blob/main/CHANGELOG.md) for details.

### Added

- Docs on algorithms for voice feature computation.
- Basic examples on how to use emvoice:
    - Loading and framing a speech signal
    - Estimating voice pitch
    - Estimating jitter and shimmer
    - Estimating formants
- Support for Python 3.10 and 3.10

### Removed

- Some algorithm details were removed from the API documentation and moved to the algorithms documentation.
