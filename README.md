# Opyrability - Process Operability Analysis in Python.

[![PyPI Downloads](https://static.pepy.tech/badge/opyrability)](https://pepy.tech/projects/opyrability)


![GitHub forks](https://img.shields.io/github/forks/codes-group/opyrability)
![GitHub Repo stars](https://img.shields.io/github/stars/codes-group/opyrability)


![GitHub top language](https://img.shields.io/github/languages/top/codes-group/opyrability)



![Website](https://img.shields.io/website?url=https%3A%2F%2Fcodes-group.github.io%2Fopyrability%2F)
![GitHub License](https://img.shields.io/github/license/codes-group/opyrability)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05966/status.svg)](https://doi.org/10.21105/joss.05966)


Welcome to opyrability, a Python-based package for performing [Process Operability](https://www.sciencedirect.com/science/article/pii/S1474667017338028) analysis.

opyrability is developed by the [Control, Optimization and Design for Energy and Sustainability (CODES) Group](https://fernandolima.faculty.wvu.edu/) at West Virginia University.

![](/docs/opyrability_overview.png)

Authors:
- [Victor Alves](https://github.com/victoraalves)
- [San Dinh](https://github.com/sanqdinh)
- [Jonh Kitchin](https://github.com/jkitchin)
- Vitor Gazzaneo
- Juan C. Carrasco
- [Fernando V. Lima](https://github.com/fvlima-codes)

## Documentation and Process Operability Principles

Full documentation and discussion regarding process operability principles are available in [opyrability's online portal.](https://codes-group.github.io/opyrability/)

## Citing Us

To cite us, please use the following BibTeX entry below:

```
@article{Alves2024, 
doi = {10.21105/joss.05966}, 
url = {https://doi.org/10.21105/joss.05966}, 
year = {2024}, 
publisher = {The Open Journal}, 
volume = {9}, 
number = {94}, 
pages = {5966}, 
author = {Victor Alves and San Dinh and John R. Kitchin and Vitor Gazzaneo and Juan C. Carrasco and Fernando V. Lima}, 
title = {Opyrability: A Python package for process operability analysis}, journal = {Journal of Open Source Software} 
}
```

A paper describing opyrability's main functionalities is available in the Journal of Open Source Software (JOSS):

[![DOI](https://joss.theoj.org/papers/10.21105/joss.05966/status.svg)](https://doi.org/10.21105/joss.05966)



## Installation

The Anaconda distribution is needed to have some of opyrability's dependencies.

### From PyPI/conda (Windows, Linux and macOS):

The following commands will install opyrability and all dependencies on any OS (Windows, Linux and macOS):

```console
pip install opyrability
```

Then install [Cyipopt](https://github.com/mechmotum/cyipopt) from **conda**:

```console
conda install -c conda-forge cyipopt
```

### From conda (Linux and macOS only):

The single command below will install opyrability and all requirements/dependencies on Linux/macOS  operating systems automatically:

```console
conda install -c codes-group -c conda-forge opyrability
```


