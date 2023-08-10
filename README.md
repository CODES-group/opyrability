# PyPO - Python-based Process Operability package.

Welcome to PyPO, a Python-based package for performing [Process Operability](https://www.sciencedirect.com/science/article/pii/S1474667017338028) analysis.

PyPO is developed by the Control, Optimization and Design for Energy and Sustainability (CODES) Group at West Virginia University under Dr. [Fernando V. Lima](https://fernandolima.faculty.wvu.edu/)'s supervision.

![](/docs/pypo_overview.png)

Authors:
[Victor Alves](https://github.com/victoraalves) and [San Dinh](https://github.com/sanqdinh)




## Installation

Download the files from the repo, and extract them to any location on your PC. Then from the terminal (or cmd if you are using Windows) navigate to the location where you extracted the files and execute:

```console
pip install -e . 
```

Then install [Cyipopt](https://github.com/mechmotum/cyipopt) which is the only non-automated dependency:

```console
conda install -c conda-forge cyipopt
```

Online Pip and Conda packages coming soon.

## Documentation and Process Operability principles

Full documentation and discussion regarding process operability principles are available in [PyPO's online portal.](https://codes-group.github.io/PyPO/)



