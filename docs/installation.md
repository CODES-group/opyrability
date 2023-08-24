## Installation

From PyPI:

```console
pip install opyrability
```

Then install [Cyipopt](https://github.com/mechmotum/cyipopt) which is the only non-automated dependency:

```console
conda install -c conda-forge cyipopt
```

Conda packaging coming soon.

## Dependencies

opyrability is allowed to exist thanks to the following libraries that are dependencies:

- [Numpy](https://numpy.org/) - Linear Algebra.
- [Scipy](https://scipy.org/) - Scientific computing in Python.
- [Polytope](https://github.com/tulip-control/polytope) - Computational Geometry.
- [matplotlib](https://matplotlib.org/) - 2D/3D Plots.
- [tqdm](https://tqdm.github.io/) - Fancy progress bars (why not?).
- [CVXOPT](https://cvxopt.org/) - Linear programming, allowing access to [GLPK](https://www.gnu.org/software/glpk/) in Python.
- [cyipopt](https://github.com/mechmotum/cyipopt) - [IPOPT](https://coin-or.github.io/Ipopt/) wrapper in Python for nonlinear programming.
- [JAX](https://jax.readthedocs.io/en/latest/) - JAX for automatic differentiation!
