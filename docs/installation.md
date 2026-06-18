## Installation

opyrability is distributed on PyPI and installs with `pip`; all required dependencies come with it.


#### From PyPI (**Windows**, Linux and macOS):

The command below installs opyrability and all of its required dependencies on any OS (Windows, Linux and macOS):

```console
pip install opyrability
```

This includes the default nonlinear-programming solver, [Pounce](https://kitchingroup.cheme.cmu.edu/pounce/) (a pure-Rust port of IPOPT installed from precompiled binaries), so the operability calculations that need a nonlinear programming solver just works out of the box with no extra setup.

**Optional IPOPT backend:** [cyipopt](https://github.com/mechmotum/cyipopt) is now optional and only needed if you pass `method='ipopt'` to the inverse mapping. Install it from conda when you want that backend:

```console
conda install -c conda-forge cyipopt
```

**Optional algebraic, equation-oriented modeling:** For Pyomo/OMLT model support, install the extras:

```console
pip install "opyrability[pyomo]"
```

**Optional interactive plots:** The dynamic operability funnels render as static matplotlib figures by default. To get interactive 3D funnels (`engine='plotly'`), install [Plotly](https://plotly.com/python/) via the extra:

```console
pip install "opyrability[plotly]"
```

## Using in a Google Colab environment:

opyrability installs in a Colab session with a single pip command:
```console
!pip install opyrability
```


## Dependencies

Opyrability can only exist thanks to the following great software libraries that are dependencies:

- [Numpy](https://numpy.org/) - Linear Algebra.
- [Scipy](https://scipy.org/) - Scientific computing in Python.
- [Polytope](https://github.com/tulip-control/polytope) - Computational Geometry.
- [matplotlib](https://matplotlib.org/) - 2D/3D Plots.
- [tqdm](https://tqdm.github.io/) - Fancy progress bars (why not?).
- [CVXOPT](https://cvxopt.org/) - Linear programming, allowing access to [GLPK](https://www.gnu.org/software/glpk/) in Python.
- [Pounce](https://kitchingroup.cheme.cmu.edu/pounce/) - The default nonlinear-programming solver (a pure-Rust port of IPOPT with the bundled FERAL linear solver).
- [JAX](https://jax.readthedocs.io/en/latest/) - JAX for automatic differentiation!

Optional dependencies (installed only and if you need them):

- [cyipopt](https://github.com/mechmotum/cyipopt) - [IPOPT](https://coin-or.github.io/Ipopt/) wrapper in Python; the optional `method='ipopt'` backend.
- [Pyomo](https://www.pyomo.org/) and [pyomo-pounce](https://kitchingroup.cheme.cmu.edu/pounce/) - equation-oriented (Pyomo/OMLT) model support, via `pip install "opyrability[pyomo]"`.
- [Plotly](https://plotly.com/python/) - interactive 3D operability funnels (`engine='plotly'`), via `pip install "opyrability[plotly]"`.
