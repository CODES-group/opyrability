---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# API Documentation

The functions described below are part of opyrability and are
classified based on their functionality. Each function also contains a worked
example based on the famous [Shower Problem](examples_gallery/operability_index_shower.ipynb){cite}`vinson00, lima10b`

## Conventional mapping (AIS to AOS)

### Forward mapping

```{eval-rst}
.. autofunction:: opyrability.AIS2AOS_map
```

###### Example
Obtaining the Achievable Output Set (AOS) for the shower problem.

Importing opyrability and Numpy:
```{code-cell}
    from opyrability import AIS2AOS_map
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = 90
```

```{code-cell}
    def shower_problem(u):
        y = np.zeros(2)
        y[0]=u[0]+u[1]
        if y[0]!=0:
            y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
        else:
            y[1]=(60+120)/2
            
        return y
```
Defining the AIS bounds, as well as the discretization resolution:
```{code-cell}
    AIS_bounds =  np.array([[0, 10], [0, 10]])
    resolution =  [25, 25]
```

Obtain discretized AIS/AOS.

```{code-cell}
    AIS, AOS =  AIS2AOS_map(shower_problem, AIS_bounds,  resolution, plot = True)
    
```

### Inverse mapping (AOS/DOS to AIS/DIS)

### NLP-Based

```{eval-rst}
.. autofunction:: opyrability.nlp_based_approach
```

###### Example
Obtaining the Feasible Desired Input Set (DIS*) for the shower problem.

Importing opyrability and Numpy:

```{code-cell}
    import numpy as np
    from opyrability import nlp_based_approach
```

Defining lower and upper bound for the AIS/DIS inverse map:

```{code-cell}
    lb = np.array([0, 0])
    ub = np.array([100,100])
```
Defining DOS bounds and resolution to obtain the inverse map:

```{code-cell}
    DOS_bound = np.array([[15, 20],
                          [80, 100]])
    resolution = [5, 5]
```

Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = 90
```

```{code-cell}
    def shower_problem(u):
        y = np.zeros(2)
        y[0]=u[0]+u[1]
        if y[0]!=0:
            y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
        else:
            y[1]=(60+120)/2
            
        return y
```

Obtaining the DIS*, DOS* and the convergence for each inverse map run. 
Additionally, using IPOPT as the NLP solver, enabling plotting of the process operability
sets, cold-starting the
NLP and using finite differences:

```{code-cell}
    
    u0 = u0 = np.array([0, 10]) # Initial estimate for inverse mapping.
    fDIS, fDOS, message = nlp_based_approach(shower_problem,
                                             DOS_bound, 
                                             resolution, 
                                             u0, 
                                             lb,
                                             ub, 
                                             method='ipopt', 
                                             plot=True, 
                                             ad=False,
                                             warmstart=False)
```


### MILP-Based (multilayer operability framework)

```{eval-rst}
.. autofunction:: opyrability.milp_based_approach
```

###### Example

Optimal modular design of the shower problem, minimizing the total water
usage subject to a desired output region and the linear input constraint
$u_1 \leq u_2$:

```{code-cell} ipython3
import numpy as np
from opyrability import milp_based_approach

# shower_problem is defined in the forward-mapping example above.
u_opt, y_opt, phi, phi_true, history = milp_based_approach(
    shower_problem,
    AIS_bound=np.array([[0.1, 10.0], [0.1, 10.0]]),
    PI_target=lambda u: u[0] + u[1],
    DOS_bounds=np.array([[6.0, 9.0], [85.0, 95.0]]),
    AIS_resolution=5,
    input_constr=(np.array([[1.0, -1.0]]), np.array([0.0])))
```

## Implicit mapping

```{eval-rst}
.. autofunction:: opyrability.implicit_map
```

###### Example

Trace the forward map (AIS to AOS) and the inverse map (DOS to DIS) of a model
written implicitly as $F(u, y) = 0$, here a simple linear example $y = A u$:

```{code-cell}
    import numpy as np
    import jax.numpy as jnp
    from opyrability import implicit_map

    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    def F(u, y):                        # F(u, y) = 0  <=>  y = A u
        return jnp.array([y[0] - (A[0, 0] * u[0] + A[0, 1] * u[1]),
                          y[1] - (A[1, 0] * u[0] + A[1, 1] * u[1])])

    # Forward: map the Available Input Set to the Achievable Output Set.
    AIS_imp = np.array([[0.0, 1.0], [0.0, 1.0]])
    fwd_in, fwd_out, _, _ = implicit_map(F,
                                         image_init=A @ AIS_imp[:, 0],
                                         domain_bound=AIS_imp,
                                         domain_resolution=[5, 5],
                                         direction='forward')

    # Inverse: map a Desired Output Set back to the Desired Input Set.
    DOS_imp = np.array([[0.0, 2.0], [0.0, 2.0]])
    inv_in, inv_out, _, _ = implicit_map(F,
                                         image_init=np.linalg.solve(A, DOS_imp[:, 0]),
                                         domain_bound=DOS_imp,
                                         domain_resolution=[5, 5],
                                         direction='inverse')
```

```{code-cell}
    print('forward AOS grid:', np.asarray(fwd_out).shape)
    print('inverse DIS grid:', np.asarray(inv_out).shape)
```

## Multimodel representation

```{eval-rst}
.. autofunction:: opyrability.multimodel_rep
```

###### Example
Obtaining the Achievable Output Set (AOS) for the shower problem.

Importing opyrability and Numpy:
```{code-cell} 
    from opyrability import multimodel_rep
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = 90
```

```{code-cell}
    def shower_problem(u):
        y = np.zeros(2)
        y[0]=u[0]+u[1]
        if y[0]!=0:
            y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
        else:
            y[1]=(60+120)/2
            
        return y
```
Defining the AIS bounds and the discretization resolution:
```{code-cell}
    AIS_bounds =  np.array([[1, 10], [1, 10]])
    AIS_resolution =  [5, 5]
```

Obtaining multimodel representation of paired polytopes for the AOS:

```{code-cell} 
    AOS_region  =  multimodel_rep(shower_problem, AIS_bounds, AIS_resolution)
```

## OI evaluation

```{eval-rst}
.. autofunction:: opyrability.OI_eval
```

###### Example
Evaluating the OI for the shower problem for a given DOS.



Importing opyrability and Numpy:
```{code-cell} 
    from opyrability import multimodel_rep, OI_eval
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = 90
```

```{code-cell}
    def shower_problem(u):
        y = np.zeros(2)
        y[0]=u[0]+u[1]
        if y[0]!=0:
            y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
        else:
            y[1]=(60+120)/2
            
        return y
```
Defining the AIS bounds and the discretization resolution:
```{code-cell}
    AIS_bounds =  np.array([[1, 10], [1, 10]])
    AIS_resolution =  [10, 10]
```

Obtaining multimodel representation of paired polytopes for the AOS:

```{code-cell} 
    AOS_region  =  multimodel_rep(shower_problem, AIS_bounds, AIS_resolution,
    plot=False)
```

Defining a DOS region between $y_1 =[10-20], y_2=[70-100]$
```{code-cell} 
    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])
```

Evaluating the OI and seeing the intersection between the operability sets:
```{code-cell} 
    OI = OI_eval(AOS_region, DOS_bounds)
```

### Ranking designs by the OI

`rank_designs` scores several process models by their Operability Index against
a shared DOS and returns them ranked from most to least operable, from either
the output perspective ($\mu(AOS \cap DOS)/\mu(DOS)$) or the input perspective,
which inverse-maps the DOS to the feasible desired input set DIS\*
($\mu(DIS^* \cap AIS)/\mu(AIS)$).

```{eval-rst}
.. autofunction:: opyrability.rank_designs
```

###### Example

Ranking two design envelopes of the shower problem by their output-space OI,
reusing ``shower_problem`` and ``DOS_bounds`` defined above:
```{code-cell}
    from opyrability import rank_designs

    ranking = rank_designs(
        {'Wide valves': shower_problem, 'Narrow valves': shower_problem},
        AIS_bound={'Wide valves': np.array([[1, 10], [1, 10]]),
                   'Narrow valves': np.array([[3, 8], [3, 8]])},
        DOS_bound=DOS_bounds,
        resolution=[10, 10],
        perspective='outputs',
        plot=True)
```

## Dynamic operability

Dynamic operability extends the steady-state sets to systems that evolve in
time {cite}`dinh23, dinh26`. Starting from an initial state, it builds the
*achievable-output funnel* over a horizon of $k$ time steps and evaluates the
Dynamic Operability Index (dOI) against the Desired Output Set (DOS) at each
step.

The recommended high-level entry point is `dynamic_operability`, which
auto-selects the propagation method (linear state-space projection for matrix
models, nonlinear projection for low-dimensional states, or n-step simulation
otherwise), evaluates the dOI, and plots the dOI-colored funnel in a single
call.

```{eval-rst}
.. autofunction:: opyrability.dynamic_operability
```

###### Example

A stable two-state linear time-invariant system, supplied as a matrices dict
$\{A, B, C\}$ (so the linear state-space projection is selected
automatically): build the funnel over four steps and evaluate the dOI against
a desired output set.

```{code-cell}
    import numpy as np
    from opyrability import dynamic_operability

    # Linear time-invariant model: x(k+1) = A x + B u, y = C x.
    model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}

    AIS = np.array([[-1.0, 1.0], [-1.0, 1.0]])   # Achievable Input Set.
    DOS = np.array([[-2.0, 2.0], [-2.0, 2.0]])   # Desired Output Set.

    result = dynamic_operability(model, x0=np.zeros(2), AIS_bound=AIS,
                                 DOS=DOS, k_max=4)

    print('method   :', result['method'])
    print('dOI/step :', np.round(result['dOI'], 3))
```

### Low-level mapping

The two low-level mappers are called internally by `dynamic_operability` but are
also exposed directly: `dynamic_operability_mapping` propagates the state-space
polytope (and uses the exact linear fast path when `A`, `B`, `C` are given), and
`dynamic_operability_nstep` builds the funnel in output space by simulation for
high-dimensional states. The examples below reuse the `AIS` and `DOS` defined in
the `dynamic_operability` example above.

```{eval-rst}
.. autofunction:: opyrability.dynamic_operability_mapping
```

###### Example
Build the funnel directly from the LTI matrices:
```{code-cell}
    from opyrability import dynamic_operability_mapping

    A = 0.9 * np.eye(2)
    mapping = dynamic_operability_mapping(x0=np.zeros(2),
                                          AIS_bound=AIS,
                                          AIS_resolution=3,
                                          k_max=4,
                                          A=A, B=np.eye(2), C=np.eye(2),
                                          plot=False)
    print('funnel slices:', len(mapping['AOS_regions']))
```

```{eval-rst}
.. autofunction:: opyrability.dynamic_operability_nstep
```

###### Example
Output-space funnel by simulation, for an arbitrary step model `step(x, u)`:
```{code-cell}
    from opyrability import dynamic_operability_nstep

    def integrator_step(x, u):
        x_next = np.asarray(x, float) + 0.1 * np.asarray(u, float)
        return x_next, x_next

    ns = dynamic_operability_nstep(integrator_step,
                                   np.zeros(2),
                                   AIS,
                                   k_max=4,
                                   AIS_resolution=3,
                                   plot=False)
    print('n-step slices:', len(ns['AOS_regions']))
```

```{eval-rst}
.. autofunction:: opyrability.dynamic_operability_scenarios
```

###### Example
Funnels across two disturbance scenarios and their robust intersection:
```{code-cell}
    from opyrability import dynamic_operability_scenarios

    def make_step(d):
        def step(x, u):
            x_next = 0.9 * np.asarray(x, float) + np.asarray(u, float) + d
            return x_next, x_next
        return step

    def make_x0(d):
        return np.zeros(2)

    scenarios = dynamic_operability_scenarios(make_step,
                                              make_x0,
                                              AIS,
                                              scenarios={'low': -0.1,
                                                         'high': 0.1},
                                              DOS=DOS,
                                              k_max=4,
                                              method='nstep',
                                              plot=False)
```

### Index evaluation, plotting and Monte Carlo

```{eval-rst}
.. autofunction:: opyrability.dOI_eval
```

###### Example
Score the funnel against the DOS at each time step:
```{code-cell}
    from opyrability import dOI_eval

    dOI = dOI_eval(mapping, DOS, plot=False)
    print('dOI per step:', np.round(dOI, 2))
```

```{eval-rst}
.. autofunction:: opyrability.plot_dynamic_funnel
```

###### Example
Stack the output-space slices into the dOI-colored funnel:
```{code-cell}
    from opyrability import plot_dynamic_funnel

    fig, ax = plot_dynamic_funnel(mapping, DOS=DOS, dOI=dOI)
```

```{eval-rst}
.. autofunction:: opyrability.plot_state_funnel
```

###### Example
The same funnel viewed in state space:
```{code-cell}
    from opyrability import plot_state_funnel

    fig, ax = plot_state_funnel(mapping)
```

```{eval-rst}
.. autofunction:: opyrability.plot_funnel_comparison
```

###### Example
Overlay the funnels of two initial states:
```{code-cell}
    from opyrability import plot_funnel_comparison

    mapping_shifted = dynamic_operability_mapping(x0=np.array([0.3, -0.3]),
                                                  AIS_bound=AIS,
                                                  AIS_resolution=3,
                                                  k_max=4,
                                                  A=A, B=np.eye(2), C=np.eye(2),
                                                  plot=False)
    fig, ax = plot_funnel_comparison({'x0 = 0': mapping,
                                      'x0 shifted': mapping_shifted})
```

```{eval-rst}
.. autofunction:: opyrability.simulate_mc_trajectories
```

###### Example
Sample Monte Carlo input-sequence trajectories through the funnel:
```{code-cell}
    from opyrability import simulate_mc_trajectories

    mc = simulate_mc_trajectories(mapping, n_trajectories=20, seed=0)
    print('trajectories shape:', mc.shape)
```

```{eval-rst}
.. autofunction:: opyrability.update_dynamic_funnel
```

###### Example
Re-center a linear funnel on a new initial state by a hyperplane shift, with no
re-simulation:
```{code-cell}
    from opyrability import update_dynamic_funnel

    updated = update_dynamic_funnel(mapping,
                                    x0_new=np.array([0.2, -0.2]),
                                    DOS=DOS)
    print('updated dOI:', np.round(updated['dOI'], 2))
```

### Gaussian-robust funnels and LTI utilities

```{eval-rst}
.. autofunction:: opyrability.propagate_output_covariance
```

###### Example
Propagate a Gaussian disturbance covariance to the output at each step:
```{code-cell}
    from opyrability import propagate_output_covariance

    Sigma_y = propagate_output_covariance(A,
                                          0.1 * np.eye(2),
                                          np.eye(2),
                                          np.eye(2),
                                          k_max=4)
    print('Sigma_y[0]:\n', np.round(Sigma_y[0], 3))
```

```{eval-rst}
.. autofunction:: opyrability.gaussian_robust_funnel
```

###### Example
Shrink each funnel slice so it stays achievable under the Gaussian uncertainty:
```{code-cell}
    from opyrability import gaussian_robust_funnel

    robust = gaussian_robust_funnel(mapping,
                                    Sigma_y,
                                    confidence=0.95,
                                    DOS=DOS)
    print('robust volumes:', np.round(robust['volumes'], 3))
```

```{eval-rst}
.. autofunction:: opyrability.identify_lti_step_tests
```

###### Example
Identify an LTI model from step tests on a nonlinear step model:
```{code-cell}
    from opyrability import identify_lti_step_tests

    def step_model(x, u):
        x_next = 0.8 * np.asarray(x, float) + 0.5 * np.asarray(u, float)
        return x_next, x_next

    ident = identify_lti_step_tests(step_model,
                                    np.zeros(2),
                                    np.zeros(2),
                                    du=1.0,
                                    n_steps=30)
    dc_gain = (ident['C']
               @ np.linalg.inv(np.eye(ident['A'].shape[0]) - ident['A'])
               @ ident['B'])
    print('identified DC gain:\n', np.round(dc_gain, 2))
```

```{eval-rst}
.. autofunction:: opyrability.make_pyomo_step_model
```

###### Example
Wrap a Pyomo model builder into a step callable for the dynamic mapping
(illustrative; requires a Pyomo-compatible IPOPT solver):
```python
import pyomo.environ as pyo
from opyrability import make_pyomo_step_model

def build_step():
    m = pyo.ConcreteModel()
    m.si = pyo.RangeSet(0, 1)
    m.ui = pyo.RangeSet(0, 1)
    m.x_current = pyo.Param(m.si, initialize=0.0, mutable=True)
    m.u = pyo.Var(m.ui, initialize=0.0)
    m.x_next = pyo.Var(m.si, initialize=0.0)

    @m.Constraint(m.si)
    def dynamics(m, i):
        return m.x_next[i] == 0.5 * m.x_current[i] + m.u[i]
    m.obj = pyo.Objective(expr=0)
    return m

step = make_pyomo_step_model(build_step, n_x=2, n_u=2)
x_next, y = step([2.0, 4.0], [1.0, 1.0])   # -> [2.0, 3.0]
```

## Utilities

```{eval-rst}
.. autofunction:: opyrability.create_grid
```
###### Example
Creating a 2-dimensional discretized rectangular grid for given DOS bounds.

```{code-cell} 
    from opyrability import create_grid

    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])

    DOS_resolution =  [3, 3]

    DOS_points = create_grid(DOS_bounds, DOS_resolution)

    print(DOS_points)
```
Visualizing this grid:
```{code-cell}
    import matplotlib.pyplot as plt

    DOS_points = DOS_points.reshape(-1, 2)

    plt.figure()
    plt.scatter(DOS_points[:, 0], DOS_points[:, 1])
```


```{eval-rst}
.. autofunction:: opyrability.points2simplices
```

###### Example
Generating paired simplicial polytopes for the AIS/AOS generated for the
shower problem example.

```{code-cell}
    from opyrability import AIS2AOS_map
    from opyrability import points2simplices

    # Obtain an input-output mapping using AIS2AOS_map
    AIS_bounds =  np.array([[0, 10], [0, 10]])
    resolution =  [5, 5]

    # The shower_problem function is the same one from the AIS2AOS_map example.
    AIS, AOS =  AIS2AOS_map(shower_problem, AIS_bounds,  resolution, plot = False)
    
    # Obtain simplices.
    AIS_poly, AOS_poly = points2simplices(AIS,AOS)

    print('AIS Simplices \n', AIS_poly)
    print('AOS Simplices \n', AOS_poly)
```



```{eval-rst}
.. autofunction:: opyrability.points2polyhedra
```
###### Example
Generating paired polyhedrons for the AIS/AOS generated for the
shower problem example.

```{code-cell}
    from opyrability import points2polyhedra
    
    AIS_poly, AOS_poly = points2polyhedra(AIS,AOS)

    print('AIS Polyhedrons \n', AIS_poly)
    print('AOS Polyhedrons \n', AOS_poly)
```

## Polytopic manipulations (advanced and internal use)

The functions below are fundamental for operability calculations, though typical 
users won't need to directly interact with them. They play a crucial role within
opyrability without requiring user intervention, but are documented here 
nevertheless.

```{eval-rst}
.. autofunction:: opyrability.get_extreme_vertices
```

```{eval-rst}
.. autofunction:: opyrability.process_overlapping_polytopes
```

```{eval-rst}
.. autofunction:: opyrability.are_overlapping
```

### API documentation list
```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   opyrability
```

