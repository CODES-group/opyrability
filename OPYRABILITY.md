# OPYRABILITY.md

A guide for **coding agents** (Claude Code, Cursor, Copilot, and friends) to drive
**process operability analysis** with [opyrability](https://github.com/CODES-group/opyrability).
Point your agent at this file, then ask it to map operability sets, score the
Operability Index, invert a design, rank candidates, or build a dynamic funnel,
and it will use the API correctly the first time.

> Drop this file in your project root (or `@`-mention it). It is self-contained:
> the API surface, the model contract, copy-paste recipes, and the mistakes to avoid.

---

## What opyrability does

opyrability answers operability questions about a process model: given the inputs a
unit can manipulate, which outputs can it reach, and can it reach the ones you want?
It maps the Available Input Set (AIS) to the Achievable Output Set (AOS), quantifies
the overlap with a Desired Output Set (DOS) as the **Operability Index (OI)**, inverts
a desired output region back to the feasible inputs/designs, intensifies designs, and
extends all of this to dynamics (achievable-output **funnels** over time). It is
model-agnostic: first-principles, data-driven surrogate (GP/neural net), or
equation-oriented (Pyomo/OMLT) models all work.

Reach for it when you have a model `y = M(u)` (or a dynamic `x_{k+1} = f(x_k, u_k)`)
and want to **quantify operability**, **compare/rank designs**, **find feasible or
intensified designs** for an output target, or **analyze whether and how fast** a
process can be driven to a target.

## Install

```bash
pip install opyrability          # includes the default Pounce NLP solver
```

- **Optional IPOPT backend** (`method='ipopt'`): `conda install -c conda-forge cyipopt`
- **Optional algebraic, equation-oriented models** (Pyomo/OMLT): `pip install "opyrability[pyomo]"`
- **Optional interactive funnels** (`engine='plotly'`): `pip install "opyrability[plotly]"`

The default solver, [Pounce](https://kitchingroup.cheme.cmu.edu/pounce/), is a pure-Rust
port of IPOPT installed from wheels, so the inverse mapping works out of the box.

## The vocabulary

| Set | Meaning |
|---|---|
| **AIS** | Available Input Set: inputs you can manipulate (bounds) |
| **AOS** | Achievable Output Set: outputs reachable from the AIS through the model |
| **DOS** | Desired Output Set: the outputs you want (target region) |
| **DIS** | Desired Input Set: inputs that achieve the DOS (`DIS*` when feasible) |
| **EDS** | Expected Disturbance Set: disturbance realizations (optional) |
| **OI**  | Operability Index `= μ(AOS ∩ DOS) / μ(DOS) × 100%`; 100% means fully operable |

## The model contract (read this first)

Provide your process as **one** of these:

- **Steady-state callable** -- `def model(u: np.ndarray) -> np.ndarray`. For JAX
  automatic differentiation (`ad=True`) write it with `jax.numpy`.
- **Equation-oriented Pyomo builder** -- `def model(m, u, y): ...` that adds the
  constraints, flagged with `model.build_pyomo_constraints = True` (auto-detected).
- **Dynamic step model** -- `def step(x, u) -> (x_next, y)` (or `step(x, u, d)` with a
  disturbance). It must return **both** the next state and the output.
- **Linear time-invariant (dynamic)** -- a dict `{'A': A, 'B': B, 'C': C}` (optionally
  `'B_d'`); no step function needed.

Bounds are `np.ndarray` of shape `(n, 2)`, i.e. `[[lo, hi], ...]`. Resolutions are an
`int` or a per-dimension list like `[5, 5]`.

## Key functions (when to use each)

| Function | Use it to | Returns |
|---|---|---|
| `AIS2AOS_map(model, AIS, resolution)` | discretize the forward map (raw points) | `(AIS, AOS)` arrays |
| `multimodel_rep(model, AIS, resolution)` | build the AOS as paired polytopes | `[pc.Region, coords]` |
| `OI_eval(AS, DS)` | score the Operability Index | `float` in `[0, 100]` |
| `rank_designs(models, AIS, DOS, resolution)` | rank competing designs by OI | list sorted by OI |
| `nlp_based_approach(model, DOS, res, u0, lb, ub)` | inverse map; feasible (P1) or intensified (P2/P3) designs | `(fDIS, fDOS, messages)` |
| `milp_based_approach(model, AIS_bound, PI_target, DOS_bounds, AIS_resolution)` | fast optimal modular design (MILP) | `(u_opt, y_opt, phi, phi_true, history)` |
| `implicit_map(model, image_init, domain_bound, domain_resolution)` | fast forward/inverse map of an `F(u, y) = 0` model | `(domain, image, ...)` |
| `dynamic_operability(model, x0, AIS, DOS=...)` | achievable-output funnel + dOI over time | results `dict` |

Lower-level dynamic helpers: `dynamic_operability_mapping`, `dynamic_operability_nstep`,
`dynamic_operability_scenarios`, `dOI_eval`, `plot_dynamic_funnel`, `plot_state_funnel`,
`plot_funnel_comparison`, `simulate_mc_trajectories`, `gaussian_robust_funnel`,
`propagate_output_covariance`, `identify_lti_step_tests`, `make_pyomo_step_model`.

## Recipes

A worked model used below (the classic "shower problem", inputs = cold/hot flow,
outputs = total flow and temperature):

```python
import numpy as np

def shower(u):
    y = np.zeros(2)
    y[0] = u[0] + u[1]
    y[1] = (60 * u[0] + 120 * u[1]) / (u[0] + u[1]) if y[0] != 0 else 90.0
    return y

AIS = np.array([[1.0, 10.0], [1.0, 10.0]])
DOS = np.array([[10.0, 20.0], [70.0, 100.0]])
```

### 1. Forward map + Operability Index

```python
from opyrability import multimodel_rep, OI_eval

AOS_region = multimodel_rep(shower, AIS, [5, 5], plot=True)   # -> [pc.Region, coords]
OI = OI_eval(AOS_region, DOS, plot=True)                      # -> ~60.2 (%)
```

### 2. Inverse mapping -- feasible designs/inputs for a target (P1)

```python
from opyrability import nlp_based_approach

fDIS, fDOS, messages = nlp_based_approach(
    shower, DOS, [5, 5],
    u0=np.array([5.0, 5.0]),
    lb=np.array([0.0, 0.0]),
    ub=np.array([100.0, 100.0]))
# method='pounce' is the default. method='ipopt' needs cyipopt. ad=True needs a JAX model.
# Each fDIS row maps through the model to its fDOS row: shower(fDIS[i]) == fDOS[i].
```

### 3. Process intensification and fast optimal design

```python
from opyrability import milp_based_approach

u_opt, y_opt, phi, phi_true, history = milp_based_approach(
    shower,
    AIS_bound=AIS,
    PI_target=lambda u: u[0] + u[1],     # minimize total water usage
    DOS_bounds=DOS,
    AIS_resolution=5)
# NLP route to the same idea: nlp_based_approach(..., problem='P2', PI_target=..., PI_bounds=...)
```

### 4. Rank competing designs by the Operability Index

```python
from opyrability import rank_designs

ranking = rank_designs(
    {'Design A': model_a, 'Design B': model_b},   # any models with the same output space
    AIS_bound=AIS,                                 # or a dict {label: bounds} for per-design AIS
    DOS_bound=DOS,
    resolution=[10, 10],
    perspective='outputs')                         # 'inputs' inverts the DOS instead
# ranking[0] is the most operable design; each entry is {'label', 'OI', 'region'}.
```

### 5. Dynamic operability funnel

```python
from opyrability import dynamic_operability

# Linear time-invariant: pass matrices, no step function needed.
model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
AIS_dyn = np.array([[-1.0, 1.0], [-1.0, 1.0]])
DOS_dyn = np.array([[-2.0, 2.0], [-2.0, 2.0]])
res = dynamic_operability(model, x0=np.zeros(2), AIS_bound=AIS_dyn,
                          DOS=DOS_dyn, k_max=4)
print(res['dOI'])          # dOI per step, here [25. 90.25 100. 100.]

# Nonlinear: pass a step model step(x, u) -> (x_next, y) in place of the dict.
```

## Gotchas an agent must get right

- **`OI_eval`'s first argument is the `[pc.Region, coords]` list from `multimodel_rep`**,
  not a raw array. Its second argument is the desired-set **bounds** as an `np.ndarray`.
  Do not pass `AIS2AOS_map`'s raw arrays to `OI_eval`.
- **`multimodel_rep` returns a 2-element list** `[region, coords]`; use `[0]` for the
  polytope `Region` if you need it directly.
- **`method='pounce'` is the default** inverse-mapping solver (pure-Rust IPOPT, via pip).
  `method='ipopt'` requires `cyipopt` (conda). `ad=True` requires the model to be written
  with `jax.numpy`.
- **Bounds are `(n, 2)` arrays**; resolution is an `int` or a per-dimension list.
- **The dynamic step contract is `step(x, u) -> (x_next, y)`** -- return the next state
  AND the output. LTI systems use the `{'A', 'B', 'C'}` dict instead.
- **Plots**: matplotlib by default; pass `engine='plotly'` for interactive funnels;
  `plot=False` to suppress.
- **Pyomo models** need `model.build_pyomo_constraints = True`, and the forward proxy
  `AIS2AOS_map(..., output_dim=n)` needs the output dimension.

## Learn more

- Documentation and example gallery: <https://codes-group.github.io/opyrability/>
- API reference (every function with a runnable example) and the "Algorithms" page
  (multimodel, NLP-based P1/P2/P3, implicit mapping, MILP, and dynamic operability,
  with references).
- Developed by [Victor Alves](https://victor-alves.com) at Carnegie Mellon University; development began in the CODES Group at West Virginia University.
