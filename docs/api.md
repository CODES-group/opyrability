---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# API documentation

The functions below are part of the pypo module and are
separted below based on their functionality.

## Conventional mapping (AIS to AOS)

### Forward mapping

```{eval-rst}
.. autofunction:: pyprop.AIS2AOS_map
```

### Inverse mapping (AOS/DOS to AIS/DIS)

### NLP-Based

```{eval-rst}
.. autofunction:: pyprop.nlp_based_approach
```

## Implicit mapping

```{eval-rst}
.. autofunction:: pyprop.implicit_map
```

## OI evaluation

```{eval-rst}
.. autofunction:: pyprop.OI_calc
```

## Multimodel representation

```{eval-rst}
.. autofunction:: pyprop.multimodel_rep
```

#### Example

```{code-cell} 
    from pyprop import multimodel_rep, OI_calc
    import numpy as np
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

```{code-cell}
    DOS_bounds =  np.array([[10, 20], [70, 100]])
    AIS_bounds =  np.array([[0, 10], [0, 10]])
    AIS_resolution =  [5, 5]
```

Obtaining multimodel representation of paired polytopes for the AOS

```{code-cell}
   AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, shower_problem)
```

## Utilities

```{eval-rst}
.. autofunction:: pyprop.create_grid
```

```{eval-rst}
.. autofunction:: pyprop.points2simplices
```

```{eval-rst}
.. autofunction:: pyprop.points2polyhedra


```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyprop
```

