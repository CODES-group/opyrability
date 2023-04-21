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


# API documentation

The functions below are part of the pypo module and are
separted below based on their functionality.

## Conventional mapping (AIS to AOS)

### Forward mapping

```{eval-rst}
.. autofunction:: pypo.AIS2AOS_map
```

###### Example
Obtaining the Achievable Output Set (AOS) for the shower problem.

Importing PyPO and Numpy:
```{code-cell} 
    from pypo import AIS2AOS_map
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = \frac{(60+120)}{2}
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
    resolution =  [5, 5]
```

Obtain discretized AIS/AOS.

```{code-cell}
    AIS, AOS =  AIS2AOS_map(shower_problem, AIS_bounds,  resolution)
    print(AOS)
```

### Inverse mapping (AOS/DOS to AIS/DIS)

### NLP-Based

```{eval-rst}
.. autofunction:: pypo.nlp_based_approach
```

###### Example
Obtaining the Feasible Desired Input Set (DIS*) for the shower problem.

Importing PyPO and Numpy:

```{code-cell}
    import numpy as np
    from pypo import nlp_based_approach
```

Defining lower and upper bound for the AIS/DIS inverse map:

```{code-cell}
    lb = np.array([0, 0])
    ub = np.array([100,100])
```
Defining DOS bounds and resolution to obtain the inverse map:

```{code-cell}
    DOS_bound = np.array([[17.5, 21.0],
                          [80.0, 100.0]])
    resolution = [10, 10]
```

Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = \frac{(60+120)}{2}
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
Additionally, using IPOPT as NLP solver, enabling plotting of the process operability
sets, cold-starting the
NLP and using finite-differences:

```{code-cell}
    
    u0 = u0 = np.array([0, 10]) # Initial estimate for inverse mapping.
    fDIS, fDOS, message = nlp_based_approach(DOS_bound, 
                                             resolution, 
                                             shower_problem, 
                                             u0, 
                                             lb,
                                             ub, 
                                             method='ipopt', 
                                             plot=True, 
                                             ad=False,
                                             warmstart=False)
```


## Implicit mapping

```{eval-rst}
.. autofunction:: pypo.implicit_map
```

## Multimodel representation

```{eval-rst}
.. autofunction:: pypo.multimodel_rep
```

###### Example
Obtaining the Achievable Output Set (AOS) for the shower problem.

Importing PyPO and Numpy:
```{code-cell} 
    from pypo import multimodel_rep
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = \frac{(60+120)}{2}
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
    AIS_bounds =  np.array([[0, 10], [0, 10]])
    AIS_resolution =  [5, 5]
```

Obtaining multimodel representation of paired polytopes for the AOS:

```{code-cell} 
    AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, shower_problem)
```

## OI evaluation

```{eval-rst}
.. autofunction:: pypo.OI
```

###### Example
Evaluating the OI for the shower problem for a given DOS.

Importing PyPO and Numpy:
```{code-cell} 
    from pypo import multimodel_rep, OI
    import numpy as np
```
Defining the equations that describe the process:

```{math}
\left\{\begin{array}{c}
y_1=u_1+u_2 \\
y_2=\frac{\left(60 u_1+120 u_2\right)}{\left(u_1+u_2\right)}
\end{array}\right. \\
\\
y_1 = 0\rightarrow y_2 = \frac{(60+120)}{2}
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
    AIS_bounds =  np.array([[0, 10], [0, 10]])
    AIS_resolution =  [10, 10]
```

Obtaining multimodel representation of paired polytopes for the AOS:

```{code-cell} 
    AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, shower_problem)
```

Defining a DOS region between $y_1 =[10-20], y_2=[70-100]$
```{code-cell} 
    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])
```

Evaluating the OI and seeing the intersection between the operability sets:
```{code-cell} 
    OI = OI(AOS_region, DOS_bounds)
```
## Utilities

```{eval-rst}
.. autofunction:: pypo.create_grid
```
###### Example
Creating a 2-dimensional discretized rectangular grid for given DOS bounds.

```{code-cell} 
    from pypo import create_grid
    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])

    DOS_resolution =  [5, 5]

    DOSPts = create_grid(DOS_bounds, DOS_resolution)

    print(DOSPts)
```
Visualizing this grid:
```{code-cell}
    import matplotlib.pyplot as plt
    DOSPts = DOSPts.reshape(-1, 2)
    plt.scatter(DOSPts[:, 0], DOSPts[:, 1])
```


```{eval-rst}
.. autofunction:: pypo.points2simplices
```

###### Example
Generating paired simplicial polytopes for the AIS/AOS generated for the
shower problem example.

```{code-cell}
    from pypo import points2simplices
    AIS_poly, AOS_poly = points2simplices(AIS,AOS)
    print('AIS Simplices \n', AIS_poly)
    print('AOS Simplices \n', AOS_poly)
```



```{eval-rst}
.. autofunction:: pypo.points2polyhedra


```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pypo
```

