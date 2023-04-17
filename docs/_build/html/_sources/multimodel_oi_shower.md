---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (Spyder)
    language: python3
    name: python3
---

# Example

```python3
from pyprop import multimodel_rep, OI_calc
import numpy as np
```

Defining the equations for the shower problem


$$
y_1 =  u_1 + u_2 \\
y_2 = \frac{(60*u_1 +  120*u_2)}{(u_1 + u_2)} \\
if \, y_1 = 0, \\ then \, y_2 = \frac{(60+120)}{2}
$$

```python3
def shower_problem_2x2(u):
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y
```

Defining The AIS and DOS Bounds

```python3
DOS_bounds =  np.array([[10, 20], 
                        [70, 100]])

AIS_bounds =  np.array([[0, 10],
                        [0, 10]])

AIS_resolution =  [5, 5]

model =  shower_problem_2x2
```

Obtaining the AOS from the AIS

```python3
AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, model)
```

```python3
OI = OI_calc(AOS_region, DOS_bounds)
```

```python3

```
