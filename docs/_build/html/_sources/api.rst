API documentation
=================
The functions below are part of the pypo module and are
separted below based on their functionality.

Conventional mapping (AIS to AOS)
---------------------------------

Forward mapping
~~~~~~~~~~~~~~~~
.. autofunction:: pyprop.AIS2AOS_map

Inverse mapping (AOS/DOS to AIS/DIS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NLP-Based
~~~~~~~~~
.. autofunction:: pyprop.nlp_based_approach

Implicit mapping
-----------------
.. autofunction:: pyprop.implicit_map

OI evaluation
-------------
.. autofunction:: pyprop.OI_calc

Multimodel representation
--------------------------
.. autofunction:: pyprop.multimodel_rep

Example
~~~~~~~~

```{code-cell}
print(2 + 2)
```



Utilities
---------
.. autofunction:: pyprop.create_grid

.. autofunction:: pyprop.points2simplices

.. autofunction:: pyprop.points2polyhedra



.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyprop
   