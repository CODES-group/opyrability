---
title: 'Operabilipy: A Python package for process operability analysis'
tags:
  - Python
  - Process systems engineering
  - Process operability
  - Process design and control
  - Computational geometry
  - nonlinear optimization
authors:
  - name: Victor Alves
    affiliation: 1
  - name: San Dinh
    affiliation: 2
  - name: John R. Kitchin
    affiliation: 3
  - name: Vitor Gazzaneo
    affiliation: 4
  - name: Juan C. Carrasco
    affiliation: 3
  - name: Fernando V. Lima
    affiliation: 1
  
affiliations:
 - name: Department of Chemical and Biomedical Engineering, West Virginia University, 
   Morgantown, West Virginia, USA
   index: 1
 - name: Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, Pennsylvania, USA
   index: 2
 - name: Department of Chemical Engineering, Universidad de Concepción, Concepción, Chile
   index: 3
 - name: Air Products and Chemicals Inc., Allentown, Pennsylvania, USA
   index: 4

date: 15 August 2023
bibliography: references.bib
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

``Gala`` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for ``Gala`` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. ``Gala`` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the ``Astropy`` package [@astropy] (``astropy.units`` and
``astropy.coordinates``).

``Gala`` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in ``Gala`` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. The source code for ``Gala`` has been
archived to Zenodo with the linked DOI: [@zenodo]


# Statement of need

Statement of need here.

# Availability

Availability of the tool here.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
