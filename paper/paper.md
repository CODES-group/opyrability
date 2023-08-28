---
title: 'Opyrability: A Python package for process operability analysis'
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
    affiliation: 2
  - name: Vitor Gazzaneo
    affiliation: 4
  - name: Juan C. Carrasco
    affiliation: 3
  - name: Fernando V. Lima
    affiliation: 1
  
affiliations:
 - name: Department of Chemical and Biomedical Engineering, West Virginia University, Morgantown, West Virginia, USA
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

When designing a chemical process/plant, two main tasks
naturally arise, when considering the processing of 
raw materials into value-added products such as chemicals
or energy:

1.	**Process design**: Which decisions should be
	made with respect to the design variables of a given process,
	in a way that its overall objectives are
	achieved (economic profitability, constraints related to
	product purity/pollutant emissions, etc.)?

2.	**Control objectives**: Which variables
	should be controlled, yielding the maximum operability of this process?
	That is, can the process "reach" its maximum operational capacity, given the 
	ranges of the manipulated/input variables?

Historically, tasks 1 and 2 were performed in a sequential manner:
Firstly, an engineer/practitioner would come up with the design decisions, 
and only then the control objectives are assessed. Unfortunately, this can 
yield a process that is designed in a way that its operability capabilities
are hindered. In other words, because the control objectives were
not considered early in the design phase, the process itself might be not
controlled or operable at all. To give you perspective on how challenging this
problem is, there are reports dating back to the 40's from Ziegler and Nichols 
[@ziegler1943process]
(The same ones from the controller tuning laws) stressing this problem,
mentioning the importance of interconnecting design and control.

With this in mind, the need of quantifying achievability of a general nonlinear
process naturally arises. The question: "Can one quantify achievability
of process design and control objectives simultaneously?" was the underlying motivation
for Prof. Christos Georgakis and his collaborators 
[@georgakis00;@vinson00;@siva05;@lima10]
to formally define **process operability**
and define a metric called the **operability index (OI) **. The OI, an inherent nonlinear
measure [@vinson00] that is independent of the control strategy and inventory control layer [@vinson02],
allows for efficient ranking of competing designs and/or control structures [@lima10b] and allows
for the systematic assessment of operability characteristics under disturbances. Hence,
process operability is formalized as **a systematic framework to simultaneously assess design and control objectives early in the conceptual phase of industrial, typically large-scale, and nonlinear chemical processes.**

To achieve the systematic assessment design and control objectives simultaneously, 
process operability is based on the definition of **operability sets**. These are spaces in the cartesian system that are defined with respect to the available inputs of a given process, their respective achievable outputs, the desired regions of operation in the input and output spaces and lastly, any expected disturbances that may be present. The thorough definitions of these spaces are
readily available in the literature [@gazzaneo20], as well as in our [documentation](https://codes-group.github.io/PyPO/operability_overview.html).

Therefore, ``opyrability`` is a Python package for process operability calculations, with its
API designed to provide a user-friendly interface to enable users to perform process operability analysis effortlessly, reducing the complexity of dealing with the programming aspects of nonlinear programming [@carrasco16] and computational geometry [@gazzaneo19], typical
operations needed when performing a process operability analysis. 


# Statement of need

``Opyrability`` corresponds to a unified approach to perform process operability
analysis in a single-bundle fashion type of package. In broader terms, process operability
gives a formal and mathematically tractable framework to systematically investigate the
operability and achievability of industrial processes earlier in the conceptual phase. This
eliminates the need for recurring to ***ad-hoc***-type solutions to the designing and control
of industrial processes, which are inherently with loss of generality. The use of the process
operability framework guarantees a solution to the operability and achievability problems that
is generalizable to any process, as long as a mathematical model of the given application is available.
Hence, the availability of a package such as ```opyrability``` in a popular, and freely
available programming language such as Python, provides the process systems engineering (PSE) community a package that enables
researchers and practitioners to focus on investigating the operability aspects of emerging and 
existent large-scale, industrial processes with ease, and to have it in an open-source and community-driven environment.
Secondly, ``opyrability`` is built on well-celebrated packages such as [numpy](https://numpy.org/),
[scipy](https://numpy.org/), for linear algebra and scientific computing; [matplotlib](https://matplotlib.org/) for visualizing the operable regions in 2D/3D; [cvxopt](https://cvxopt.org/) allowing access to 
[glpk](https://www.gnu.org/software/glpk/) for linear programming, enabling efficient polytopic calculations when using [polytope](https://tulip-control.github.io/polytope/). Lastly, [cyipopt](https://cyipopt.readthedocs.io/en/latest/?badge=latest) allows access to IPOPT [@Wachter2006], a state-of-the-art
nonlinear programming solver, allowing efficient inverse mapping operation within the operability framework. The inverse mapping task is further extended with full support for automatic differentiation, powered by JAX [@jax2018github]. This effort thus might further facilitate the 
dissemination of operability concepts in the PSE field. \autoref{fig:fig1}
illustrates the dependency graph for ``opyrability``.


![Dependency graph generated with [pydeps](https://github.com/thebjorn/pydeps/) illustrating all numerical packages and visualization tools that allow for ``opyrability`` to exist.\label{fig:fig1}](./images/dependencies_opyrability2.pdf)

# Vignette

As a quick example of ``opyrability's`` capabilities, the example below available
in the [examples gallery of the proposed tool](https://codes-group.github.io/PyPO/examples_gallery/index_example_gallery.html)
depicts the operability analysis of a continuous stirred tank reactor (CSTR), in 
which the operability index (OI) is evaluated for a desired region of operation for the concentration of reactants A and B, respectively. It is desired to obtain intuition
about the design and operating region of this process, in terms of the reactor
radius and its operating temperature. Process operability is employed to quickly
and systematically analyze which designs and operating temperatures are able to
attain the requirements related to the concentrations of A and B.

The fundamental idea of ``opyrability`` is to be an environment for chemical engineers
and researchers that reduces the burden of dealing with the
transitioning among different software packages and environments to perform process
operability calculations to a minimum, being a tool that allows the user to only
focus on the investigation of operability properties of industrial systems. In
addition, the knowledge about computational geometry and constrained nonlinear 
programming can be limited to be only theoretical rather than having the users 
implement the operability algorithms since ``opyrability`` already encapsulates all
the necessary calculations.

In the example below, the user only needs to possess (i) a fairly simple 
Python programming knowledge, limited to be able to perform mathematical modeling
and manipulate Numpy arrays, and (ii) be able to interact with ``opyrability's``
functions, namely ``multimodel_rep``, ``OI_eval`` and ``nlp_based_approach``.
\autoref{fig:cstr1} illustrates the process: 


![``Opyrability`` multimodel representation. (A) Chemical reactor schematic. (B) Jupyter notebook illustrating the use of the ``multimodel_rep`` and  ``OI_eval`` functions, as well as the set-up to use these. (C) Visualization of the Achievable Output Set (AOS) for the CSTR example: The user is able to easily visualize the frontiers of the process. (D) Quantification of the Operability Index (OI), in which ``opyrability`` calculates that only 39.14% of the desired operation can be achieved.\label{fig:cstr1}](./images/cstr_process_1.pdf)

Lastly, \autoref{fig:cstr2} depicts the use of ``opyrability's`` inverse mapping
features, allowing to obtain from a desired region in the output space, the region
in the input space that guarantees the desired operation:

![``opyrability's`` inverse mapping, namely ``nlp_based_approach``, in which the input space that guarantees the desired output set region is attained can be obtained.\label{fig:cstr2}](./images/cstr_process_2.pdf)

# Availability

``opyrability`` is freely available in both [PyPI](https://pypi.org/) and [conda-forge](https://conda-forge.org/) stores, as well as 
having its source code hosted on [GitHub](https://github.com/CODES-group/PyPO). In addition, its documentation contains
not only a thorough [description of the API](https://codes-group.github.io/PyPO/api.html) but also a [theoretical background discussion](https://codes-group.github.io/PyPO/operability_overview.html)
on process operability concepts, an [examples gallery](https://codes-group.github.io/PyPO/examples_gallery/index_example_gallery.html) with live code capabilities, and [instructions](https://codes-group.github.io/PyPO/process_model.html) on how to set up a process model
following ``opyrability`` design principles. The idea is not only to supply proper documentation to
the users in the PSE community but also to give the users the necessary amount of theory allowing them
to use process operability principles in their specific application.
 
# Acknowledgements

We acknowledge the support from the National Science Foundation CAREER Award 1653098.

# References
