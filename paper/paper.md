---
title: 'Opyrability: A Python package for process operability analysis'
tags:
  - Python
  - Process systems engineering
  - Process operability
  - Process design and control
  - Computational geometry
  - Nonlinear optimization
authors:
  - name: Victor Alves
    affiliation: 1
  - name: San Dinh
    affiliation: "1, 2"
  - name: John R. Kitchin
    affiliation: 2
  - name: Vitor Gazzaneo
    affiliation: 1
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

date: 12 September 2023
bibliography: references.bib
---

# Summary

When designing a chemical process/plant, two main tasks
naturally arise when considering the processing of 
raw materials into value-added products such as chemicals
or energy:

1.	**Process design decisions**: Which decisions should be
	made with respect to the design variables of a given process,
	in a way that its overall objectives are
	achieved? (e.g., economic profitability, constraints related to
	product purity/pollutant emissions, sustainability, etc.).

2.	**Process control objectives**: Which variables
	should be controlled, yielding the maximum operability of the process?
	That is, can the process reach its maximum operational capacity, given the 
	ranges of the manipulated/input variables when subject to disturbances?

Historically, Tasks 1 and 2 have been performed sequentially: Engineers/practitioners would come up with the design decisions, 
and only then the control objectives would be assessed. Unfortunately, this can 
yield a process that is designed in a way that its operability capabilities
are hindered. In other words, because the control objectives were
not considered early in the design phase, the process itself might be not
controllable or operable at all. To give some perspective on how challenging this
problem can be, there are reports dating back to the 1940s from well-known authors in the process control field such as Ziegler and Nichols 
[@ziegler1943process] mentioning the importance of interconnecting design and control.

Considering this, the need of quantifying achievability for a general nonlinear
process naturally arises. The underlying motivation of determining whether it would be possible to measure the operability of a process to simultaneously achieve process design and control objectives led Georgakis and coworkers [@georgakis00;@vinson00;@siva05;@lima10]
to formally define *process operability*
and a metric called the *Operability Index (OI)*. The OI was conceptualized as a measure to quantify achievability of nonlinear processes [@vinson00], which was proven to be independent of the control strategy and inventory control layer [@vinson02]. In addition, it allows for the efficient ranking of competing designs and/or control structures [@lima10b] and enables the systematic assessment of operability characteristics under the presence of disturbances. Hence,
process operability was formalized as *a systematic framework to simultaneously assess design and control objectives early in the conceptual phase of industrial, typically large-scale, and nonlinear chemical processes.*

To achieve the systematic assessment of design and control objectives simultaneously, 
process operability is based on the definition of *operability sets*. These are spaces in the cartesian system that are defined with respect to the available inputs of a given process, their respective achievable outputs, the desired regions of operation in the input and output spaces, and lastly, any expected disturbances that may be present. The thorough definitions of these spaces are
readily available in the literature [@gazzaneo20], as well as in the opyrability [documentation](https://codes-group.github.io/opyrability/operability_overview.html).

Therefore, ``opyrability`` is a Python package for process operability analysis and calculations, with its
API designed to provide a user-friendly interface to enable users to perform process operability analysis seamlessly. This has the aim of reducing the complexity of dealing with the programming aspects of nonlinear programming [@carrasco16] and computational geometry [@gazzaneo19] operations needed when performing process operability analyses. 


# Statement of need

``Opyrability`` corresponds to a unified software tool to perform process operability
analysis in a single-bundle fashion. In broader terms, opyrability
provides a formal and mathematically tractable framework to systematically investigate the
operability and achievability of industrial processes earlier in the conceptual phase. This
eliminates the need for resorting to *ad-hoc*-type solutions to the design and control
of industrial processes, which are inherently with loss of generality. The use of this framework thus guarantees a solution to the operability problem that
is generalizable to any process, as long as a mathematical model of the given application is available.
Hence, the introduction of ``opyrability`` in Python, a widely used and freely available programming language, is a significant advancement in the process operability field. Being open-source and hosted in a community-driven environment, it offers a valuable resource to the process systems engineering, computational catalysis and material sciences communities that would benefit from operability direct/inverse mappings. This package empowers researchers and practitioners to easily investigate the operability aspects of both emerging and existing large-scale industrial processes. Additionally, on a lab scale, it can aid in the examination of material properties that guide design decisions, such as reactions rate and membrane parameters that would be needed to reach certain product specifications.

Moreover, ``opyrability`` is built on well-known and developed packages such as (i) [numpy](https://numpy.org/) and (ii) [scipy](https://scipy.org/) for linear algebra and scientific computing; (iii) [matplotlib](https://matplotlib.org/) for visualizing the operable regions in 2D/3D; (iv) [cvxopt](https://cvxopt.org/) that allows access to 
[glpk](https://www.gnu.org/software/glpk/) for linear programming; (v) [polytope](https://tulip-control.github.io/polytope/) that enables efficient polytopic calculations; and (vi) [cyipopt](https://cyipopt.readthedocs.io/en/latest/?badge=latest) that allows access to IPOPT [@Wachter2006], a state-of-the-art
nonlinear programming solver, enabling efficient inverse mapping operations within the operability framework. The inverse mapping task is further extended with full support for automatic differentiation, powered by JAX [@jax2018github]. This effort thus facilitates the 
dissemination of operability concepts and calculations in process systems engineering and other fields. \autoref{fig:fig1}
illustrates the dependency graph for ``opyrability``.


![Dependency graph generated with [pydeps](https://github.com/thebjorn/pydeps/) illustrating all numerical packages and visualization tools used in ``opyrability``.\label{fig:fig1}](./images/dependencies_opyrability.pdf)

# Vignette

As a quick illustration of ``opyrability's`` capabilities, the example below available
in the [examples gallery of the proposed tool](https://codes-group.github.io/opyrability/examples_gallery/index_example_gallery.html)
depicts the operability analysis of a continuous stirred tank reactor. In this example,
the OI is evaluated for a desired region of operation for the concentration of reactants A and B (as outputs). In particular, it is desired to obtain insight
on the design and operating region of this process, in terms of the reactor
radius and its operating temperature (as inputs). Process operability is employed to systematically analyze which designs and operating temperatures are able to
attain the requirements related to the concentrations of A and B.

The fundamental idea of ``opyrability`` is to be an environment for engineers
and scientists that eliminates the burden of dealing with the
transitioning among different software packages and environments to perform process
operability calculations. In
addition, the knowledge about computational geometry and constrained nonlinear 
programming can be limited to only theoretical rather than having the users 
implement the operability algorithms since ``opyrability`` already encapsulates all
the necessary calculations.

In the example below, the user only needs to: (i) have simple Python programming knowledge, limited to be able to perform mathematical modeling
and manipulate numpy arrays; and (ii) be able to interact with ``opyrability's``
functions, namely ``multimodel_rep``, ``OI_eval`` and ``nlp_based_approach``, as shown in the [API documentation](https://codes-group.github.io/opyrability/api.html).
\autoref{fig:cstr1} illustrates the process and example in focus.


![``Opyrability`` multimodel representation. (A) Chemical reactor schematic. (B) Jupyter notebook illustrating the use of the ``multimodel_rep`` and  ``OI_eval`` functions, as well as the set-up of these. (C) Visualization of the Achievable Output Set for the continuous stirred tank reactor example including the operable boundaries of the process studied. (D) Quantification of the Operability Index (OI), in which ``opyrability`` calculates that 39.14% of the desired operation can be achieved.\label{fig:cstr1}](./images/cstr_process_1.pdf)

Lastly, \autoref{fig:cstr2} depicts the use of ``opyrability's`` inverse mapping
features by using the ``nlp_based_approach`` function, allowing the user to obtain from a desired region in the output space, the region
in the input space that guarantees the desired operation.

![``opyrability's`` inverse mapping using ``nlp_based_approach``, in which the input space that guarantees the desired output set region is attained.\label{fig:cstr2}](./images/cstr_process_2.pdf)

# Availability

``Opyrability`` is freely available in both [PyPI](https://pypi.org/project/opyrability/) and [conda](https://anaconda.org/codes-group/opyrability) stores, as well as 
have its source code hosted on [GitHub](https://github.com/CODES-group/opyrability). In addition, its documentation contains
not only a thorough [description of the API](https://codes-group.github.io/opyrability/api.html), but also a [theoretical background discussion](https://codes-group.github.io/opyrability/operability_overview.html)
on process operability concepts, an [examples gallery](https://codes-group.github.io/opyrability/examples_gallery/index_example_gallery.html), and [instructions](https://codes-group.github.io/opyrability/process_model.html) on how to set up a process model
following ``opyrability`` design principles. The idea is to supply both proper documentation to
the users in the open-source software community as well as to give the users the necessary amount of theory allowing them to employ process operability principles in their specific application.
 
# Acknowledgements

The authors acknowledge the support from the National Science Foundation CAREER Award 1653098.

# References
