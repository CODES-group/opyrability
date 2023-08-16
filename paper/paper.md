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
and define a metric called the **operability index**. Hence, the definition of
process operability is formalized as **a systematic framework to simultaneously assess design and control objectives early in the conceptual phase of industrial, typically large-scale, and nonlinear chemical processes.**

To achieve the systematic assessment design and control objectives simultaneously, 
process operability is based on the definition of **operability sets**. These are spaces in the cartesian system that are defined with respect to the available inputs of a given process, their respective achievable outputs, the desired regions of operation in the input and output spaces and lastly, any expected disturbances that may be present. The thorough definitions of these spaces are
readily available in the literature, as well as in our [documentation](https://codes-group.github.io/PyPO/operability_overview.html).

Therefore, ``operabilipy`` is a Python package for process operability calculations. ``operabilipy's``
API was designed to provide a user-friendly interface to enable users to perform process operability analysis effortlessly, reducing the complexity of dealing with the programming aspects of nonlinear programming and computational geometry, typical
operations needed when performing a process operability analysis. The main philosophy of ``operabilipy``
is to proved to the process systems engineering (PSE) community a package that would enable
researchers and practitioners to focus on investigating the operability aspects of emerging and 
existent large-scale, industrial processes with ease, and to have in an open-source, community-driven,
and freely available programming language such as Python. This effort thus might further facilitate the 
dissemination of operability concepts in the PSE field.


# Statement of need

Statement of need here.

# Availability

Availability of the tool here.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
