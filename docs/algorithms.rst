Process Operability Algorithms
===============================

There are two main operations that may be needed to be performed to perform an 
operability analysis.

#. **Operability sets quantification:** Obtaining and quantifying the AIS, AOS, DOS,
   DIS, EDS, and so on. This yields insights into the achievability of a given process.
   If the analysis is in low dimensions (:math:`\leq3`) the analysis can 
   even be visual. In higher dimensions, the OI still serves as a valuable metric to 
   assess the operability of a process. Irrespective of dimensionality, computational
   geometry algorithms and polytopic calculations are necessary.
#. **Inverse mapping:** From a DOS, evaluate the corresponding DIS. For a general nonlinear
   process model, frequently represented in mathematical terms as a nonlinear system of 
   equation as a vector-valued function, this may be a non-trivial task.

Luckily, algorithms have been developed in the literature to address these challenges:
For quantification of the operability sets and thus, the OI itself, the multimodel approach
has been developed. For inverse mapping, the nonlinear programming-based (NLP-based) approach
and the implicit mapping approach have been successfully employed to evaluate the inverse map
of a given process model. This section will go briefly over these methods but the
reader is encouraged to go over the [literature](bibliography.rst) for a more thorough explanation of them.


The multimodel approach 
------------------------

The multimodel approach :cite:`gazzaneo18,gazzaneo19,gazzaneo20` employs 
the paradigm of a series of polytopes being able
to represent any nonlinear space as a collection of such polytopes. This approach 
simplifies the obtainment of operability sets since polytopes are by definition convex and,
therefore, can be represented in half-space representation (:math:`\mathcal{H}-rep`) or in
vertex representation (:math:`\mathcal{V}-rep`), being a system of linear 
inequality constraints as a direct consequence of this.

**Pictorial Example**

Let's consider the schematic below, in which each point corresponds to a coordinate 
in the input space and their respective coordinate in the output space, obtained
through the process model :math:`M`. Paired polytopes can be "drawn" to represent
the AOS accordingly:

.. figure:: ./images/multimodel_01.gif
   :align: center

   AIS-AOS representation using paired polytopes.

In the animation above, :math:`P_1^u` is paired with :math:`P_1^y` and so on, in
the general form :math:`P_k = \{P_k^u,P_k^y\}`

Another example in which the non-convex AOS is approximated as a series of paired
polytopes can be depicted in the animation below, in which one can see that the
polytopes approximate the overall non-convex AOS region with relative accuracy,
as well as the DOS and DIS:


.. figure:: ./images/multimodel_02.gif
   :align: center

   AIS-AOS polytopic approximation.

Due to its intrinsic roots in computational geometry and linear programming,
the multimodel approach is a suitable process operability algorithm for

#. **Replacement of non-convex regions with paired polytopes**, allowing efficient OI 
   computation and representation of the operability sets.
#. **OI evaluation**, in which the multimodel approach allows to rank 
   competing designs and control structures.

The nonlinear programming-based (NLP-based) approach 
-----------------------------------------------------

The NLP-based approach :cite:`carrasco16,carrasco2017` converts the inverse mapping 
task into a nonlinear program.
The premise of this algorithm is that the DOS can be discretized into a series of
coordinate points in the output space (AOS/DOS) and that an objective function of 
error minimization nature (or sort of Euclidean distance if you prefer) is posed 
between the feasible operation and desired operation (DOS). The solution of the
nonlinear program at each discretized point is the DIS that attains the operation
of the DOS. Mathematically, this can be posed as the following optimization problem:

.. math::
   \begin{gathered}
   \emptyset_{\mathrm{k}}=\min _{\mathrm{u}_k^*} \sum_{j=1}^n\left(\left(\mathrm{y}_{j, k}-\mathrm{y}_{j, k}^*\right) / \mathrm{y}_{j, k}\right)^2 \\
   \text { s.t: Process model } (M) \\
   \mathrm{u}_k^{\min } \leq \mathrm{u}_k^* \leq \mathrm{u}_k^{\max } \\
   \mathbf{c}_1\left(\mathrm{u}_k^*\right) \leq 0
   \end{gathered}

for :math:`j=1:n` output variables in the AOS/DOS, and  :math:`k` discretized points.
Lastly, nonlinear constraints :math:`c`  might be imposed to the inverse map problem if
needed: These can be a result of product specifications, equipment limitations and so on.


**Pictorial Example**

In the animation below, the optimization problem is posed for each discretized DOS
point. Then, for each point, a corresponding feasible DIS solution is obtained.
Due to the nature of the objective function (error/distance minimization) the DOS
points will be shifted as close as possible to feasible operation:

.. figure:: ./images/nlp_01.gif
   :align: center

   Inverse mapping using NLP-based approach.

The main features of the NLP-based approach are

#. Feasible operability sets, since the output points are shifted to be as close as possible to 
   feasible operation.
#. Useful to search for new AIS unexplored regions, giving insights about process feasibility.
#. Imposing constraints to the inverse mapping operation.

Since a sucessfull solution of an NLP is always feasible, the DOS and DIS that achieve
the error minimzation between feasible and desired operation are named slightly different as
the Feasible Desired Output Set (DOS*) and Feasible Desired Input Set (DIS*).

Lastly, the NLP-based approach can be extended to encompass the search for intensified
and/or modular designs, as proposed in the literature :cite:`carrasco2017`.