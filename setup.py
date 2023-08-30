# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

modules = \
['opyrability', 'PolyhedraVolAprox']
install_requires = \
['cvxopt>=1.2.7,<2.0.0',
 'jax[cpu]>=0.4.13,<0.5.0',
 'matplotlib>=3.6.2,<4.0.0',
 'numpy>=1.24.1,<2.0.0',
 'polytope>=0.2.4,<0.3.0',
 'scipy>=1.10.0,<2.0.0',
 'tqdm>=4.64.1,<5.0.0']

setup_kwargs = {
    'name': 'opyrability',
    'version': '1.2',
    'description': 'Process operability analysis in Python',
    'long_description': "# opyrability - Process Operability Analysis in Python.\n\nWelcome to opyrability, a Python-based package for performing [Process Operability](https://www.sciencedirect.com/science/article/pii/S1474667017338028) analysis.\n\nopyrability is developed by the Control, Optimization and Design for Energy and Sustainability (CODES) Group at West Virginia University under Dr. [Fernando V. Lima](https://fernandolima.faculty.wvu.edu/)'s supervision.\n\n![](/docs/opyrability_overview.png)\n\nAuthors:\n[Victor Alves](https://github.com/victoraalves) and [San Dinh](https://github.com/sanqdinh)\n\n\n## Installation\n\nFrom PyPI:\n\n```console\npip install opyrability\n```\n\nThen install [Cyipopt](https://github.com/mechmotum/cyipopt) (non-automated dependency) and [Polytope's latest version hosted on GitHub](https://github.com/tulip-control/polytope):\n\n```console\nconda install -c conda-forge cyipopt\n```\n\n```\npip install git+https://github.com/tulip-control/polytope.git@main\n```\n\nConda packaging coming soon.\n\n## Documentation and Process Operability Principles\n\nFull documentation and discussion regarding process operability principles are available in [opyrability's online portal.](https://codes-group.github.io/opyrability/)\n\n\n\n",
    'author': 'Victor Alves',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'py_modules': modules,
    'install_requires': install_requires,
    'python_requires': '>=3.9.0,<4.0.0',
}


setup(**setup_kwargs)

