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
    'version': '1.3',
    'description': 'Process operability analysis in Python',
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

