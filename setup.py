# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

modules = \
['opyrability', 'PolyhedraVolAprox']
install_requires = \
['cvxopt',
 'jax[cpu]>=0.4.13,<0.5.0',
 'matplotlib',
 'numpy',
 'polytope>=0.2.4,<0.3.0',
 'scipy',
 'tqdm']

setup_kwargs = {
    'name': 'opyrability',
    'version': '1.4.1',
    'description': 'Process operability analysis in Python',
    'author': 'Victor Alves',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'py_modules': modules,
    'install_requires': install_requires,
    'python_requires': '>=3.9',
}


setup(**setup_kwargs)

