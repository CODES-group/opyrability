# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

modules = \
['pyprop', 'PolyhedraVolAprox']
install_requires = \
['cvxopt>=1.3.0,<2.0.0',
 'cyipopt @ git+https://github.com/mechmotum/cyipopt.git@master',
 'matplotlib>=3.6.2,<4.0.0',
 'numpy>=1.24.1,<2.0.0',
 'polytope @ git+https://github.com/tulip-control/polytope.git@main',
 'scipy>=1.10.0,<2.0.0',
 'tqdm>=4.64.1,<5.0.0']

setup_kwargs = {
    'name': 'pyprop',
    'version': '0.0.1',
    'description': 'Process Operability Calculations in Python',
    'long_description': '# DUMMY README FILE',
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

