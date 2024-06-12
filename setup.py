from setuptools import setup, find_packages

setup(
    name='DIP_solver',                 # Name of the package
    version='2.0.0',                  # Version
    author='Abdelhamed Eid',             # Author name
    description='A library to solve DIP and some CV problems, implemented from scratch in pure python for easy use, made originally to learn the implementation of the algorithms',# Short description
    packages=find_packages(),       # Automatically finds packages with __init__.py
    # install_requires=[              # Dependencies
    #     'matplotlib'
    # ],
)
