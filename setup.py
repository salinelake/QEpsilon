from setuptools import setup, find_packages

DESCRIPTION = 'Python package for hybrid quantum-classical simulation of open quantum systems'
setup(
    name='QEpsilon',
    version='0.1.0',
    author="Pinchen Xie",
    author_email="<pinchenxie@lbl.gov>",
    packages=['qepsilon'],
    description=DESCRIPTION,
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.24.0',
        'torch>=2.4.0',
    ],
)
