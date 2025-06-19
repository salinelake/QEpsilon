from setuptools import setup, find_packages

DESCRIPTION = 'Python package for hybrid quantum-classical simulation of open quantum systems'

# Documentation dependencies
docs_require = [
    'sphinx>=7.0.0',
    'sphinx-rtd-theme>=2.0.0',
    'myst-parser>=2.0.0',
    'nbsphinx>=0.9.0',
    'matplotlib>=3.7.0',
    'jupyterlab>=4.0.0',
]

setup(
    name='QEpsilon',
    version='0.1.0',
    author="Pinchen Xie",
    author_email="<pinchenxie@lbl.gov>",
    packages=['qepsilon'],
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.24.0',
        'torch>=2.4.0',
    ],
    extras_require={
        'docs': docs_require,
        'dev': docs_require + ['pytest', 'black', 'flake8'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
