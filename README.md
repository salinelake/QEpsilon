# QEpsilon

## About QEpsilon
QEpsilon is a Python package for modeling open quantum system. QEpsilon is designed to minimize the effort required to build data-driven quantum master equation of an open quantum system and to perform time evolution of the master equation. Applications of QEpsilon span from artificial quantum systems (qubits) to molecular systems. 

## Highlighted features
- **General and flexible**, supporting the parameterization and simulation of spin, fermionic, bosonic  systems and their combinations. 
- **GPU and sparse linear algebra supports**, making it efficient to simulate relatively large quantum systems (~20 spins or several bosonic modes). 
- **highly modularized**, easy to implement many-body operators. 

## Installation
Clone the repository and 
pip install -e .

## QEpsilon in a nutshell
The quantum master equation modelled by QEpsilon writes as
$$
\frac{d}{dt} \rho(t) = -i[H(t), \rho(t)] + \sum_{k} \left( L_k \rho(t) L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho(t)\}\right)
$$

## Quick start

## Examples

