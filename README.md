# Multiscale optimization

Base implementation for multiscale optimization problems through homogenization using [Firedrake](https://www.firedrakeproject.org), where the macroscopic partial differential equation (PDE) comprises a term representing the microscopic scale. The microscopic problem consists in solving another PDE pointwise at the microscopic level.

## Setup

This work relies on the ExternalOperator interface within the Firedrake finite element system (see [paper](https://arxiv.org/abs/2111.00945)). Therefore, you need to install Firedrake and use the ExternalOperator branches.

### Installing Firedrake

Firedrake is installed via its installation script, which you can download by running:

```download_install_script
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
```

Then, you can install Firedrake and specify the required branches using:

```install_firedrake_external_operator_branches
python3 firedrake-install --package-branch ufl external-operator_dualspace --package-branch firedrake pointwise-adjoint-operator_dualspace --package-branch pyadjoint dualspace
```

Finally, you will need to activate the Firedrake virtual environment:

```activate_venv
source firedrake/bin/activate
```

For more details about installing Firedrake: see [here](https://www.firedrakeproject.org/download.html).

### Testing installation

We recommend that you run the test suite after installation to check that your setup is fully functional. Activate the venv as above and then run:

```install_firedrake_external_operator_branches
pytest tests
```

## Usage

This repository is mostly designed for ongoing work on downstream applications that require this type of Firedrake external operator. More specifically, it is used for:

- Multiscale optimization through homogenization
- Control of periodic solutions of nonlinear partial differential equations
