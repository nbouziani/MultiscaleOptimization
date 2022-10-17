# Run commands (examples):
# -> python solve_displacement.py -operator pde_operator -nx 20 -ny 20 -degree 2
# -> python solve_displacement.py -operator multiscale_operator -nx 5 -ny 5 -degree 1
import numpy as np
import argparse

from firedrake import *
from firedrake_adjoint import *

from external_operators.pde_operator import pde_operator
from external_operators.multiscale_operator import multiscale_operator
from tests.test_pde_operators_assembly import compute_adjoint, compute_tlm


# Retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nx', type=int, default=5)
parser.add_argument('-ny', type=int, default=5)
parser.add_argument('-degree', type=int, default=1)
parser.add_argument('-operator', type=str, default='pde_operator',
                    help='one of [`pde_operator`, `multiscale_operator`]')
args = parser.parse_args()
# Set hyperparameters
nx, ny = args.nx, args.ny
degree = args.degree
operator = args.operator
if operator not in ('pde_operator', 'multiscale_operator'):
    raise ValueError('Invalid operator: %s' % operator)


def solve_elasticity(nx, ny, external_operator=False, options=None, **kwargs):
    length = 1
    width = 0.2
    mesh = RectangleMesh(nx, ny, length, width)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    rho = Constant(0.01)
    g = Constant(1)
    f = as_vector([0, -rho * g])
    mu = Constant(1)
    lambda_ = Constant(0.25)
    Id = Identity(mesh.geometric_dimension())  # 2x2 Identity tensor

    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lambda_ * div(u) * Id + 2 * mu * epsilon(u)

    bc = DirichletBC(V, Constant([0, 0]), 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    if external_operator:
        operator_data = {'pde_params': {'mesh': mesh, 'degree': degree,
                                        'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}}
        Vf = FunctionSpace(mesh, "CG", degree)
        external_op = pde_operator if operator == 'pde_operator' else multiscale_operator
        p = external_op(function_space=Vf, operator_data=operator_data)
        # Set rhs to 0 (trivial example)
        f_poisson = Function(Vf)
        # N is the solution of the following PDE at each point:
        #  - \Delta u = f_poisson * g(xi, yi) in \Omega
        #           u = 1   on \partial \Omega
        #   with xi, yi \in \Omega.
        #
        #   => For f_poisson = 0, we have u = 1
        #   => N = 1
        N = p(f_poisson)
        a = inner(sigma(u), N * epsilon(v)) * dx
    else:
        a = inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx

    uh = Function(V)
    solve(a == L, uh, bcs=bc, solver_parameters=options, **kwargs)
    return uh


# --  Check assembly of the operator, i.e. N(u; v*) - Elasticity problem -- #

options_elasticity = {"ksp_type": "cg",
                      "ksp_max_it": 100,
                      "pc_type": "gamg",
                      "mg_levels_pc_type": "sor",
                      "mat_type": "aij",
                      "ksp_converged_reason": None}


u = solve_elasticity(nx, ny, options=options_elasticity)
u_external_operator = solve_elasticity(nx, ny, external_operator=True, options=options_elasticity)
assert np.allclose(u.dat.data_ro, u_external_operator.dat.data_ro)


# --  Check assembly of the Jacobian action of the operator, i.e. dNdu(f; delta_f, v*) -- #

# Set problem
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "CG", degree)
# Set rhs
x, y = SpatialCoordinate(mesh)
f_poisson = Function(V).interpolate(cos(2 * pi * x) * sin(2 * pi * y))
# Set coordinates
coordspace = VectorFunctionSpace(mesh, "CG", degree)
coords = interpolate(as_vector([x, y]), coordspace)

# N is solution of:
#  - \Delta u = f_poisson in Omega
#           u = 1   on \partial \Omega
operator_data = {'pde_params': {'mesh': mesh, 'degree': degree,
                                'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}}
external_op = pde_operator if operator == 'pde_operator' else multiscale_operator
p = external_op(function_space=V, operator_data=operator_data)
N = p(f_poisson)


def J_N_action(N, f_poisson, delta_f):
    # Assemble the Jacobian action of N
    dNdf = derivative(N, f_poisson)
    dNdf_action = action(dNdf, delta_f)
    return assemble(dNdf_action)


# Check Jacobian action of N
delta_f = Function(V).assign(1)
a = J_N_action(N, f_poisson, delta_f)
forward_pde = N._solve_pde
# Compute TLM
if operator == 'pde_operator':
    b = compute_tlm(forward_pde, f_poisson, tlm_value=delta_f, **operator_data['pde_params'])
else:
    b = Function(V)
    for i, ci in enumerate(coords.dat.data_ro):
        operator_data['pde_params']['coords'] = ci
        b.dat.data[i] = compute_tlm(forward_pde, f_poisson, tlm_value=delta_f, **operator_data['pde_params'])
# Check
assert np.allclose(a.dat.data_ro, b.dat.data_ro)


# --  Check assembly of the action of the Jacobian adjoint of the operator, i.e. dNdu(f; uhat, delta_N) -- #

def J_N_adjoint_action(N, f_poisson, delta_N):
    # Assemble the action of the Jacobian adjoint of N
    dNdf = derivative(N, f_poisson)
    from ufl.algorithms.ad import expand_derivatives
    dNdf = expand_derivatives(dNdf)
    dNdf_adj = adjoint(dNdf)
    dNdf_adj_action = action(dNdf_adj, delta_N)
    return assemble(dNdf_adj_action)


# Check action of Jacboian adjoint of N
delta_N = Cofunction(V.dual())
delta_N.vector()[:] = 1
a = J_N_adjoint_action(N, f_poisson, delta_N)
# Compute adjoint
if operator == 'pde_operator':
    b = compute_adjoint(forward_pde, f_poisson, adj_value=delta_N.vector(), **operator_data['pde_params'])
else:
    b = Function(V)
    for i, ci in enumerate(coords.dat.data_ro):
        operator_data['pde_params']['coords'] = ci
        bi = compute_adjoint(forward_pde, f_poisson, adj_value=delta_N.vector()[i], **operator_data['pde_params'])
        b.dat.data[i] = bi.dat.data_ro[i]
assert np.allclose(a.dat.data_ro, b.dat.data_ro)
