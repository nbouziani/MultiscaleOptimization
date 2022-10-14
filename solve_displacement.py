from firedrake import *
from firedrake_adjoint import *
import numpy as np
from external_operators.PDE_operator import pde_operator
from tests.test_pde_operators_assembly import compute_adjoint, compute_tlm


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
        operator_data = {'pde_params': {'mesh': mesh, 'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}}
        Vf = FunctionSpace(mesh, "CG", 2)
        p = pde_operator(function_space=Vf, operator_data=operator_data)
        # Set rhs to 0
        f_poisson = Function(Vf)
        # N is solution of:
        #  - \Delta u = f_poisson in Omega
        #           u = 1   on \partial \Omega
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


u = solve_elasticity(100, 100, options=options_elasticity)
u_external_operator = solve_elasticity(100, 100, external_operator=True, options=options_elasticity)
assert np.allclose(u.dat.data_ro, u_external_operator.dat.data_ro)


# --  Check assembly of the Jacobian action of the operator, i.e. dNdu(f; delta_f, v*) -- #

# Set problem
mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, "CG", 2)
# Set rhs
x, y = SpatialCoordinate(mesh)
f_poisson = Function(V).interpolate(cos(2 * pi * x) * sin(2 * pi * y))

# N is solution of:
#  - \Delta u = f_poisson in Omega
#           u = 1   on \partial \Omega
operator_data = {'pde_params': {'mesh': mesh, 'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}}
p = pde_operator(function_space=V, operator_data=operator_data)
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
b = compute_tlm(forward_pde, f_poisson, tlm_value=delta_f, **operator_data['pde_params'])
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
forward_pde = N._solve_pde
b = compute_adjoint(forward_pde, f_poisson, adj_value=delta_N, **operator_data['pde_params'])
assert np.allclose(a.dat.data_ro, b.dat.data_ro)
