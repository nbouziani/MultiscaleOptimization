import numpy as np
from firedrake import *
from firedrake_adjoint import *

from external_operators import PDEOperator


def forward_pde(*pde_inputs, **pde_params):
    """Solve a Poisson equation

        pde_inputs: rhs `f` of Poisson equation
        pde_params: parameters (n, solver_parameters, etc.)
    """
    # Get rhs
    f, = pde_inputs
    # Set the variational problem
    mesh = pde_params.get('mesh')
    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
    w = Function(V)
    # Solve PDE
    solver_parameters = pde_params.get('solver_parameters', {})
    solve(a == L, w, bcs=bcs, solver_parameters=solver_parameters)

    return w


class TestPDEOperator(PDEOperator):
    def __init__(self, *operands, **kwargs):
        PDEOperator.__init__(self, *operands, **kwargs)

    def _solve_pde(self, *pde_inputs, **pde_params):
        return forward_pde(*pde_inputs, **pde_params)


def test_assembly_operator():
    # Setup
    mesh = UnitSquareMesh(20, 20)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(cos(2 * pi * x) * sin(2 * pi * y))

    # Define the external operator
    pde_params = {'mesh': mesh, 'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}
    operator_data = {'pde_params': pde_params}
    N = TestPDEOperator(f, function_space=V, operator_data=operator_data)

    # Assemble external operator
    assembled_N = assemble(N)

    # Get PDE solution
    u = forward_pde(f, **pde_params)

    assert np.allclose(assembled_N.dat.data_ro, u.dat.data_ro)


def test_assembly_Jacobian_action():
    # Setup
    mesh = UnitSquareMesh(20, 20)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(cos(2 * pi * x) * sin(2 * pi * y))
    delta_f = Function(V).assign(1)

    # Define the external operator
    pde_params = {'mesh': mesh, 'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}
    operator_data = {'pde_params': pde_params}
    N = TestPDEOperator(f, function_space=V, operator_data=operator_data)

    # Assemble external operator
    dN = derivative(N, f)
    dN_action = action(dN, delta_f)
    assembled_dN_action = assemble(dN_action)

    # Get PDE solution
    N_tlm = compute_tlm(forward_pde, f, tlm_value=delta_f, **pde_params)
    assert np.allclose(assembled_dN_action.dat.data_ro, N_tlm.dat.data_ro)


def test_assembly_Jacobian_adjoint_action():
    # Setup
    mesh = UnitSquareMesh(20, 20)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(cos(2 * pi * x) * sin(2 * pi * y))
    delta_N = Cofunction(V.dual())  # .assign(1)
    delta_N.vector()[:] = 1

    # Define the external operator
    pde_params = {'mesh': mesh, 'solver_parameters': {"ksp_type": "preonly", "pc_type": "lu"}}
    operator_data = {'pde_params': pde_params}
    N = TestPDEOperator(f, function_space=V, operator_data=operator_data)

    # Assemble external operator
    dN = derivative(N, f)
    from ufl.algorithms.ad import expand_derivatives
    dN = expand_derivatives(dN)
    dN_adj = adjoint(dN)
    dN_adj_action = action(dN_adj, delta_N)
    assembled_dN_adj_action = assemble(dN_adj_action)

    # Get PDE solution
    N_adj = compute_adjoint(forward_pde, f, adj_value=delta_N, **pde_params)
    assert np.allclose(assembled_dN_adj_action.dat.data_ro, N_adj.dat.data_ro)


def compute_tlm(forward_operator, f, tlm_value, **operator_kwargs):
    # Get and clear tape
    tape = get_working_tape()
    tape.clear_tape()

    # Set control
    c = Control(f)

    # Compute forward problem
    res = forward_operator(f, **operator_kwargs)

    # Reset TLM values
    tape.reset_tlm_values()

    # Set seed for TLM
    c.block_variable.tlm_value = tlm_value

    # Evaluate TLM
    tape.evaluate_tlm()

    # Get TLM value
    dJdm = res.block_variable.tlm_value
    assert dJdm is not None

    return dJdm


def compute_adjoint(forward_operator, f, adj_value, **operator_kwargs):
    # Get and clear tape
    tape = get_working_tape()
    tape.clear_tape()

    # Set control
    c = Control(f)

    # Compute forward problem
    res = forward_operator(f, **operator_kwargs)

    # Reset adjoint values
    tape.reset_variables()

    # Set seed for adjoint
    res.block_variable.adj_value = adj_value

    # Evaluate adjoint
    tape.evaluate_adj()

    # Get adjoint value
    dJdm_adj = c.block_variable.adj_value
    assert dJdm_adj is not None

    return dJdm_adj.function
