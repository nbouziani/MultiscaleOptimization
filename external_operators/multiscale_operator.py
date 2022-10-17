from firedrake import *
from firedrake_adjoint import *

from external_operators.pde_operator import PDEOperator


class MultiscaleOperator(PDEOperator):

    def __init__(self, *operands, function_space, derivatives=None, **kwargs):
        PDEOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, **kwargs)

    @assemble_method(0, (0,))
    def eval_operator(self, *args, **kwargs):
        """Evaluate the operator which equates to computing the solutions of the micro-scale PDE pointwise"""

        print('\n\n Evaluate operator \n\n')

        # Get ufl operands
        pde_params = self.operator_data['pde_params']
        # Solve PDE
        u = self.solve_micro_scale_pde(*self.ufl_operands, **pde_params)
        return u

    @assemble_method((1,), (0, None))
    def eval_dN_df_action(self, *args, **kwargs):
        """Evaluate the action of the Jacobian of N on delta_f"""

        print('\n\n Evaluate Jacobian action \n\n')

        # `dJdm` will contain the TLM value pointwise
        dJdm = Function(self.function_space())
        # Set coordinates
        mesh = self.operator_data['pde_params'].get('mesh')
        degree = self.operator_data['pde_params'].get('degree', 2)
        coordspace = VectorFunctionSpace(mesh, "CG", degree)
        coords = interpolate(SpatialCoordinate(mesh), coordspace)

        # Get ufl operands
        f, = self.ufl_operands
        # Get action coefficient
        delta_f = self.argument_slots()[-1]

        # Get PDE params
        pde_params = self.operator_data['pde_params']

        for i, ci in enumerate(coords.dat.data_ro):
            # Set coordinates for solving the local PDE at the point `ci`
            pde_params['coords'] = ci

            # Prepare TLM computation
            self.prepare_tlm_computation(f, pde_params, recompute_tape=True)

            # Assign tlm value
            self.operator_data['control'].block_variable.tlm_value = delta_f

            # Evaluate TLM
            self.operator_data['tape'].evaluate_tlm()

            # Get TLM value
            dJdmi = self.operator_data['solution'].block_variable.tlm_value
            assert dJdmi is not None

            dJdm.dat.data[i] = dJdmi

        return dJdm

    @assemble_method((1,), (None, 0))
    def eval_dN_df_adjoint_action(self, *args, **kwargs):
        """Evaluate the action of the Jacobian adjoint of N on delta_N"""

        print('\n\n Evaluate action of Jacobian adjoint \n\n')

        # Set coordinates
        mesh = self.operator_data['pde_params'].get('mesh')
        degree = self.operator_data['pde_params'].get('degree', 2)
        coordspace = VectorFunctionSpace(mesh, "CG", degree)
        coords = interpolate(SpatialCoordinate(mesh), coordspace)

        # Get ufl operands
        f, = self.ufl_operands
        # `dJdm_adj` will contain the adjoint value pointwise
        dJdm_adj = Function(f.function_space())
        # Get action coefficient
        delta_N = self.argument_slots()[0]

        # Get PDE params
        pde_params = self.operator_data['pde_params']

        for i, ci in enumerate(coords.dat.data_ro):
            # Set coordinates for solving the local PDE at the point `ci`
            pde_params['coords'] = ci

            # Prepare adjoint computation
            self.prepare_adjoint_computation(f, pde_params, recompute_tape=True)

            # Assign adjoint value
            self.operator_data['solution'].block_variable.adj_value = delta_N.vector()[i]

            # Evaluate adjoint
            self.operator_data['tape'].evaluate_adj()

            # Get adjoint value
            dJdmi_adj = self.operator_data['control'].block_variable.adj_value
            assert dJdmi_adj is not None

            dJdm_adj.dat.data[i] = dJdmi_adj.function.dat.data_ro[i]

        return dJdm_adj

    def solve_micro_scale_pde(self, *pde_inputs, **pde_params):
        u = Function(self.function_space())
        mesh = self.operator_data['pde_params'].get('mesh')
        degree = self.operator_data['pde_params'].get('degree', 2)
        coordspace = VectorFunctionSpace(mesh, "CG", degree)
        coords = interpolate(SpatialCoordinate(mesh), coordspace)
        # Populate u at each point with the norm of the solution
        # of the micro scale PDE at that point.
        for i, ci in enumerate(coords.dat.data_ro):
            # Solve the micro scale PDE given the coordinates at that point
            pde_params['coords'] = ci
            u.dat.data[i] = self._solve_pde(*pde_inputs, **pde_params)
        return u

    def _solve_pde(self, *pde_inputs, **pde_params):
        """Solve a Poisson equation and return the norm of the solution

            pde_inputs: rhs `f` of Poisson equation
            pde_params: parameters (n, solver_parameters, etc.)
        """
        # Get rhs
        f, = pde_inputs
        # Get mesh
        mesh = pde_params.get('mesh')
        # Construct g: g(xi, yi) = (xi - yi)**2
        coords = pde_params.get('coords')
        xi, yi = coords
        g = Function(self.function_space()).assign((xi - yi)**2)
        # Set the variational problem
        degree = pde_params.get('degree', 2)
        V = FunctionSpace(mesh, "CG", degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(grad(u), grad(v)) * dx
        L = f * g * v * dx
        bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
        w = Function(V)
        # Solve PDE
        solver_parameters = pde_params.get('solver_parameters', {})
        solve(a == L, w, bcs=bcs, solver_parameters=solver_parameters)
        # Compute norm of the solution
        alpha = Function(V).assign(1)  # Normalization
        norm_w = assemble(w**2 * dx) / assemble(alpha * dx)
        return norm_w


# Helper function #
def multiscale_operator(function_space, operator_data):
    return partial(MultiscaleOperator, operator_data=operator_data, function_space=function_space)
