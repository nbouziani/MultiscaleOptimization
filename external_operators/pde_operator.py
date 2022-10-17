from firedrake import *
import firedrake_adjoint as fda


class PDEOperator(AbstractExternalOperator):
    r"""External operator that represents the solution of a given PDE.

        The PDEOperator class is equipped with 3 assembly methods that correspond to the assembly of: the operator,
        the action of its Jacobian, and the action of the adjoint of the Jacobian of N. A PDEOperator object can
        be added to any PDE or PDE-constrained optimization problem and the assembly of the terms derived from N
        will be delegated to the assembly methods of the PDEOperator class.

        Let `N` be a PDEOperator:

        - The evaluation of N will produce the solution of the underlying PDE of the PDEOperator class.
          -> Remark: The underlying PDE can be changed by overwriting the `_solve_pde` method.

        - The action of the Jacobian of N is evaluated by computing the tangent linear model of the PDE.

        - Likewise, the action of the adjoint of the Jacobian of N is evaluated by computing the adjoint model of the PDE.

        Example: We want to find u such that:

            - \Delta u + u = f  in  \Omega
                         u = g  on  \partial Omega

        where g is the boundary condition and where f is the solution of another PDE.
        This can be implemented by defining the above PDE where f is a PDEOperator object
        representing the solution of the PDE satisfied by f.
    """

    def __init__(self, *operands, function_space, derivatives=None, **kwargs):
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, **kwargs)

    def _solve_pde(self, *pde_inputs, **pde_params):
        """Solve a Poisson equation

            pde_inputs: rhs `f` of Poisson equation
            pde_params: parameters (n, solver_parameters, etc.)
        """
        # Get rhs
        f, = pde_inputs
        # Set the variational problem
        mesh = pde_params.get('mesh')
        degree = pde_params.get('degree', 2)
        V = FunctionSpace(mesh, "CG", degree)
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

    # Evaluate operator, i.e. solve the PDE.
    @assemble_method(0, (0,))
    def eval_operator(self, *args, **kwargs):
        # Get ufl operands
        pde_params = self.operator_data['pde_params']
        # Solve PDE
        u = self._solve_pde(*self.ufl_operands, **pde_params)
        return u

    @assemble_method((1,), (0, None))
    def eval_dN_df_action(self, *args, **kwargs):
        # Get ufl operands
        f, = self.ufl_operands

        # Prepare TLM computation
        pde_params = self.operator_data['pde_params']
        self.prepare_tlm_computation(f, pde_params)

        # Assign tlm value
        self.operator_data['control'].block_variable.tlm_value = self.argument_slots()[-1]

        # Evaluate TLM
        self.operator_data['tape'].evaluate_tlm()

        # Get TLM value
        dJdm = self.operator_data['solution'].block_variable.tlm_value
        assert dJdm is not None

        return dJdm

    @assemble_method((1,), (None, 0))
    def eval_dN_df_adjoint_action(self, *args, **kwargs):
        # Get ufl operands
        f, = self.ufl_operands

        # Prepare adjoint computation
        pde_params = self.operator_data['pde_params']
        self.prepare_adjoint_computation(f, pde_params)

        # Assign adjoint value
        self.operator_data['solution'].block_variable.adj_value = self.argument_slots()[0].vector()

        # Evaluate adjoint
        self.operator_data['tape'].evaluate_adj()

        # Get adjoint value
        dJdm_adj = self.operator_data['control'].block_variable.adj_value
        assert dJdm_adj is not None

        return dJdm_adj.function

    # Evaluate the forward model to populate the tape
    def compute_tape(self, f, pde_params):
        # Get working tape
        working_tape = fda.get_working_tape()

        # Create tape
        forward_tape = fda.Tape()
        fda.set_working_tape(forward_tape)
        forward_tape.clear_tape()

        # Annotate current tape
        is_annotated = fda.annotate_tape()
        i_cont = 0
        while not is_annotated:
            fda.continue_annotation()
            is_annotated = fda.annotate_tape()
            i_cont += 1

        # Create control variable
        c = fda.Control(f)
        self.operator_data['control'] = c

        # Evaluate forward model
        solution = self._solve_pde(f, **pde_params)

        # Record result
        self.operator_data['solution'] = solution

        # Record tape
        self.operator_data['tape'] = forward_tape

        # Reset tape counter to original value
        for _ in range(i_cont):
            fda.pause_annotation()

        fda.set_working_tape(working_tape)

    def prepare_tlm_computation(self, f, pde_params, recompute_tape=False):
        # Check if we need to recompute the tape
        if recompute_tape or ('tape' not in self.operator_data.keys()):
            self.compute_tape(f, pde_params)

        # Reset TLM values
        self.operator_data['tape'].reset_tlm_values()

        # Check if we need to recompute the blocks
        if norm(f - self.operator_data['control'].data()) < 1e-15:
            recompute_block = False
        else:
            recompute_block = True

        # Assign control value
        self.operator_data['control'].update(f)

        # Recompute blocks
        if recompute_block:
            import time
            current_time = time.time()
            for b in self.operator_data['tape'].get_blocks():
                b.recompute()
            print("Recomputing tape blocks in %.2e s" % (time.time() - current_time), flush=True)

    def prepare_adjoint_computation(self, f, pde_params, recompute_tape=False):
        # Check if we need to recompute the tape
        if recompute_tape or ('tape' not in self.operator_data.keys()):
            self.compute_tape(f, pde_params)

        # Reset adjoint values
        self.operator_data['tape'].reset_variables()

        # Check if we need to recompute the blocks
        if norm(f - self.operator_data['control'].data()) < 1e-15:
            recompute_block = False
        else:
            recompute_block = True

        # Assign control value
        self.operator_data['control'].update(f)

        # Recompute blocks
        if recompute_block:
            import time
            current_time = time.time()
            for b in self.operator_data['tape'].get_blocks():
                b.recompute()
            print("Recomputing tape blocks in %.2e s" % (time.time() - current_time), flush=True)


# Helper function #
def pde_operator(function_space, operator_data):
    return partial(PDEOperator, operator_data=operator_data, function_space=function_space)
