import os

from m2cgen import ast
from m2cgen.interpreters import mixins, utils
from m2cgen.interpreters.rust.code_generator import RustCodeGenerator
from m2cgen.interpreters.interpreter import ImperativeToCodeInterpreter


class RustInterpreter(ImperativeToCodeInterpreter,
                      mixins.LinearAlgebraMixin):

    supported_bin_vector_ops = {
        ast.BinNumOpType.ADD: "add_vectors",
    }

    supported_bin_vector_num_ops = {
        ast.BinNumOpType.MUL: "mul_vector_number",
    }

    abs_function_name = "abs"
    atan_function_name = "atan"
    exponent_function_name = "exp"
    logarithm_function_name = "ln"
    log1p_function_name = "ln_1p"
    power_function_name = "powf"
    sigmoid_function_name = "sigmoid"
    softmax_function_name = "softmax"
    sqrt_function_name = "sqrt"
    tanh_function_name = "tanh"

    with_sigmoid_expr = False
    with_softmax_expr = False

    def __init__(self, indent=4, function_name="score", *args, **kwargs):
        self.function_name = function_name

        cg = RustCodeGenerator(indent=indent)
        super().__init__(cg, *args, **kwargs)

    def interpret(self, expr):
        self._cg.reset_state()
        self._reset_reused_expr_cache()

        with self._cg.function_definition(
                name=self.function_name,
                args=[self._feature_array_name]):
            last_result = self._do_interpret(expr)
            self._cg.add_return_statement(last_result)

        current_dir = os.path.dirname(__file__)

        if self.with_linear_algebra:
            filename = os.path.join(current_dir, "linear_algebra.rs")
            self._cg.add_code_lines(utils.get_file_content(filename))

        if self.with_softmax_expr:
            filename = os.path.join(current_dir, "softmax.rs")
            self._cg.add_code_lines(utils.get_file_content(filename))

        if self.with_sigmoid_expr:
            filename = os.path.join(current_dir, "sigmoid.rs")
            self._cg.add_code_lines(utils.get_file_content(filename))

        return self._cg.finalize_and_get_generated_code()
