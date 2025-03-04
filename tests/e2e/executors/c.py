import os
import subprocess

from m2cgen import assemblers, interpreters
from tests import utils
from tests.e2e.executors.base import BaseExecutor


EXECUTOR_CODE_TPL = """
#include <stdio.h>

{model_code}

int main(int argc, char *argv[])
{{
    double input [argc-1];
    for (int i = 1; i < argc; ++i) {{
        sscanf(argv[i], "%lf", &input[i-1]);
    }}

    {print_code}

    return 0;
}}
"""

EXECUTE_AND_PRINT_SCALAR = """
    printf("%f\\n", score(input));
"""

EXECUTE_AND_PRINT_VECTOR_TPL = """
    double result[{size}];
    score(input, result);

    for (int i = 0; i < {size}; ++i) {{
        printf("%f ", *(result+i));
    }}
"""


class CExecutor(BaseExecutor):

    def __init__(self, model):
        self.model_name = "score"
        self.model = model
        self.interpreter = interpreters.CInterpreter()

        assembler_cls = assemblers.get_assembler_cls(model)
        self.model_ast = assembler_cls(model).assemble()

        self.exec_path = None

    def predict(self, X):
        exec_args = [self.exec_path, *map(utils.format_arg, X)]
        return utils.predict_from_commandline(exec_args)

    def prepare(self):
        if self.model_ast.output_size > 1:
            print_code = EXECUTE_AND_PRINT_VECTOR_TPL.format(
                size=self.model_ast.output_size)
        else:
            print_code = EXECUTE_AND_PRINT_SCALAR

        executor_code = EXECUTOR_CODE_TPL.format(
            model_code=self.interpreter.interpret(self.model_ast),
            print_code=print_code)

        file_name = os.path.join(self._resource_tmp_dir, f"{self.model_name}.c")
        with open(file_name, "w") as f:
            f.write(executor_code)

        self.exec_path = os.path.join(self._resource_tmp_dir, self.model_name)
        flags = ["-std=c99", "-lm"]
        subprocess.call([
            "gcc",
            file_name,
            "-o",
            self.exec_path,
            *flags
        ])
