import ast
import json
import re
from typing import List

from pydantic import BaseModel


class ScratchpadEntry(BaseModel):
    task: str
    result: str

    def __str__(self):
        return f"Task: {self.task}\nResult: {self.result}"


class Scratchpad(BaseModel):
    entries: List[ScratchpadEntry] = []

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def clear(self):
        self.entries = []

    def copy(self) -> "Scratchpad":
        return Scratchpad(entries=self.entries.copy())

    def add(self, task: str, result: str):
        self.entries.append(ScratchpadEntry(task=task, result=result))

    def entries_repr(self) -> str:
        if self.is_empty():
            return "<no previous steps available>"
        else:
            return "\n\n".join(str(entry) for entry in self.entries)

    def results_repr(self) -> str:
        if self.is_empty():
            return "<no context information available>"
        else:
            return "\n".join([se.result for se in self.entries])


def object_from_schema(schema, return_keys=False):
    keys = []
    obj = _object_from_schema(schema, keys)

    if return_keys:
        return obj, keys
    else:
        return obj


def prop_order_from_schema(schema):
    _, keys = object_from_schema(schema, return_keys=True)
    return keys


def _object_from_schema(schema, keys):
    """Returns a JSON object of given schema, with descriptions as values."""
    if 'properties' in schema:
        example = {}
        for key, value in schema['properties'].items():
            keys.append(key)
            example[key] = _object_from_schema(value, keys=keys)
        return example
    elif 'items' in schema:
        return [_object_from_schema(schema['items'], keys=keys)]
    elif 'description' in schema:
        return schema['description']
    else:
        return None


def extract_json(s: str) -> dict:
    match = re.search(r"```json(.*)```", s, re.DOTALL)
    if not match:
        raise ValueError(f"json could not be extracted (input='{s}')")
    return json.loads(match.group(1))


def extract_code(s: str, remove_print_statements: bool = False) -> str:
    match = re.search(r"```python(.*?)```", s, re.DOTALL)
    if not match:
        raise ValueError(f"code could not be extracted (input='{s}')")
    code = match.group(1)
    if remove_print_statements:
        code = _remove_print_statements(code)
    return code


def _remove_print_statements(code: str) -> str:
    import ast

    class RemovePrints(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
                return None
            return node

    tree = ast.parse(code)
    tree = RemovePrints().visit(tree)
    return f"\n{ast.unparse(tree)}\n"


def exec_code(code: str, result_variable_name: str):
    try:
        loc_variables = {}
        exec(code, globals(), loc_variables)
        return loc_variables[result_variable_name]
    except Exception as e:
        raise ValueError(f"code could not be executed (code='{code}')", e)


def parse_function_call(call):
    tree = ast.parse(call)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            return args, kwargs
