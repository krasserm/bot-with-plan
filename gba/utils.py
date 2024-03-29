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


def extract_code(s: str) -> str:
    match = re.search(r"```(.*)```", s, re.DOTALL)
    if not match:
        raise ValueError(f"code could not be extracted (input='{s}')")
    return match.group(1)


def exec_code(code: str, result_variable_name: str):
    try:
        loc_variables = {}
        exec(code, globals(), loc_variables)
        result = loc_variables[result_variable_name]

        if isinstance(result, float):
            result = round(result, 5)
        return result
    except Exception as e:
        raise ValueError(f"code could not be executed (code='{code}')", e)


def parse_function_call(call):
    tree = ast.parse(call)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            return args, kwargs
