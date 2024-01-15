import ast
import re


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
