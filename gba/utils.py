def object_from_schema(schema, return_keys=False):
    keys = []
    obj = _object_from_schema(schema, keys)

    if return_keys:
        return obj, keys
    else:
        return obj


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
