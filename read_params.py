# read_params.py
from pathlib import Path
import re
from typing import Any, Dict, List

# Map string representations to boolean values
_BOOL_MAP = {"T": True, "F": False, "True": True,
             "False": False, "true": True, "false": False}

def _coerce_scalar(token: str) -> Any:
    # Check if token maps to a boolean
    if token in _BOOL_MAP:
        return _BOOL_MAP[token]

    # Attempt to convert to integer
    try:
        return int(token)
    except ValueError:
        pass

    # Attempt to convert to float
    try:
        return float(token)
    except ValueError:
        pass

    # If no conversion is successful, return the original string
    return token

def parse_argm(path: Path) -> Dict[str, Any]:

    params: Dict[str, Any] = {}
    # Calling is_file() method from pathlib.Path
    if not path.is_file():
        raise FileNotFoundError(f"Argm file not found: {path}")

    # Calling open() method from pathlib.Path
    with path.open('r') as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip empty lines or comment lines
            if not line or line.startswith("#"):
                continue

            # Remove in-line comments
            comment_pos = line.find("#")
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()

            # Calling split() from re module
            # Split into key and value string
            parts = re.split(r"\s+", line, maxsplit=1)
            # Corrected parsing: key is the first part, value_str is the second
            key = parts[0]
            value_str = parts[1] if len(parts) > 1 else ""
            value_tokens = value_str.split()

            processed_value: Any
            if not value_tokens:
                # If no value tokens, assume boolean True
                processed_value = True
            elif len(value_tokens) == 1:
                # Calling local function _coerce_scalar
                # CORRECTED: Pass the token (string), not the list
                processed_value = _coerce_scalar(value_tokens[0])
            else:
                # Handle multiple tokens, attempt list of numbers
                try:
                    num_list = [float(t) for t in value_tokens]
                    if all(f.is_integer() for f in num_list):
                        processed_value = [int(f) for f in num_list]
                    else:
                        processed_value = num_list
                except ValueError:
                    # Calling local function _coerce_scalar
                    # Fallback to list of coerced strings
                    # CORRECTED: Iterate over value_tokens, not value_tokens[0]
                    processed_value = [_coerce_scalar(t)
                                       for t in value_tokens]

            # Handle duplicate keys: if key exists, convert to list
            # and append, otherwise assign directly
            if key in params:
                if not isinstance(params[key], list):
                    params[key] = [params[key]]
                params[key].append(processed_value)
            else:
                params[key] = processed_value

    # Calling resolve() method from pathlib.Path
    # Store the absolute path of the argm file
    params['argm_file_path'] = str(path.resolve())

    return params
