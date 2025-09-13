# utils/read_params.py
from pathlib import Path
import re
import numpy as np

_BOOL_MAP = {"T": True, "F": False, "True": True, "False": False, "true": True, "false": False}

def _coerce_scalar(token):
    """Converts a string token to a boolean, integer, or float if possible."""
    if token in _BOOL_MAP:
        return _BOOL_MAP[token]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token

def convert_units(params):
    """Converts parameters from specified units to atomic units."""
    FS_TO_AU = 1.0 / 0.02418884326509

    if params.get('time_units') == 'femtoseconds':
        params['end_time_au'] = float(params['end_time']) * FS_TO_AU
        params['dt_au'] = float(params['dt']) * FS_TO_AU
    else:
        params['end_time_au'] = float(params['end_time'])
        params['dt_au'] = float(params['dt'])

    return params

def parse_argm(path):
    """Parses a .argm input file into a dictionary."""
    params = {}
    if not path.is_file():
        raise FileNotFoundError(f"Argm file not found: {path}")

    with path.open('r') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            comment_pos = line.find("#")
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()

            parts = re.split(r"\s+", line, maxsplit=1)
            key = parts[0]
            value_str = parts[1] if len(parts) > 1 else ""
            value_tokens = value_str.split()

            if not value_tokens:
                processed_value = True
            elif len(value_tokens) == 1:
                processed_value = _coerce_scalar(value_tokens[0])
            else:
                processed_value = [_coerce_scalar(t) for t in value_tokens]

            params[key] = processed_value

    params['argm_file_path'] = str(path.resolve())

    params = convert_units(params)
    return params
