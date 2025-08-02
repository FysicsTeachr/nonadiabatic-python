# utils/read_params.py
from pathlib import Path
import re
from typing import Any, Dict
import numpy as np

_BOOL_MAP = {"T": True, "F": False, "True": True, "False": False, "true": True, "false": False}

def _coerce_scalar(token: str) -> Any:
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

def read_hamiltonian_matrix(file_path: Path, F: int) -> np.ndarray:
    """
    Reads a symmetric matrix from a file containing the lower triangle.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Hamiltonian file not found: {file_path}")

    e_matrix = np.zeros((F, F))
    with file_path.open('r') as f:
        lines = f.readlines()
        if len(lines) < F:
            raise ValueError(f"Hamiltonian file has too few rows. Expected {F}, got {len(lines)}.")
        for i in range(F):
            row_vals = [float(v) for v in lines[i].strip().split()]
            if len(row_vals) != i + 1:
                raise ValueError(f"Incorrect number of columns in row {i} of Hamiltonian file.")
            for j in range(i + 1):
                e_matrix[i, j] = e_matrix[j, i] = row_vals[j]
    return e_matrix

def convert_units(params: Dict[str, Any]) -> Dict[str, Any]:
    """Converts parameters from specified units to atomic units."""
    WAVENUMBERS_TO_AU = 1.0 / 219474.63
    KB_AU = 3.1668114e-6
    FS_TO_AU = 1.0 / 0.02418884326509

    if 'lambda_cm' in params:
        params['lambda'] = float(params['lambda_cm']) * WAVENUMBERS_TO_AU

    if 'gamma_fs' in params: # Handles characteristic time tau in fs
        tau_fs = float(params['gamma_fs'])
        params['gamma'] = 1.0 / (tau_fs * FS_TO_AU) if tau_fs > 0 else 0.0
    elif 'gamma_cm' in params: # Handles cutoff frequency wc in cm-1
        params['gamma'] = float(params['gamma_cm']) * WAVENUMBERS_TO_AU

    if 'T_kelvin' in params:
        T = float(params['T_kelvin'])
        params['beta'] = 1.0 / (KB_AU * T) if T > 1e-6 else float('inf')

    if params.get('time_units') == 'femtoseconds':
        params['end_time_au'] = float(params['end_time']) * FS_TO_AU
        params['dt_au'] = float(params['dt']) * FS_TO_AU
    else:
        params['end_time_au'] = float(params['end_time'])
        params['dt_au'] = float(params['dt'])

    # Convert Hamiltonian matrix if it exists
    if 'e_matrix' in params:
        params['e_matrix'] *= WAVENUMBERS_TO_AU

    return params

def parse_argm(path: Path) -> Dict[str, Any]:
    """Parses a .argm input file into a dictionary."""
    params: Dict[str, Any] = {}
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

    # Read Hamiltonian matrix if specified in the .argm file
    if 'Hel_file' in params:
        F = int(params['F'])
        # The path to Hel_file is now relative to the main project directory
        hel_file_path = Path(params['Hel_file'])
        params['e_matrix'] = read_hamiltonian_matrix(hel_file_path, F)

    params = convert_units(params)
    return params
