# nuclear_diabatic_H-dH.py
from typing import Dict, Any, List
import numpy as np
import math
from scipy.linalg import eigh # Required for this fix

class SpinBoson:
    def __init__(self, params, built_bath_params):
        self.delta: float = params["delta"]
        self.epsilon: float = params["epsilon"]
        self.F = 2 # The number of electronic states is 2 for Spin-Boson

        self.omega_k: np.ndarray = built_bath_params["omega_k"]
        self.d_k: np.ndarray = built_bath_params["d_k"]
        self.shift: np.ndarray = built_bath_params["shift"]
        self.m: float = built_bath_params["bath_mass"]
        self.bath_N: int = built_bath_params["n_modes"]

    def H(self, R_coords: np.ndarray) -> np.ndarray:
        H00_potential = 0.5 * np.sum(self.m * (self.omega_k**2) * \
                       ((R_coords - self.d_k * self.shift[0])**2))
        H00 = H00_potential + self.epsilon

        H11_potential = 0.5 * np.sum(self.m * (self.omega_k**2) * \
                       ((R_coords - self.d_k * self.shift[1])**2))
        H11 = H11_potential - self.epsilon

        return np.array([[H00, self.delta], [self.delta, H11]])

    def dH_dRk(self, R_coords: np.ndarray, k_mode: int) -> np.ndarray:
        dH00_dRk = self.m * (self.omega_k[k_mode]**2) * \
         (R_coords[k_mode] - self.d_k[k_mode] * self.shift[0])
        dH11_dRk = self.m * (self.omega_k[k_mode]**2) * \
         (R_coords[k_mode] - self.d_k[k_mode] * self.shift[1])
        return np.array([[dH00_dRk, 0.0], [0.0, dH11_dRk]])

    # --- ADDED BACK: Missing methods for numerical stability ---
    def _mat_correct_phase(self, old_U: np.ndarray, new_U: np.ndarray) -> np.ndarray:
        """ Ensures the phase of the eigenvectors is consistent between steps. """
        corr_U = new_U.copy()
        for j in range(new_U.shape[1]):
            if np.real(np.dot(old_U[:, j].conj(), new_U[:, j])) < 0:
                corr_U[:, j] *= -1
        return corr_U

    def get_adiabatic_properties(self, R_coords: np.ndarray, U_prev: np.ndarray):
        """ Calculates adiabatic energies and the phase-corrected transformation matrix. """
        H_diab = self.H(R_coords)
        adiab_E_k, U_k_raw = eigh(H_diab)
        U_k = self._mat_correct_phase(U_prev, U_k_raw)
        return adiab_E_k, U_k

