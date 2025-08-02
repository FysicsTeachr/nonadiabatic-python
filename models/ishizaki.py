# models/ishizaki.py
import numpy as np
from .base import NuclearModelBase
from models import baths

class Ishizaki(NuclearModelBase):
    """
    Implements the specific Hamiltonian, nuclear derivatives, and initialization
    for the 2-site Ishizaki model.
    """
    def __init__(self, params: dict, built_bath_params: dict):
        super().__init__(params, built_bath_params)
        if self.F != 2:
            raise ValueError("The Ishizaki model class is implemented only for F=2 states.")

    def H(self, R_coords: np.ndarray) -> np.ndarray:
        h = self.e_matrix.copy()
        v0_total = self.V0(R_coords[0, :]) + self.V0(R_coords[1, :])
        h[0, 0] += v0_total + self.V1(R_coords[0, :])
        h[1, 1] += v0_total + self.V1(R_coords[1, :])
        return h

    def get_nuclear_derivs(self, rho: np.ndarray, R: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates nuclear derivatives for the Ishizaki model in the diabatic representation.
        (Formerly the 'derivs_nuclear_ishizaki' function).
        """
        dR_dt = np.zeros((self.F, self.n_modes))
        dP_dt = np.zeros((self.F, self.n_modes))
        for i in range(self.F):
            dR_dt[i, :] = P[i, :] / self.m
            for k in range(self.n_modes):
                # This requires a dH_dRi_k method, which can live in the base class
                # if it's general enough, or be defined here.
                dH_dRi_k = self.dH_dRi_k(R, i, k)
                force_ik = -np.trace(rho @ dH_dRi_k)
                dP_dt[i, k] = force_ik
        return dR_dt, dP_dt

    def initialize_nuclear_coordinates(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Initializes the Ishizaki model by sampling from the Wigner distribution
        and then explicitly shifting the initial state's coordinates.
        """
#        R_unshifted, P = baths.sample_bath_wigner(self.params["built_bath_params"], rng)
#        R = baths.shift_bath_coordinates(R_unshifted, self.params, self.params["built_bath_params"])
        R, P = baths.sample_bath_wigner(self.params["built_bath_params"], rng)
        return R, P

    # --- Model-Specific Helper Methods ---
#    def dH_dRi_k(self, R_coords: np.ndarray, i_pigment: int, k_mode: int) -> np.ndarray:
#        R_ik = R_coords[i_pigment, k_mode]
#        dV0_dRik = self.m * (self.omega_k[k_mode]**2) * R_ik
#        dV1_dRik = -self.c_k[k_mode]
#        dH_matrix = np.zeros((self.F, self.F))
#        for j in range(self.F):
#            dH_matrix[j, j] += dV0_dRik
#        dH_matrix[i_pigment, i_pigment] += dV1_dRik
#        return dH_matrix
        
    def get_NAC_vectors(self, adiab_E: np.ndarray, Hel_dR_adia_list: list) -> np.ndarray:
        """Overrides base class method with a specialized ishizaki)"""
        energy_diff = adiab_E[1] - adiab_E[0]
        if abs(energy_diff) < 1e-12:
            energy_diff = np.copysign(1e-12, energy_diff)
        nac_vectors = np.array([Hel_dR_k[0, 1] / energy_diff for Hel_dR_k in Hel_dR_adia_list])
        return nac_vectors
