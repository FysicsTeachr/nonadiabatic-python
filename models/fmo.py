# models/fmo.py
import numpy as np
from .base import NuclearModelBase
from models import baths

class FMO(NuclearModelBase):
    """
    Implements the specific Hamiltonian, nuclear derivatives, and initialization
    for the FMO site-exciton model.
    """
    def H(self, R_coords: np.ndarray) -> np.ndarray:
        h = self.e_matrix.copy()
        v0_total = 0.0
        for i in range(self.F):
            v0_total += self.V0(R_coords[i, :])
        for i in range(self.F):
            h[i, i] += v0_total + self.V1(R_coords[i, :])
        return h

    def get_nuclear_derivs(self, rho: np.ndarray, R_reshaped: np.ndarray, P_reshaped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates nuclear derivatives for the FMO model in the diabatic representation.
        (Formerly the 'derivs_nuclear_fmo' function).
        """
        dR_dt = P_reshaped / self.m
        force = np.zeros_like(R_reshaped)
        for i in range(self.F):
            v0_dR = self.m * (self.omega_k**2) * R_reshaped[i, :]
            v1_dR = -self.m * (self.omega_k**2) * self.d_k
            force[i, :] = -(np.trace(rho).real * v0_dR + rho[i, i].real * v1_dR)
        return dR_dt, force

    def initialize_nuclear_coordinates(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Initializes the FMO model by sampling from the ground state Wigner distribution.
        The Hamiltonian itself contains the displacement.
        """
        return baths.sample_bath_wigner(self.params["built_bath_params"], rng)
