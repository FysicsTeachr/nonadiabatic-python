# models/base.py
import numpy as np
from scipy.linalg import eigh

class NuclearModelBase:
    """
    An abstract base class that defines the 'contract' for all nuclear models.
    """
    def __init__(self, params: dict, built_bath_params: dict):
        self.params = params
        self.e_matrix = params["e_matrix"]
        self.F = int(params.get("F", 2))
        self.n_modes = int(params.get("n_modes", 100))
        self.omega_k = built_bath_params["omega_k"]
        self.c_k = built_bath_params["c_k"]
        self.d_k = built_bath_params["d_k"]
        self.m = built_bath_params["bath_mass"]

    def H(self, R_coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Each model must implement its own Hamiltonian.")

    def get_nuclear_derivs(self, rho: np.ndarray, R: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Each model must implement its own nuclear derivatives.")

    def initialize_nuclear_coordinates(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Each model must define its own initialization.")

    # --- Common Helper Methods ---
    def V0(self, R_coords_pigment: np.ndarray) -> float:
        return 0.5 * np.sum(self.m * (self.omega_k**2) * (R_coords_pigment**2))

    def V1(self, R_coords_pigment: np.ndarray) -> float:
        interaction = -np.sum(self.c_k * R_coords_pigment)
        reorganization = 0.5 * np.sum(self.c_k * self.d_k)
        return interaction + reorganization

    # --- ADDED: This method is now common to all models ---
    def dH_dRi_k(self, R_coords: np.ndarray, i_pigment: int, k_mode: int) -> np.ndarray:
        """Calculates the derivative of the Hamiltonian w.r.t. a single nuclear coordinate."""
        R_ik = R_coords[i_pigment, k_mode]
        dV0_dRik = self.m * (self.omega_k[k_mode]**2) * R_ik
        dV1_dRik = -self.c_k[k_mode]
        dH_matrix = np.zeros((self.F, self.F))
        for j in range(self.F):
            dH_matrix[j, j] += dV0_dRik
        dH_matrix[i_pigment, i_pigment] += dV1_dRik
        return dH_matrix

    def get_adiabatic_properties(self, R_coords: np.ndarray, U_prev: np.ndarray):
        H_diab = self.H(R_coords)
        adiab_E, U = eigh(H_diab)
        U = self._mat_correct_phase(U_prev, U)
        return adiab_E, U

    def get_Hel_dR_adia(self, U: np.ndarray, dH_dR_diab_list: list) -> list:
        U_T = U.T.conj()
        return [U_T @ dH_dR_k @ U for dH_dR_k in dH_dR_diab_list]

    def get_NAC_vectors(self, adiab_E: np.ndarray, Hel_dR_adia_list: list) -> np.ndarray:
        n_nucl_coords = len(Hel_dR_adia_list)
        nac_tensor = np.zeros((self.F, self.F, n_nucl_coords))
        for i in range(self.F):
            for j in range(i + 1, self.F):
                energy_diff = adiab_E[j] - adiab_E[i]
                if abs(energy_diff) < 1e-12:
                    nac_vec = np.zeros(n_nucl_coords)
                else:
                    nac_vec = np.array([Hel_dR_k[i, j] / energy_diff for Hel_dR_k in Hel_dR_adia_list])
                nac_tensor[i, j, :] = nac_vec
                nac_tensor[j, i, :] = -nac_vec
        return nac_tensor

    def _mat_correct_phase(self, old_U: np.ndarray, new_U: np.ndarray) -> np.ndarray:
        corr_U = new_U.copy()
        for j in range(new_U.shape[1]):
            if np.real(np.dot(old_U[:, j].conj(), new_U[:, j])) < 0:
                corr_U[:, j] *= -1
        return corr_U
