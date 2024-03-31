import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax.scipy as jsp
import numpy as np

# os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, lax
from jax import numpy as jnp
from jax import tree_util, vjp, vmap

from nn_eph import wavefunctions


# base class
class wave_function(ABC):
    """Base class for wave functions"""

    @abstractmethod
    def build_walker_data(self, walker, parameters: Any, lattice: Any) -> Any:
        """Build helpers"""
        pass

    @abstractmethod
    def serialize(self, parameters: Any) -> jnp.ndarray:
        """Serialize the parameters into a single array

        Parameters
        ----------
        parameters : Any
            Parameters of the wave function

        Returns
        -------
        jnp.ndarray
            Serialized parameters
        """
        pass

    @abstractmethod
    def update_parameters(self, parameters: Any, update: jnp.ndarray) -> Sequence:
        """Update the parameters of the wave function

        Parameters
        ----------
        parameters : Any
            Parameters of the wave function
        update : jnp.ndarray
            Update to the parameters

        Returns
        -------
        Sequence
            Updated parameters
        """
        pass

    @abstractmethod
    def calc_overlap(self, walker_data: Any, parameters: Any, lattice: Any) -> complex:
        """Calculate log overlap of the wave function with a walker

        Parameters
        ----------
        walker_data : Any
            Walker along with helper data
        parameters : Any
            Parameters of the wave function
        lattice : Any
            Lattice object

        Returns
        -------
        complex
            Logarithm of the overlap of the wave function with the walker
        """
        pass

    @abstractmethod
    def calc_overlap_ratio(
        self,
        walker_data: Any,
        excitation: Any,
        parameters: Any,
        lattice: Any,
    ) -> complex:
        """Calculate the overlap ratio of the wave function with a walker excitation

        Parameters
        ----------
        walker_data : Any
            Walker along with helper data
        excitation : Any
            Excitation to be applied to the walker
        parameters : Any
            Parameters of the wave function
        lattice : Any
            Lattice object

        Returns
        -------
        complex
            Overlap ratio of the wave function with the walker after the excitation
        """
        pass

    def calc_overlap_ratio_t(
        self,
        walker_data: Any,
        excitation: Any,
        parameters: Any,
        lattice: Any,
    ) -> complex:
        """This function is a slightly hacky workaround for the parity issues arising from translation operations.
        It should be redefined in all reference fermionic wave functions. Does not include parity, unlike calc_overlap_ratio.
        """
        return self.calc_overlap_ratio(walker_data, excitation, parameters, lattice)

    def calc_parity(self, walker_data: Any, excitation: Any, lattice: Any):
        """Another workaround for translataion parity issues, used to undo the parity in the overlap ratio calculation."""
        return 1.0

    def calc_overlap_gradient(
        self, walker_data: Any, parameters: Any, lattice: Any
    ) -> jnp.ndarray:
        """
        Calculate the gradient of the logarithm of the overlap of the wave function with the walker with respect to the parameters

        Parameters
        ----------
        walker_data : dict
            Walker data including walker and a helper matrix r_mat = inv(overlap_mat) used for fast overlap ratio calculations
        parameters : Sequence
            Parameters of the wave function
        lattice : Lattice
            Lattice object

        Returns
        -------
        jnp.ndarray
            Gradient of the log overlap
        """
        _, grad_fun = vjp(self.calc_overlap, walker_data, parameters, lattice)
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[1]
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @abstractmethod
    def __hash__(self):
        pass


@dataclass
class product_state(wave_function):
    """Product of multiple states

    Attributes
    ----------
    states : Tuple
        Tuple of states
    walker_adapters : Sequence
        Converts Hamiltonian dependent walker to state-specific walkers, default is identity
    excitation_adapters : Sequence
        Converts Hamiltonian dependent excitation to state-specific excitations, default is identity
    n_parameters : int
        Number of parameters in the wave function
    """

    states: Tuple[wave_function]
    walker_adapters: Tuple[Callable] = None
    excitation_adapters: Tuple[Callable] = None
    n_parameters: int = None

    def __post_init__(self):
        self.n_parameters = sum([state.n_parameters for state in self.states])
        if self.walker_adapters is None:
            self.walker_adapters = (lambda x: x,) * len(self.states)
        if self.excitation_adapters is None:
            self.excitation_adapters = (lambda x: x,) * len(self.states)

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(
        self, walker: Any, parameters: Sequence, lattice: Any
    ) -> Sequence:
        walker_data = [
            state.build_walker_data(
                self.walker_adapters[i](walker), parameters[i], lattice
            )
            for i, state in enumerate(self.states)
        ]
        return walker_data

    def serialize(self, parameters: Sequence) -> jnp.ndarray:
        return jnp.concatenate(
            [
                state.serialize(parameters_i)
                for state, parameters_i in zip(self.states, parameters)
            ]
        )

    def update_parameters(self, parameters: Sequence, update: jnp.ndarray) -> Sequence:
        counter = 0
        for i in range(len(self.states)):
            n_params = self.states[i].n_parameters
            parameters[i] = self.states[i].update_parameters(
                parameters[i], update[counter : counter + n_params]
            )
            counter += n_params
        return parameters

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: Sequence, parameters: Sequence, lattice: Any
    ) -> complex:
        return jnp.sum(
            jnp.array(
                [
                    state.calc_overlap(walker_data[i], parameters[i], lattice)
                    for i, state in enumerate(self.states)
                ]
            )
        )

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: Sequence,
        excitation: Any,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        return jnp.prod(
            jnp.array(
                [
                    state.calc_overlap_ratio(
                        walker_data[i],
                        self.excitation_adapters[i](excitation),
                        parameters[i],
                        lattice,
                    )
                    for i, state in enumerate(self.states)
                ]
            )
        )

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio_t(
        self,
        walker_data: Sequence,
        excitation: Any,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        return jnp.prod(
            jnp.array(
                [
                    state.calc_overlap_ratio_t(
                        walker_data[i],
                        self.excitation_adapters[i](excitation),
                        parameters[i],
                        lattice,
                    )
                    for i, state in enumerate(self.states)
                ]
            )
        )

    @partial(jit, static_argnums=(0, 3))
    def calc_parity(self, walker_data: Any, excitation: Any, lattice: Any) -> complex:
        return jnp.prod(
            jnp.array(
                [
                    state.calc_parity(
                        walker_data[i],
                        self.excitation_adapters[i](excitation),
                        lattice,
                    )
                    for i, state in enumerate(self.states)
                ]
            )
        )

    def __hash__(self):
        return hash(
            (
                self.n_parameters,
                self.states,
                self.walker_adapters,
                self.excitation_adapters,
            )
        )


@dataclass
class t_projected_state(wave_function):
    """Projected state with a translation operator

    Attributes
    ----------
    state : wave_function
        State to be projected
    walker_translator : Callable
        Function to translate the walker
    excitation_translator : Callable
        Function to translate the excitation
    n_parameters : int
        Number of parameters in the wave function
    """

    state: wave_function
    walker_translator: Callable
    excitation_translator: Callable
    n_parameters: int = None
    k: Tuple = None
    symm_factors: Tuple = None

    def __init__(
        self,
        state: wave_function,
        walker_translator: Callable,
        excitation_translator: Callable,
        n_parameters: int = None,
        k: Tuple = None,
        symm_factors: Tuple = None,
        lattice: Any = None,
    ):
        self.state = state
        self.walker_translator = walker_translator
        self.excitation_translator = excitation_translator
        self.n_parameters = self.state.n_parameters
        self.k = k
        if self.k is None:
            self.k = (0.0,)
        if lattice is not None:
            self.symm_factors = tuple(
                np.sum(
                    2.0j
                    * np.pi
                    * np.array(self.k)
                    @ np.array(site)
                    / np.array(lattice.shape)
                )
                for site in lattice.sites
            )
        else:
            self.symm_factors = (0.0,)

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker: Any, parameters: Any, lattice: Any) -> dict:
        walker_data_0 = self.state.build_walker_data(walker, parameters, lattice)
        translated_walkers = vmap(self.walker_translator, in_axes=(None, 0, None))(
            walker, jnp.array(lattice.sites), lattice
        )
        walker_data_list = vmap(self.state.build_walker_data, in_axes=(0, None, None))(
            translated_walkers, parameters, lattice
        )
        overlaps = vmap(self.state.calc_overlap, in_axes=(0, None, None))(
            walker_data_list, parameters, lattice
        )
        walker_data = {
            "walker_data_0": walker_data_0,
            "walker_data_list": walker_data_list,
        }
        walker_data["log_overlaps"] = overlaps
        walker_data["overlap"] = jnp.sum(
            jnp.exp(overlaps + jnp.array(self.symm_factors))
        )
        walker_data["walker"] = walker
        return walker_data

    def serialize(self, parameters: Any) -> jnp.ndarray:
        return self.state.serialize(parameters)

    def update_parameters(self, parameters: Any, update: jnp.ndarray) -> Sequence:
        return self.state.update_parameters(parameters, update)

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(self, walker_data: dict, parameters: Any, lattice: Any) -> complex:
        overlaps = vmap(self.state.calc_overlap, in_axes=(0, None, None))(
            walker_data["walker_data_list"], parameters, lattice
        )
        return jnp.sum(jsp.special.logsumexp(overlaps + jnp.array(self.symm_factors)))

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: Any,
        parameters: Any,
        lattice: Any,
    ) -> complex:
        translated_excitations = vmap(
            self.excitation_translator, in_axes=(None, 0, None, None)
        )(excitation, jnp.array(lattice.sites), walker_data, lattice)
        ratios = vmap(
            self.state.calc_overlap_ratio_t,
            in_axes=(0, 0, None, None),
        )(
            walker_data["walker_data_list"],
            translated_excitations,
            parameters,
            lattice,
        )
        parity = self.state.calc_parity(
            walker_data["walker_data_0"], excitation, lattice
        )
        new_overlaps = ratios * jnp.exp(walker_data["log_overlaps"])
        return (
            parity
            * jnp.sum(new_overlaps * jnp.exp(jnp.array(self.symm_factors)))
            / walker_data["overlap"]
        )

    def __hash__(self):
        return hash(
            (
                self.n_parameters,
                self.state,
                self.walker_translator,
                self.excitation_translator,
                self.k,
                self.symm_factors,
            )
        )


@partial(jit, static_argnums=(2,))
def walker_translator_ee(
    walker: jnp.ndarray, displacement: jnp.ndarray, lattice: Any
) -> jnp.ndarray:
    """Translate an electronic walker

    Parameters
    ----------
    walker : jnp.ndarray
        Electronic walker as occupation number array [elec_occ_up, elec_occ_dn]
    displacement : jnp.ndarray
        Displacement to be applied to the walker

    Returns
    -------
    jnp.ndarray
        Translated walker
    """
    walker_t = jnp.roll(
        walker, displacement, axis=tuple(range(1, len(lattice.shape) + 1))
    )
    return walker_t


@partial(jit, static_argnums=(3,))
def excitation_translator_ee(
    excitation: jnp.ndarray, displacement: jnp.ndarray, walker_data: Any, lattice: Any
) -> jnp.ndarray:
    """Translate an electronic excitation

    Parameters
    ----------
    excitation : jnp.ndarray
        Excitation to be applied (sigma, i, a) for i,sigma -> a,sigma
    displacement : jnp.ndarray
        Displacement to be applied to the excitation

    Returns
    -------
    jnp.ndarray
        Translated excitation
    """
    # walker_occ = walker_data["walker"]
    # elec_pos = jnp.nonzero(walker_occ[excitation[0]].reshape(-1), size=)[0]
    i_pos = jnp.array(lattice.sites)[excitation["idx"][1]]
    a_pos = jnp.array(lattice.sites)[excitation["idx"][2]]
    new_i_idx = lattice.get_site_num((i_pos + displacement) % jnp.array(lattice.shape))
    new_a_idx = lattice.get_site_num((a_pos + displacement) % jnp.array(lattice.shape))
    excitation_t = excitation
    excitation_t["idx"] = (
        excitation_t["idx"].at[1:].set(jnp.array([new_i_idx, new_a_idx]))
    )
    return excitation_t


@dataclass
class uhf(wave_function):
    """
    Slater determinant for an unrestricted Hartree-Fock state

    Attributes
    ----------
    n_parameters : int
        Number of orbital coefficients
    n_elec : tuple
        Number of spin-up and spin-down electrons
    """

    n_elec: tuple
    n_parameters: int

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(
        self, walker_occ: jnp.ndarray, parameters: Sequence, lattice: Any
    ) -> dict:
        """
        Build helpers for fast local energy evaluation and package with the walker

        Parameters
        ----------
        walker : jnp.ndarray
            Walker to be packaged as occupation number array
        parameters : Sequence
            Orbital coefficient matrices
        lattice : Lattice
            Lattice object

        Returns
        -------
        dict
            Walker data including walker (occ and pos) and a helper matrix r_mat = inv(overlap_mat) used for fast overlap ratio calculations
        """
        elec_pos_up = jnp.nonzero(walker_occ[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker_occ[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        overlap_mat_up = parameters[0][elec_pos_up, :]
        overlap_mat_dn = parameters[1][elec_pos_dn, :]
        overlap = jnp.linalg.det(overlap_mat_up) * jnp.linalg.det(overlap_mat_dn)
        r_mat = (
            parameters[0] @ jnp.linalg.inv(overlap_mat_up),
            parameters[1] @ jnp.linalg.inv(overlap_mat_dn),
        )
        return {
            "walker": walker_pos,
            "walker_occ": walker_occ,
            "r_mat": r_mat,
            "overlap": overlap,
        }

    def serialize(self, parameters: Sequence) -> jnp.ndarray:
        return jnp.concatenate(
            [parameters_i.reshape(-1) for parameters_i in parameters]
        )

    def update_parameters(self, parameters: Sequence, update: jnp.ndarray) -> Sequence:
        for i in range(len(parameters)):
            parameters[i] = parameters[i] + update[
                parameters[i].size * i : parameters[i].size * (i + 1)
            ].reshape(parameters[i].shape)
        return parameters

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker"]
        overlap_up = jnp.linalg.det(parameters[0][walker[0], :])
        overlap_dn = jnp.linalg.det(parameters[1][walker[1], :])
        return jnp.log(overlap_up * overlap_dn)

    # only supports single excitation for now
    # does not consider parity
    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: dict,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        excitation_sm = excitation["sm_idx"]
        return lax.cond(
            excitation_sm[0] == 0,
            lambda x: walker_data["r_mat"][0][excitation_sm[2], excitation_sm[1]],
            lambda x: walker_data["r_mat"][1][excitation_sm[2], excitation_sm[1]],
            0.0,
        )

    @partial(jit, static_argnums=(0, 3))
    def calc_parity(self, walker_data: Any, excitation: Any, lattice: Any):
        spin = excitation["sm_idx"][0]
        i_pos = excitation["sm_idx"][1]
        a_abs_pos = excitation["sm_idx"][2]
        a_pos = lax.cond(
            spin == 0,
            lambda x: jnp.searchsorted(walker_data["walker"][0], a_abs_pos),
            lambda x: jnp.searchsorted(walker_data["walker"][1], a_abs_pos),
            0,
        )
        return (i_pos < a_pos) * (-1) ** (a_pos - i_pos - 1) + (i_pos >= a_pos) * (
            -1
        ) ** (a_pos - i_pos)

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio_t(
        self, walker_data: dict, excitation: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        spin = excitation["idx"][0]
        excitation_idx = excitation["idx"]
        i_abs_pos = excitation_idx[1]
        i_rel_pos = lax.cond(
            spin == 0,
            lambda x: jnp.searchsorted(walker_data["walker"][0], i_abs_pos),
            lambda x: jnp.searchsorted(walker_data["walker"][1], i_abs_pos),
            0,
        )
        a_abs_pos = excitation_idx[2]
        a_rel_pos = lax.cond(
            spin == 0,
            lambda x: jnp.searchsorted(walker_data["walker"][0], a_abs_pos),
            lambda x: jnp.searchsorted(walker_data["walker"][1], a_abs_pos),
            0,
        )
        parity = (i_rel_pos < a_rel_pos) * (-1) ** (a_rel_pos - i_rel_pos - 1) + (
            i_rel_pos >= a_rel_pos
        ) * (-1) ** (a_rel_pos - i_rel_pos)
        excitation["sm_idx"] = (spin, i_rel_pos, a_abs_pos)
        return parity * self.calc_overlap_ratio(
            walker_data, excitation, parameters, lattice
        )

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio_t_slow(
        self, walker_data: dict, excitation: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker_occ"].copy()
        excitation_idx = excitation["idx"]
        i_pos = jnp.array(lattice.sites)[excitation_idx[1]]
        a_pos = jnp.array(lattice.sites)[excitation_idx[2]]
        walker = lax.cond(
            excitation_idx[0] == 0,
            lambda x: walker.at[(0, *i_pos)].set(0),
            lambda x: walker.at[(1, *i_pos)].set(0),
            i_pos,
        )
        walker = lax.cond(
            excitation_idx[0] == 0,
            lambda x: walker.at[(0, *a_pos)].set(1),
            lambda x: walker.at[(1, *a_pos)].set(1),
            a_pos,
        )

        elec_pos_up = jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        overlap_up = jnp.linalg.det(parameters[0][elec_pos_up, :])
        overlap_dn = jnp.linalg.det(parameters[1][elec_pos_dn, :])
        ratio = overlap_up * overlap_dn / walker_data["overlap"]
        return ratio

    def __hash__(self):
        return hash((self.n_parameters, self.n_elec))


@dataclass
class complex_uhf(uhf):
    """complex uhf"""

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker_occ, parameters: Sequence, lattice: Any) -> dict:
        elec_pos_up = jnp.nonzero(walker_occ[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker_occ[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        parameters_1 = parameters[2] + 1.0j * parameters[3]
        overlap_mat_up = parameters_0[elec_pos_up, :]
        overlap_mat_dn = parameters_1[elec_pos_dn, :]
        overlap = jnp.linalg.det(overlap_mat_up) * jnp.linalg.det(overlap_mat_dn)
        r_mat = (
            parameters_0 @ jnp.linalg.inv(overlap_mat_up),
            parameters_1 @ jnp.linalg.inv(overlap_mat_dn),
        )
        return {
            "walker": walker_pos,
            "walker_occ": walker_occ,
            "r_mat": r_mat,
            "overlap": overlap,
        }

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker"]
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        parameters_1 = parameters[2] + 1.0j * parameters[3]
        overlap = jnp.linalg.det(parameters_0[walker[0], :]) * jnp.linalg.det(
            parameters_1[walker[1], :]
        )
        return jnp.log(overlap + 0.0j)

    def __hash__(self):
        return hash((self.n_parameters, self.n_elec))


@dataclass
class kuhf(uhf):
    """Complex projected uhf, complex orbitals, real overlap"""

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker_occ, parameters: Sequence, lattice: Any) -> dict:
        elec_pos_up = jnp.nonzero(walker_occ[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker_occ[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        parameters_1 = parameters[2] + 1.0j * parameters[3]
        overlap_mat_up = parameters_0[elec_pos_up, :]
        overlap_mat_dn = parameters_1[elec_pos_dn, :]
        complex_overlap = jnp.linalg.det(overlap_mat_up) * jnp.linalg.det(
            overlap_mat_dn
        )
        r_mat = (
            parameters_0 @ jnp.linalg.inv(overlap_mat_up),
            parameters_1 @ jnp.linalg.inv(overlap_mat_dn),
        )
        return {
            "walker": walker_pos,
            "walker_occ": walker_occ,
            "r_mat": r_mat,
            "complex_overlap": complex_overlap,
        }

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker"]
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        parameters_1 = parameters[2] + 1.0j * parameters[3]
        overlap = jnp.linalg.det(parameters_0[walker[0], :]) * jnp.linalg.det(
            parameters_1[walker[1], :]
        )
        return jnp.log((overlap).real + 0.0j)

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: jnp.ndarray,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        excitation_sm = excitation["sm_idx"]
        complex_overlap_ratio = lax.cond(
            excitation_sm[0] == 0,
            lambda x: walker_data["r_mat"][0][excitation_sm[2], excitation_sm[1]],
            lambda x: walker_data["r_mat"][1][excitation_sm[2], excitation_sm[1]],
            0.0,
        )
        return (
            complex_overlap_ratio * walker_data["complex_overlap"]
        ).real / walker_data["complex_overlap"].real

    def __hash__(self):
        return hash((self.n_parameters, self.n_elec))


@dataclass
class ghf(uhf):
    """Generalized Hartree-Fock, real orbitals, real overlap"""

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker_occ, parameters: Sequence, lattice: Any) -> dict:
        n_sites = lattice.n_sites
        elec_pos_up = jnp.nonzero(walker_occ[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker_occ[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        walker_pos_ghf = jnp.concatenate((elec_pos_up, elec_pos_dn + n_sites))
        overlap_mat = parameters[0][walker_pos_ghf, :]
        r_mat = parameters[0] @ jnp.linalg.inv(overlap_mat)
        return {
            "walker": walker_pos,
            "walker_ghf": walker_pos_ghf,
            "walker_occ": walker_occ,
            "r_mat": r_mat,
        }

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker_ghf"]
        overlap = jnp.linalg.det(parameters[0][walker, :])
        return jnp.log(overlap)

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: jnp.ndarray,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        n_sites = lattice.n_sites
        excitation_sm = excitation["sm_idx"]
        sigma = excitation_sm[0]
        return walker_data["r_mat"][
            excitation_sm[2] + sigma * n_sites,
            excitation_sm[1] + sigma * self.n_elec[0],
        ]

    def __hash__(self):
        return hash((self.n_parameters, self.n_elec))


@dataclass
class kghf(ghf):
    """Complex projected Generalized Hartree-Fock, complex orbitals, real overlap"""

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker_occ, parameters: Sequence, lattice: Any) -> dict:
        n_sites = lattice.n_sites
        elec_pos_up = jnp.nonzero(walker_occ[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker_occ[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        walker_pos_ghf = jnp.concatenate((elec_pos_up, elec_pos_dn + n_sites))
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        overlap_mat = parameters_0[walker_pos_ghf, :]
        complex_overlap = jnp.linalg.det(overlap_mat)
        r_mat = parameters_0 @ jnp.linalg.inv(overlap_mat)
        return {
            "walker": walker_pos,
            "walker_ghf": walker_pos_ghf,
            "walker_occ": walker_occ,
            "r_mat": r_mat,
            "complex_overlap": complex_overlap,
        }

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        walker = walker_data["walker_ghf"]
        parameters_0 = parameters[0] + 1.0j * parameters[1]
        overlap = jnp.linalg.det(parameters_0[walker, :])
        return jnp.log((overlap).real + 0.0j)

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: jnp.ndarray,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        n_sites = lattice.n_sites
        excitation_sm = excitation["sm_idx"]
        sigma = excitation_sm[0]
        complex_overlap_ratio = walker_data["r_mat"][
            excitation_sm[2] + sigma * n_sites,
            excitation_sm[1] + sigma * self.n_elec[0],
        ]
        return (
            complex_overlap_ratio * walker_data["complex_overlap"]
        ).real / walker_data["complex_overlap"].real

    def __hash__(self):
        return hash((self.n_parameters, self.n_elec))


@dataclass
class nn(wave_function):
    """General neural network wave function

    NB: in the case of electrons, this wave function does not include parity factors in overlap ratios

    Attributes
    ----------
    nn_apply : Callable
        Neural network function
    apply_excitation : Callable
        Function to apply an excitation to the walker
    n_parameters : int
        Number of parameters in the wave function
    """

    nn_apply: Callable
    apply_excitation: Callable
    n_elec: tuple
    n_parameters: int

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(
        self, walker: jnp.ndarray, parameters: Any, lattice: Any
    ) -> dict:
        walker_data = {"walker_occ": walker}
        walker_data["log_overlap"] = self.calc_overlap(walker_data, parameters, lattice)
        elec_pos_up = jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_dn = jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        walker_pos = (elec_pos_up, elec_pos_dn)
        walker_data["walker_pos"] = walker_pos
        return walker_data

    def serialize(self, parameters: Any) -> jnp.ndarray:
        flat_tree = tree_util.tree_flatten(parameters)[0]
        serialized = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized = jnp.concatenate((serialized, jnp.reshape(params, -1)))
        return serialized

    def update_parameters(self, parameters: Any, update: jnp.ndarray) -> Any:
        flat_tree, tree = tree_util.tree_flatten(parameters)
        counter = 0
        for i in range(len(flat_tree)):
            flat_tree[i] += update[counter : counter + flat_tree[i].size].reshape(
                flat_tree[i].shape
            )
            counter += flat_tree[i].size
        return tree_util.tree_unflatten(tree, flat_tree)

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(self, walker_data: dict, parameters: Any, lattice: Any) -> complex:
        inputs = walker_data["walker_occ"]
        outputs = jnp.array(self.nn_apply(parameters, inputs + 0.0j), dtype="complex64")
        return outputs[0]

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: Any,
        parameters: Any,
        lattice: Any,
    ) -> complex:
        """No fast overlap ratio, so simply calculates new overlaps"""
        new_walker = self.apply_excitation(
            walker_data["walker_occ"], excitation, lattice
        )
        outputs = jnp.array(
            self.nn_apply(parameters, new_walker + 0.0j), dtype="complex64"
        )
        return jnp.exp(outputs[0] - walker_data["log_overlap"])

    def __hash__(self):
        return hash((self.n_parameters, self.nn_apply, self.apply_excitation))


@partial(jit, static_argnums=(2,))
def apply_excitation_ee(
    walker: jnp.ndarray, excitation: jnp.ndarray, lattice: Any
) -> jnp.ndarray:
    """Apply electron excitation to an electronic walker

    Parameters
    ----------
    walker : jnp.ndarray
        Electronic walker as occupation number array [elec_occ_up, elec_occ_dn]
    excitation : jnp.ndarray
        Excitation to be applied (sigma, i, a) for i,sigma -> a,sigma
    lattice : Lattice
        Lattice object

    Returns
    -------
    jnp.ndarray
        New walker after excitation
    """
    walker_copy = walker.copy()
    i_pos = jnp.array(lattice.sites)[excitation["idx"][1]]
    a_pos = jnp.array(lattice.sites)[excitation["idx"][2]]
    walker_copy = walker_copy.at[(excitation["idx"][0], *i_pos)].set(0)
    walker_copy = walker_copy.at[(excitation["idx"][0], *a_pos)].set(1)
    return walker_copy


@dataclass
class nn_jastrow_n(wave_function):
    """
    Neural net electron phonon jastrow on top of an electronic mean field state

    Attributes
    ----------
    nn_apply_r : Callable
        Neural network function for the absolute value of the amplitude
    nn_apply_phi : Callable
        Neural network function for the phase of the amplitude
    reference : Any
        Reference wave function
    n_parameters : int
        Number of parameters in the wave function
        These are set in the constructor as only the number of NN parameters but get modified to include reference parameters after construction
    mask : Sequence
        Determines which parameters are optimized
    get_input : Callable
        Function to get the input for the NN
    """

    nn_apply_r: Callable
    nn_apply_phi: Callable
    reference: Any
    n_parameters: int
    mask: Sequence = None
    get_input: Callable = wavefunctions.get_input_r

    def __post_init__(self):
        self.n_parameters += self.reference.n_parameters
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    @partial(jit, static_argnums=(0, 3))
    def build_walker_data(self, walker, parameters: Sequence, lattice: Any) -> dict:
        """
        Build helpers for fast local energy evaluation and package with the walker

        Parameters
        ----------
        walker : Sequence
            walker : [ (elec_occ_up, elec_occ_dn), phonon_occ ]
        parameters : Sequence
            Parameters of the wave function
        lattice : Lattice
            Lattice object

        Returns
        -------
        dict
            Walker data including walker (occ and pos) and a helper matrix r_mat = inv(overlap_mat) used for fast overlap ratio calculations
        """
        elec_occ = walker[0]
        phonon_occ = walker[1]
        ref_walker_data = self.reference.build_walker_data(
            elec_occ, parameters[2], lattice
        )
        return {
            "walker": walker,
            "phonon_occ": phonon_occ,
            "ref_walker_data": ref_walker_data,
        }

    def serialize(self, parameters: Sequence) -> jnp.ndarray:
        """
        Serialize the parameters into a single array

        Parameters
        ----------
        parameters : Sequence
            Parameters of the wave function: nn_r, nn_phi, reference

        Returns
        -------
        jnp.ndarray
            Serialized parameters
        """
        # nn_r
        flat_tree = tree_util.tree_flatten(parameters[0])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))

        # nn_phi
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))

        serialized = jnp.concatenate(
            (serialized_1, serialized_2, self.reference.serialize(parameters[2]))
        )
        return serialized

    def update_parameters(self, parameters: Sequence, update: jnp.ndarray) -> Sequence:
        """
        Update the parameters of the wave function

        Parameters
        ----------
        parameters : Sequence
            Parameters of the wave function
        update : jnp.ndarray
            Update to the parameters

        Returns
        -------
        Sequence
            Updated parameters
        """
        flat_tree, tree = tree_util.tree_flatten(parameters[0])
        counter = 0
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[0] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)

        parameters[2] = self.reference.update_parameters(
            parameters[2], update[counter:]
        )
        return parameters

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(
        self, walker_data: dict, parameters: Sequence, lattice: Any
    ) -> complex:
        """
        Calculate the overlap of the wave function with a walker

        Parameters
        ----------
        walker_data : dict
            Helper data for the walker
        parameters : Sequence
            Parameters of the wave function
        lattice : Lattice
            Lattice object

        Returns
        -------
        complex
            Logarithm of the overlap of the wave function with the walker
        """
        ref_walker_data = walker_data["ref_walker_data"]
        ref_overlap = self.reference.calc_overlap(
            ref_walker_data, parameters[2], lattice
        )
        inputs = self.get_input(walker_data["walker"])
        outputs_r = jnp.array(
            self.nn_apply_r(parameters[0], inputs + 0.0j), dtype="complex64"
        )
        outputs_phi = jnp.array(
            self.nn_apply_phi(parameters[1], inputs + 0.0j), dtype="complex64"
        )
        nn_overlap = outputs_r[0] + 1.0j * jnp.sum(outputs_phi)
        return ref_overlap + nn_overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_ratio(
        self,
        walker_data: dict,
        excitation: jnp.ndarray,
        parameters: Sequence,
        lattice: Any,
    ) -> complex:
        """
        Calculate the overlap ratio of the wave function with a walker after a single electronic excitation

        Parameters
        ----------
        walker_data : dict
            Helper data for the walker
        excitation : jnp.ndarray
            Excitation to be performed on the walker (sigma, i, a) for i,sigma -> a,sigma
        parameters : Sequence
            Parameters of the wave function
        lattice : Lattice
            Lattice object

        Returns
        -------
        complex
            overlap ratio of the wave function with the walker after the excitation
        """
        ref_walker_data = walker_data["ref_walker_data"]
        ref_overlap_ratio = self.reference.calc_overlap_ratio(
            ref_walker_data, excitation, parameters[2], lattice
        )
        return ref_overlap_ratio

    def __hash__(self):
        return hash(
            (
                self.nn_apply_r,
                self.nn_apply_phi,
                self.reference,
                self.n_parameters,
                self.mask,
                self.get_input,
            )
        )
