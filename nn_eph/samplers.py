from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

from jax import jit, lax
from jax import numpy as jnp
from jax import vjp


@dataclass
class continuous_time_1:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=100.0,
    ):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                overlap,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight, carry[0], overlap)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d) + 1.0e-4

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [_, weight, energy, gradient, lene_gradient, qp_weight, _, _], (
            energies,
            qp_weights,
            weights,
            walkers,
            overlaps,
        ) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                gradient,
                lene_gradient,
                qp_weight,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
            walkers,
            overlaps,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class continuous_time:
    n_eql: int
    n_samples: int
    green_reweight: bool = True
    property_function: Callable = lambda *x: 0.0

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=jnp.inf,
    ):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d) + 1.0e-4

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [_, weight, energy, gradient, lene_gradient, qp_weight, _, _], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                gradient,
                lene_gradient,
                qp_weight,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
        )

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling_lr(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=jnp.inf,
    ):
        def _local_energy_and_update_wrapper(x, y, z):
            if self.green_reweight:
                energy, qp_weight, gradient, weight, walker, overlap = (
                    ham.local_energy_and_update_lr(x, y, wave, lattice, z)
                )
            else:
                energy, qp_weight, gradient, weight, walker, overlap = (
                    ham.local_energy_and_update(x, y, wave, lattice, z)
                )
            return energy, (qp_weight, gradient, weight, walker, overlap)

        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                (
                    qp_weight,
                    gradient,
                    weight,
                    carry[0],
                    overlap,
                ),
            ) = _local_energy_and_update_wrapper(
                carry[0], parameters, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            z_n = (
                jnp.abs(overlap) ** 2
                + self.green_reweight * (jnp.abs(gradient * overlap) ** 2).sum()
            ).real
            # z_n = jnp.abs(overlap) ** 2
            carry[2] += (
                weight
                * (jnp.real(energy * jnp.abs(overlap) ** 2 / z_n) - carry[2])
                / carry[1]
            )
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d) + 1.0e-4

        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, metric, h, dev_thresh, median_energy ]
        def scanned_fun_g(carry, x):
            energy, grad_fun, (qp_weight, gradient, weight, carry[0], overlap) = vjp(
                _local_energy_and_update_wrapper,
                carry[0],
                parameters,
                random_numbers[x],
                has_aux=True,
            )
            ene_grad = wave.serialize(grad_fun(1.0 + 0.0j)[1])
            # extended vectors
            ext_gradient = jnp.zeros(wave.n_parameters + 1) + 0.0j
            ext_gradient = ext_gradient.at[1:].set(gradient)
            ext_gradient = ext_gradient.at[0].set(1.0)
            ext_ene_grad = jnp.zeros(wave.n_parameters + 1) + 0.0j
            ext_ene_grad = ext_ene_grad.at[1:].set(ene_grad)
            # ext_ene_grad = ext_ene_grad.at[0].set(energy)
            gradient = ext_gradient
            ene_grad = ext_ene_grad
            weight = (jnp.abs(energy - carry[9]) < carry[8]) * weight + 1.0e-8
            z_n = (
                jnp.abs(overlap) ** 2 * (1 - self.green_reweight)
                + self.green_reweight * (jnp.abs(gradient * overlap) ** 2).sum()
            ).real
            # z_n = jnp.abs(overlap) ** 2
            energy = jnp.where(weight > 1.0e-8, energy, carry[9])
            carry[1] += weight
            carry[2] += (
                weight
                * (jnp.real(energy * jnp.abs(overlap) ** 2 / z_n) - carry[2])
                / carry[1]
            )
            # carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            # carry[4] = (
            #    carry[4]
            #    + weight
            #    * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
            #    / carry[1]
            # )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            carry[6] += (
                weight
                * (
                    (
                        jnp.einsum("i,j->ij", jnp.conj(ext_gradient), ext_gradient)
                        * jnp.abs(overlap) ** 2
                        / z_n
                    )
                    - carry[6]
                )
                / carry[1]
            )
            carry[7] += (
                weight
                * (
                    (
                        jnp.einsum(
                            "i,j->ij", jnp.conj(gradient), ene_grad + energy * gradient
                        )
                        * jnp.abs(overlap) ** 2
                        / z_n
                    )
                    - carry[7]
                )
                / carry[1]
            )
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters + 1)
        lene_gradient = jnp.zeros(wave.n_parameters + 1)
        metric = jnp.zeros((wave.n_parameters + 1, wave.n_parameters + 1)) + 0.0j
        h = jnp.zeros((wave.n_parameters + 1, wave.n_parameters + 1)) + 0.0j
        qp_weight = 0.0
        [_, weight, energy, gradient, lene_gradient, qp_weight, metric, h, _, _], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun_g,
            [
                walker,
                weight,
                energy,
                gradient,
                lene_gradient,
                qp_weight,
                metric,
                h,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            h,
            energies,
            qp_weights,
            weights,
        )

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling_lr_sf(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=jnp.inf,
    ):
        def _local_energy_and_update_wrapper(x, y, z):
            energy, prop, gradient, weight, walker, overlap = (
                ham.local_energy_and_update_lr(x, y, wave, lattice, z, parameters)
            )
            return energy, (prop, gradient, weight, walker, overlap)

        # carry : [ walker, weight, energy, norm, lene_grad, prop, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                (
                    prop,
                    gradient,
                    weight,
                    carry[0],
                    overlap,
                ),
            ) = _local_energy_and_update_wrapper(
                carry[0], parameters, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            z_n = (
                jnp.abs(overlap) ** 2 * (1 - self.green_reweight)
                + self.green_reweight * (jnp.abs(gradient * overlap) ** 2).sum()
            ).real
            # z_n = jnp.abs(overlap) ** 2
            carry[2] += (
                weight
                * (jnp.real(energy * jnp.abs(overlap) ** 2 / z_n) - carry[2])
                / carry[1]
            )
            carry[3] = (
                carry[3]
                + weight * (jnp.real(jnp.abs(overlap) ** 2 / z_n) - carry[3]) / carry[1]
            )
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(prop) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), prop, weight)

        weight = 0.0
        energy = 0.0
        norm = 0.0
        lene_gradient = jnp.zeros(wave.n_parameters)
        prop = jnp.zeros(3) + 0.0j
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, norm, lene_gradient, prop, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d) + 1.0e-4

        # carry : [ walker, weight, energy, norm, prop, vector_property, metric, h, dev_thresh, median_energy ]
        def scanned_fun_g(carry, x):
            vector_property = self.property_function(
                carry[0], parameters, wave, lattice
            )
            energy, grad_fun, (prop, gradient, weight, carry[0], overlap) = vjp(
                _local_energy_and_update_wrapper,
                carry[0],
                parameters,
                random_numbers[x],
                has_aux=True,
            )
            ene_grad = wave.serialize(grad_fun(1.0 + 0.0j)[1])
            ene_grad = jnp.where(jnp.isnan(ene_grad), 0.0, ene_grad)
            gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)

            # extended vectors
            ext_gradient = jnp.zeros(wave.n_parameters + 1) + 0.0j
            ext_gradient = ext_gradient.at[1:].set(gradient)
            ext_gradient = ext_gradient.at[0].set(1.0)
            ext_ene_grad = jnp.zeros(wave.n_parameters + 1) + 0.0j
            ext_ene_grad = ext_ene_grad.at[1:].set(ene_grad)
            # ext_ene_grad = ext_ene_grad.at[0].set(energy)
            gradient = ext_gradient
            ene_grad = ext_ene_grad
            weight = (jnp.abs(energy - carry[9]) < carry[8]) * weight + 1.0e-8
            z_n = (
                jnp.abs(overlap) ** 2 * (1 - self.green_reweight)
                + self.green_reweight * (jnp.abs(gradient * overlap) ** 2).sum()
            ).real
            # prop_norm = jnp.abs(prop) ** 2
            prop_1 = jnp.zeros((gradient.size, 3)) + 0.0j
            prop_1 = prop_1.at[:, :2].set(
                jnp.einsum("i,j->ij", jnp.conj(gradient), prop[:2])
            )
            prop_1 = prop_1.at[:, 2].set(jnp.abs(gradient) ** 2)
            # prop = prop_1
            # z_n = jnp.abs(overlap) ** 2
            energy = jnp.where(weight > 1.0e-8, energy, carry[9])
            carry[1] += weight
            carry[2] += (
                weight
                * (jnp.real(energy * jnp.abs(overlap) ** 2 / z_n) - carry[2])
                / carry[1]
            )
            carry[3] = (
                carry[3]
                + weight * (jnp.real(jnp.abs(overlap) ** 2 / z_n) - carry[3]) / carry[1]
            )
            carry[4] = (
                carry[4]
                + weight
                * ((prop_1 * jnp.abs(overlap) ** 2 / z_n) - carry[4])
                / carry[1]
            )
            carry[5] = (
                carry[5]
                + weight
                * ((vector_property * jnp.abs(overlap) ** 2 / z_n) - carry[5])
                / carry[1]
            )
            carry[6] += (
                weight
                * (
                    (
                        jnp.einsum("i,j->ij", jnp.conj(ext_gradient), ext_gradient)
                        * jnp.abs(overlap) ** 2
                        / z_n
                    )
                    - carry[6]
                )
                / carry[1]
            )
            carry[7] += (
                weight
                * (
                    (
                        jnp.einsum(
                            "i,j->ij", jnp.conj(gradient), ene_grad + energy * gradient
                        )
                        * jnp.abs(overlap) ** 2
                        / z_n
                    )
                    - carry[7]
                )
                / carry[1]
            )
            return carry, (jnp.real(energy), 0.0, weight)

        weight = 0.0
        energy = 0.0
        norm = 0.0
        prop = jnp.zeros((wave.n_parameters + 1, 3)) + 0.0j
        # prop_norm = jnp.zeros(3)
        vector_property = 0.0 * self.property_function(
            walker, parameters, wave, lattice
        )
        metric = jnp.zeros((wave.n_parameters + 1, wave.n_parameters + 1)) + 0.0j
        h = jnp.zeros((wave.n_parameters + 1, wave.n_parameters + 1)) + 0.0j
        [_, weight, energy, norm, prop, vector_property, metric, h, _, _], (
            energies,
            norms,
            weights,
        ) = lax.scan(
            scanned_fun_g,
            [
                walker,
                weight,
                energy,
                norm,
                prop,
                vector_property,
                metric,
                h,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            norm,
            prop,
            vector_property,
            metric,
            h,
            energies,
            norms,
            weights,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class continuous_time_corr:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=100.0,
    ):
        # carry : [ walker, weight, energy, qp_weight, ke, phonon_numbers, eph_corr, phonon_pos, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            ke, phonon_numbers, eph_corr, phonon_pos = ham.local_correlation_functions(
                carry[0], parameters, wave, lattice
            )
            (
                energy,
                qp_weight,
                _,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[-1]) < carry[-2]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[-1])
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] += weight * (jnp.real(qp_weight) - carry[3]) / carry[1]
            carry[4] += weight * (jnp.real(ke) - carry[4]) / carry[1]
            carry[5] += weight * (phonon_numbers - carry[5]) / carry[1]
            carry[6] += weight * (jnp.real(eph_corr) - carry[6]) / carry[1]
            carry[7] += weight * (jnp.real(phonon_pos) - carry[7]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        qp_weight = 0.0
        ke = 0.0
        phonon_numbers = jnp.zeros(lattice.shape)
        eph_corr = 0.0 * phonon_numbers
        phonon_pos = 0.0
        [walker, _, _, _, _, _, _, _, _, _], (energies_eq, _, _) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                qp_weight,
                ke,
                phonon_numbers,
                eph_corr,
                phonon_pos,
                jnp.inf,
                0.0,
            ],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d) + 1.0e-4

        weight = 0.0
        energy = 0.0
        qp_weight = 0.0
        ke = 0.0
        phonon_numbers = jnp.zeros(lattice.shape)
        eph_corr = 0.0 * phonon_numbers
        phonon_pos = 0.0
        [
            _,
            weight,
            energy,
            qp_weight,
            ke,
            phonon_numbers,
            eph_corr,
            phonon_pos,
            _,
            _,
        ], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                qp_weight,
                ke,
                phonon_numbers,
                eph_corr,
                phonon_pos,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            qp_weight,
            ke,
            phonon_numbers,
            eph_corr,
            phonon_pos,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class continuous_time_sr:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
    ):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, metric ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            carry[6] += (
                weight
                * (
                    jnp.real(jnp.einsum("i,j->ij", jnp.conj(gradient), gradient))
                    - carry[6]
                )
                / carry[1]
            )
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        [walker, _, _, _, _, _, _], (_, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, metric],
            jnp.arange(self.n_eql),
        )

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        [_, weight, energy, gradient, lene_gradient, qp_weight, metric], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, metric],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class deterministic:
    max_n_phonon: int
    basis: Sequence
    n_samples: int = 1

    def __init__(self, max_n_phonon, lattice):
        self.max_n_phonon = max_n_phonon
        self.basis = lattice.make_polaron_basis(max_n_phonon)

    # @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight ]
        # def scanned_fun(carry, x):
        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        energies = jnp.zeros(len(self.basis))
        qp_weights = jnp.zeros(len(self.basis))
        weights = jnp.zeros(len(self.basis))
        for i, x in enumerate(self.basis):
            (
                energy_x,
                qp_weight_x,
                gradient_x,
                _,
                _,
                overlap_x,
            ) = ham.local_energy_and_update(x, parameters, wave, lattice, 0.0)
            weight_x = jnp.real(overlap_x.conj() * overlap_x)
            weights = weights.at[i].set(weight_x)
            weight += weight_x
            energy += weight_x * (jnp.real(energy_x) - energy) / weight
            energies = energies.at[i].set(jnp.real(energy_x))
            gradient = gradient + weight_x * (jnp.real(gradient_x) - gradient) / weight
            lene_gradient = (
                lene_gradient
                + weight_x
                * (jnp.real(jnp.conjugate(energy_x) * gradient_x) - lene_gradient)
                / weight
            )
            qp_weight += weight_x * (qp_weight_x - qp_weight) / weight
            qp_weights = qp_weights.at[i].set(qp_weight_x)

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return 0


@dataclass
class deterministic_sr:
    max_n_phonon: int
    basis: Sequence
    n_samples: int = 1

    def __init__(self, max_n_phonon, lattice):
        self.max_n_phonon = max_n_phonon
        self.basis = lattice.make_polaron_basis(max_n_phonon)

    # @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight ]
        # def scanned_fun(carry, x):
        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        energies = jnp.zeros(len(self.basis))
        qp_weights = jnp.zeros(len(self.basis))
        weights = jnp.zeros(len(self.basis))
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        for i, x in enumerate(self.basis):
            (
                energy_x,
                qp_weight_x,
                gradient_x,
                _,
                _,
                overlap_x,
            ) = ham.local_energy_and_update(x, parameters, wave, lattice, 0.0)
            weight_x = jnp.real(overlap_x.conj() * overlap_x)
            weights = weights.at[i].set(weight_x)
            weight += weight_x
            energy += weight_x * (jnp.real(energy_x) - energy) / weight
            energies = energies.at[i].set(jnp.real(energy_x))
            gradient = gradient + weight_x * (jnp.real(gradient_x) - gradient) / weight
            lene_gradient = (
                lene_gradient
                + weight_x
                * (jnp.real(jnp.conjugate(energy_x) * gradient_x) - lene_gradient)
                / weight
            )
            qp_weight += weight_x * (qp_weight_x - qp_weight) / weight
            qp_weights = qp_weights.at[i].set(qp_weight_x)
            metric = (
                metric
                + weight_x
                * (
                    jnp.real(jnp.einsum("i,j->ij", jnp.conj(gradient_x), gradient_x))
                    - metric
                )
                / weight
            )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return 0
