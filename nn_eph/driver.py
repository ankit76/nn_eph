import pickle
import time
from functools import partial

import numpy as np

from nn_eph import config

config.setup_jax()

import jax.numpy as jnp
from jax import random, vmap

from nn_eph import samplers, stat_utils

print = partial(print, flush=True)


def driver(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    n_steps=1000,
    step_size=0.1,
    seed=0,
    print_stats=True,
    dev_thresh_fac=1.0e6,
    n_walkers=1,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    moment_1 = jnp.zeros(wave.n_parameters)
    moment_2 = jnp.zeros(wave.n_parameters)
    decay_1 = 0.1
    decay_2 = 0.01
    momentum = 0.0
    key = random.PRNGKey(seed + rank)

    if rank == 0:
        print(f"# iter       ene           qp_weight        grad_norm           time")
    iter_energy = []
    calc_time = 0.0
    best_parameters = parameters.copy()
    min_energy = np.inf
    # dev_thresh_fac = 1.0e6
    for iteration in range(n_steps):
        key, subkey = random.split(key)
        random_numbers = random.uniform(
            subkey,
            shape=(
                n_walkers,
                sampler.n_samples,
            ),
        )
        init = time.time()
        batched_sampling = vmap(
            sampler.sampling,
            in_axes=(
                None,
                None,
                None,
                None,
                None,
                0,
                None,
            ),  # Batch only over random_numbers
        )
        (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
        ) = batched_sampling(
            walker, ham, parameters, wave, lattice, random_numbers, dev_thresh_fac
        )
        weight_per_walker = jnp.sum(weights, axis=1)
        gradient = weight_per_walker.reshape(-1, 1) * gradient
        gradient = jnp.sum(gradient, axis=0)
        lene_gradient = weight_per_walker.reshape(-1, 1) * lene_gradient
        lene_gradient = jnp.sum(lene_gradient, axis=0)
        gradient /= jnp.sum(weight_per_walker)
        lene_gradient /= jnp.sum(weight_per_walker)

        energies *= weights
        energies = jnp.sum(energies, axis=0)
        qp_weights *= weights
        qp_weights = jnp.sum(qp_weights, axis=0)
        weights = jnp.sum(weights, axis=0)
        energies /= weights
        qp_weights /= weights

        # reject outliers
        samples_clean, _ = stat_utils.reject_outliers(
            np.stack((energies, qp_weights, weights)).T, 0, dev_thresh_fac / 100.0
        )
        energies_clean = samples_clean[:, 0]
        qp_weights_clean = samples_clean[:, 1]
        weights_clean = samples_clean[:, 2]
        weight = np.sum(weights_clean)
        energy = np.average(energies_clean, weights=weights_clean)
        qp_weight = np.average(qp_weights_clean, weights=weights_clean)

        # average and print energies for the current step
        weight = np.array([weight], dtype="float32")
        energy = np.array([weight * energy], dtype="float32")
        qp_weight = np.array([weight * qp_weight], dtype="float32")
        # energy = np.array([ energy ])
        total_weight = 0.0 * weight
        total_energy = 0.0 * weight
        total_qp_weight = 0.0 * weight
        gradient = np.array(weight * gradient, dtype="float32")
        lene_gradient = np.array(weight * lene_gradient, dtype="float32")
        # gradient = np.array(gradient)
        # lene_gradient = np.array(lene_gradient)
        total_gradient = 0.0 * gradient
        total_lene_gradient = 0.0 * lene_gradient

        comm.barrier()
        comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
        comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
        comm.Reduce(
            [qp_weight, MPI.FLOAT], [total_qp_weight, MPI.FLOAT], op=MPI.SUM, root=0
        )
        comm.Reduce(
            [gradient, MPI.FLOAT], [total_gradient, MPI.FLOAT], op=MPI.SUM, root=0
        )
        comm.Reduce(
            [lene_gradient, MPI.FLOAT],
            [total_lene_gradient, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        comm.barrier()

        new_parameters = None
        if rank == 0:
            total_energy /= total_weight
            total_qp_weight /= total_weight
            total_gradient /= total_weight
            total_lene_gradient /= total_weight

            ene_gradient = 2 * total_lene_gradient - 2 * total_gradient * total_energy
            calc_time += time.time() - init
            print(
                f"{iteration: 5d}   {total_energy[0]: .6e}    {total_qp_weight[0]: .6e}   {jnp.linalg.norm(ene_gradient): .6e}     {calc_time: .6e}"
            )
            # print(f'iter: {iteration: 5d}, ene: {total_energy[0]: .6e}, qp_weight: {total_qp_weight[0]: .6e}, grad: {jnp.linalg.norm(ene_gradient): .6e}')
            # print(f'total_weight: {total_weight}')
            # print(f'total_energy: {total_energy}')
            # print(f'total_gradient: {total_gradient}')

            if iteration == 0:
                update = step_size * ene_gradient
                moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
                moment_2 = jnp.maximum(
                    moment_2,
                    decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2,
                )
            else:
                moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
                moment_2 = jnp.maximum(
                    moment_2,
                    decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2,
                )
                update = step_size * moment_1 / (moment_2**0.5 + 1.0e-8)
                # update = momentum * update + step_size * ene_gradient

            new_parameters = wave.update_parameters(parameters, -update)
            if iteration > 0:
                with open("parameters.bin", "ab") as fh:
                    pickle.dump(new_parameters, fh)
            else:
                with open("parameters.bin", "wb") as fh:
                    pickle.dump(new_parameters, fh)

            if iteration > 0.8 * n_steps:
                if total_energy[0] < min_energy:
                    best_parameters = parameters.copy()
                    min_energy = total_energy[0]
                    with open("best_parameters.bin", "wb") as fh:
                        pickle.dump(best_parameters, fh)
                iter_energy.append(total_energy[0])

            # new_parameters = [ update_parameters(parameters[0], -update), parameters[1] ]
            # new_parameters = parameters - update
            # new_parameters = new_parameters.at[0].set(0.)
            # print(f'new_parameters: {new_parameters}')

        comm.barrier()
        parameters = comm.bcast(new_parameters, root=0)
        comm.barrier()

    lowest_energy = 0.0
    comm.barrier()
    parameters = comm.bcast(best_parameters, root=0)
    comm.barrier()

    # final accurate run
    n_eql = 10 * sampler.n_eql
    n_samples = 10 * sampler.n_samples
    sampler_1 = samplers.continuous_time(n_eql, n_samples)
    key, subkey = random.split(key)
    random_numbers = random.uniform(
        subkey,
        shape=(
            n_walkers,
            sampler_1.n_samples,
        ),
    )
    batched_sampling = vmap(
        sampler_1.sampling,
        in_axes=(
            None,
            None,
            None,
            None,
            None,
            0,
            None,
        ),  # Batch only over random_numbers
    )
    out = batched_sampling(
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac * 100.0,
    )
    energies = out[-3]
    qp_weights = out[-2]
    weights = out[-1]
    energies *= weights
    energies = jnp.sum(energies, axis=0)
    qp_weights *= weights
    qp_weights = jnp.sum(qp_weights, axis=0)
    weights = jnp.sum(weights, axis=0)
    energies /= weights
    qp_weights /= weights

    samples_clean, _ = stat_utils.reject_outliers(
        np.stack((energies, qp_weights, weights)).T, 0, 1000.0
    )
    energies_clean = samples_clean[:, 0].reshape(-1)
    qp_weights_clean = samples_clean[:, 1].reshape(-1)
    weights_clean = samples_clean[:, 2].reshape(-1)
    weight = np.sum(weights_clean)
    energy = np.average(energies_clean, weights=weights_clean)

    weight = np.array([weight], dtype="float32")
    energy = np.array([weight * energy], dtype="float32")
    total_weight = 0.0 * weight
    total_energy = 0.0 * weight

    comm.barrier()
    comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.barrier()

    weights = np.array(weights, dtype="float64")
    energies = np.array(weights * energies, dtype="float64")
    total_weights = 0.0 * weights
    total_energies = 0.0 * weights
    comm.barrier()
    comm.Reduce([weights, MPI.DOUBLE], [total_weights, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce(
        [energies, MPI.DOUBLE], [total_energies, MPI.DOUBLE], op=MPI.SUM, root=0
    )
    comm.barrier()

    if rank == 0:
        total_energy /= total_weight
        total_energies /= total_weights
        print("Clean energy: ", total_energy[0])

        samples_clean, _ = stat_utils.reject_outliers(
            np.stack((total_energies, total_weights)).T,
            0,
            dev_thresh_fac / 1000.0,
            printQ=True,
        )
        energies_clean = samples_clean[:, 0].reshape(-1)
        weights_clean = samples_clean[:, 1].reshape(-1)

        lowest_energy, _ = stat_utils.blocking_analysis(
            weights_clean,
            energies_clean,
            neql=0,
            printQ=print_stats,
            writeBlockedQ=False,
        )
        print(f"Lowest energy: {lowest_energy}")

    lowest_energy = comm.bcast(lowest_energy, root=0)

    return lowest_energy, parameters


def driver_corr(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    seed=0,
    print_stats=True,
    dev_thresh_fac=1.0e6,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    key = random.PRNGKey(seed + rank)

    # dev_thresh_fac = 1.0e6
    key, subkey = random.split(key)
    random_numbers = random.uniform(subkey, shape=(sampler.n_samples,))
    (
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
    ) = sampler.sampling(
        walker, ham, parameters, wave, lattice, random_numbers, dev_thresh_fac
    )

    # average
    weight = np.array([weight], dtype="float32")
    energy = np.array([weight * energy], dtype="float32")
    qp_weight = np.array([weight * qp_weight], dtype="float32")
    ke = np.array([weight * ke], dtype="float32")
    phonon_numbers = np.array(weight * phonon_numbers, dtype="float32")
    eph_corr = np.array(weight * eph_corr, dtype="float32")
    phonon_pos = np.array([weight * phonon_pos], dtype="float32")
    # energy = np.array([ energy ])
    total_weight = 0.0 * weight
    total_energy = 0.0 * weight
    total_qp_weight = 0.0 * weight
    total_ke = 0.0 * weight
    total_phonon_numbers = 0.0 * phonon_numbers
    total_eph_corr = 0.0 * eph_corr
    total_phonon_pos = 0.0 * phonon_pos

    comm.barrier()
    comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce(
        [qp_weight, MPI.FLOAT], [total_qp_weight, MPI.FLOAT], op=MPI.SUM, root=0
    )
    comm.Reduce([ke, MPI.FLOAT], [total_ke, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce(
        [phonon_numbers, MPI.FLOAT],
        [total_phonon_numbers, MPI.FLOAT],
        op=MPI.SUM,
        root=0,
    )
    comm.Reduce([eph_corr, MPI.FLOAT], [total_eph_corr, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce(
        [phonon_pos, MPI.FLOAT], [total_phonon_pos, MPI.FLOAT], op=MPI.SUM, root=0
    )
    comm.barrier()

    if rank == 0:
        total_energy /= total_weight
        total_qp_weight /= total_weight
        total_ke /= total_weight
        total_phonon_numbers /= total_weight
        total_eph_corr /= total_weight
        total_phonon_pos /= total_weight

        np.savetxt("phonon_numbers.dat", total_phonon_numbers)
        np.savetxt("eph_corr.dat", total_eph_corr)

    comm.barrier()
    total_energy = comm.bcast(total_energy, root=0)
    total_qp_weight = comm.bcast(total_qp_weight, root=0)
    total_ke = comm.bcast(total_ke, root=0)
    total_phonon_pos = comm.bcast(total_phonon_pos, root=0)
    comm.barrier()

    return (
        total_energy,
        total_qp_weight,
        total_ke,
        total_phonon_pos[0],
    )


def driver_sr(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    n_steps=1000,
    step_size=0.1,
    seed=0,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    moment_1 = jnp.zeros(wave.n_parameters)
    moment_2 = jnp.zeros(wave.n_parameters)
    decay_1 = 0.1
    decay_2 = 0.01
    key = random.PRNGKey(seed + rank)

    if rank == 0:
        print(f"# iter       ene           qp_weight        grad_norm           time")

    calc_time = 0.0
    for iteration in range(n_steps):
        key, subkey = random.split(key)
        random_numbers = random.uniform(subkey, shape=(sampler.n_samples,))
        # g_scan = g #* (1 - jnp.exp(-iteration / 10))
        # starter_iters = 100
        # if iteration < starter_iters:
        #  g_scan = 0.
        # else:
        #  g_scan = g * (1. - jnp.exp(-(iteration - starter_iters) / 500))
        # g_scan = 6. - (6. - g) * (1. - jnp.exp(-(iteration - starter_iters) / 500))
        init = time.time()
        (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            energies,
            qp_weights,
            weights,
        ) = sampler.sampling(walker, ham, parameters, wave, lattice, random_numbers)

        # average and print energies for the current step
        weight = np.array([weight], dtype="float32")
        energy = np.array([weight * energy], dtype="float32")
        qp_weight = np.array([weight * qp_weight], dtype="float32")
        metric = np.array(weight * metric, dtype="float32")
        # energy = np.array([ energy ])
        total_weight = 0.0 * weight
        total_energy = 0.0 * weight
        total_qp_weight = 0.0 * weight
        total_metric = 0.0 * metric
        gradient = np.array(weight * gradient, dtype="float32")
        lene_gradient = np.array(weight * lene_gradient, dtype="float32")
        # gradient = np.array(gradient)
        # lene_gradient = np.array(lene_gradient)
        total_gradient = 0.0 * gradient
        total_lene_gradient = 0.0 * lene_gradient

        comm.barrier()
        comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
        comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
        comm.Reduce(
            [qp_weight, MPI.FLOAT], [total_qp_weight, MPI.FLOAT], op=MPI.SUM, root=0
        )
        comm.Reduce([metric, MPI.FLOAT], [total_metric, MPI.FLOAT], op=MPI.SUM, root=0)
        comm.Reduce(
            [gradient, MPI.FLOAT], [total_gradient, MPI.FLOAT], op=MPI.SUM, root=0
        )
        comm.Reduce(
            [lene_gradient, MPI.FLOAT],
            [total_lene_gradient, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        comm.barrier()

        new_parameters = None
        if rank == 0:
            total_energy /= total_weight
            total_qp_weight /= total_weight
            total_metric /= total_weight
            total_gradient /= total_weight
            total_lene_gradient /= total_weight

            metric = np.zeros((wave.n_parameters + 1, wave.n_parameters + 1))
            metric[0, 0] = 1.0
            metric[1:, 1:] = total_metric + np.diag(np.ones(wave.n_parameters) * 1.0e-3)
            metric[0, 1:] = total_gradient
            metric[1:, 0] = total_gradient
            try:
                ev, _ = np.linalg.eigh(metric)
            except:
                if rank == 0:
                    np.savetxt(f"metric_{iteration}.dat", metric)
                    print(f"total energy: {total_energy}")
                    print(f"gradient: {total_gradient}")
                    print(f"lene gradient: {total_lene_gradient}")
                    print(f"parameters:\n{parameters}")
                exit()
            y_vec = np.zeros(wave.n_parameters + 1)
            y_vec[0] = 1 - step_size * total_energy
            y_vec[1:] = total_gradient - total_lene_gradient * step_size
            update = np.linalg.solve(metric, y_vec)
            # np.savetxt(f'update_{iteration}', update)
            update = update[1:] / update[0]
            update[np.abs(update) > 1.0e3 * 0.999**iteration] = 0.0
            new_parameters = wave.update_parameters(parameters, update)
            with open("parameters.bin", "wb") as fh:
                pickle.dump(new_parameters, fh)

            ene_gradient = 2 * total_lene_gradient - 2 * total_gradient * total_energy
            calc_time += time.time() - init
            print(
                f"{iteration: 5d}   {total_energy[0]: .6e}    {total_qp_weight[0]: .6e}   {jnp.linalg.norm(ene_gradient): .6e}     {calc_time: .6e}"
            )
            # print(f'iter: {iteration: 5d}, ene: {total_energy[0]: .6e}, qp_weight: {total_qp_weight[0]: .6e}, grad: {jnp.linalg.norm(ene_gradient): .6e}')
            # print(f'parameters: {parameters}')
            # print(f'total_weight: {total_weight}')
            # print(f'total_energy: {total_energy}')
            # print(f'total_gradient: {total_gradient}')

            # if iteration == 0:
            #  update = step_size * ene_gradient
            #  moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
            #  moment_2 = jnp.maximum(moment_2, decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2)
            # else:
            #  moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
            #  moment_2 = jnp.maximum(moment_2, decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2)
            #  update = step_size * moment_1 / (moment_2**0.5 + 1.e-8)
            #  #update = momentum * update + step_size * ene_gradient
            # new_parameters = wave.update_parameters(parameters, -update)

            # new_parameters = [ update_parameters(parameters[0], -update), parameters[1] ]
            # new_parameters = parameters - update
            # new_parameters = new_parameters.at[0].set(0.)
            # print(f'new_parameters: {new_parameters}')

        comm.barrier()
        parameters = comm.bcast(new_parameters, root=0)
        comm.barrier()
        step_size *= 0.999

    weights = np.array(weights)
    energies = np.array(weights * energies)
    total_weights = 0.0 * weights
    total_energies = 0.0 * weights

    comm.barrier()
    comm.Reduce([weights, MPI.FLOAT], [total_weights, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([energies, MPI.FLOAT], [total_energies, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.barrier()

    mean_energy = 0.0
    if rank == 0:
        total_energies /= total_weights
        # np.savetxt('parameters.dat', parameters[0])
        np.savetxt("samples.dat", np.stack((total_weights, total_energies)).T)
        mean_energy, _ = stat_utils.blocking_analysis(
            total_weights, total_energies, neql=0, printQ=True, writeBlockedQ=False
        )

    mean_energy = comm.bcast(mean_energy, root=0)
    return mean_energy, parameters


def driver_lr(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    seed=0,
    print_stats=True,
    dev_thresh_fac=1.0e6,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    key = random.PRNGKey(seed + rank)
    walker_0 = walker.copy()

    key, subkey = random.split(key)
    random_numbers = random.uniform(subkey, shape=(sampler.n_samples,))
    (
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
    ) = sampler.sampling_lr(
        walker, ham, parameters, wave, lattice, random_numbers, dev_thresh_fac
    )

    # reject outliers
    # samples_clean, clean_ind = stat_utils.reject_outliers(
    #     np.stack((energies, qp_weights, weights)).T, 0, dev_thresh_fac / 100.0
    # )
    # energies_clean = samples_clean[:, 0]
    # qp_weights_clean = samples_clean[:, 1]
    # weights_clean = samples_clean[:, 2]
    # weight = np.sum(weights_clean)
    # energy = np.average(energies_clean, weights=weights_clean)
    # qp_weight = np.average(qp_weights_clean, weights=weights_clean)

    # average and print energies for the current step
    weight = np.array([weight], dtype="float32")
    energy = np.array([weight * energy], dtype="float32")
    qp_weight = np.array([weight * qp_weight], dtype="float32")
    # energy = np.array([ energy ])
    total_weight = 0.0 * weight
    total_energy = 0.0 * weight
    total_qp_weight = 0.0 * weight
    # gradient = np.array(weight * gradient, dtype="float32")
    # lene_gradient = np.array(weight * lene_gradient, dtype="float32")

    # throw away zero norm vectors in the metric
    pos_ind = metric.diagonal().real > 0
    comm.barrier()
    pos_ind = comm.bcast(pos_ind, root=0)
    comm.barrier()
    metric = metric[pos_ind][:, pos_ind]
    h = h[pos_ind][:, pos_ind]

    metric = np.array(weight * metric)  # , dtype="complex64")
    h = np.array(weight * h)  # , dtype="complex64")
    # gradient = np.array(gradient)
    # lene_gradient = np.array(lene_gradient)
    total_metric = 0.0 * metric
    total_h = 0.0 * h

    comm.barrier()
    comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce(
        [qp_weight, MPI.FLOAT], [total_qp_weight, MPI.FLOAT], op=MPI.SUM, root=0
    )
    # comm.Reduce([metric, MPI.FLOAT], [total_metric, MPI.FLOAT], op=MPI.SUM, root=0)
    # comm.Reduce(
    #     [h, MPI.FLOAT],
    #     [total_h, MPI.FLOAT],
    #     op=MPI.SUM,
    #     root=0,
    # )
    total_metric = comm.reduce(metric, op=MPI.SUM, root=0)
    total_h = comm.reduce(h, op=MPI.SUM, root=0)
    comm.barrier()
    k_vec = None
    if rank == 0:
        total_energy /= total_weight
        total_qp_weight /= total_weight
        assert total_metric is not None
        assert total_h is not None
        total_metric /= total_weight
        total_h /= total_weight
        print(f"energy: {total_energy / total_metric[0,0]}")

        total_h = (total_h + total_h.T.conj()) / 2
        # overlap = wave.calc_overlap(walker_0[0], walker_0[1], parameters, lattice)
        # grad = wave.calc_overlap_gradient(walker_0[0], walker_0[1], parameters, lattice)
        # ext_grad = jnp.zeros(wave.n_parameters + 1) + 0.0j
        # ext_grad = ext_grad.at[0].set(1)
        # ext_grad = ext_grad.at[1:].set(grad)
        # k_vec = ext_grad * jnp.exp(overlap)
        # k_vec = k_vec[pos_ind]

    comm.barrier()
    total_h = comm.bcast(total_h, root=0)
    total_metric = comm.bcast(total_metric, root=0)
    # k_vec = comm.bcast(k_vec, root=0)
    comm.barrier()

    return total_metric, total_h, pos_ind


def driver_lr_sf(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    seed=0,
    print_stats=True,
    dev_thresh_fac=1.0e6,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    key = random.PRNGKey(seed + rank)
    walker_0 = walker.copy()

    # dev_thresh_fac = 1.0e6
    key, subkey = random.split(key)
    random_numbers = random.uniform(subkey, shape=(sampler.n_samples,))
    (
        weight,
        energy,
        norm,
        prop,
        vector_prop,
        metric,
        h,
        energies,
        norms,
        weights,
    ) = sampler.sampling_lr_sf(
        walker, ham, parameters, wave, lattice, random_numbers, dev_thresh_fac
    )
    # return metric, h

    # reject outliers
    # samples_clean, clean_ind = stat_utils.reject_outliers(
    #     np.stack((energies, qp_weights, weights)).T, 0, dev_thresh_fac / 100.0
    # )
    # energies_clean = samples_clean[:, 0]
    # qp_weights_clean = samples_clean[:, 1]
    # weights_clean = samples_clean[:, 2]
    # weight = np.sum(weights_clean)
    # energy = np.average(energies_clean, weights=weights_clean)
    # qp_weight = np.average(qp_weights_clean, weights=weights_clean)

    # average and print energies for the current step
    weight = np.array([weight], dtype="float32")
    energy = np.array([weight * energy], dtype="float32")
    norm = np.array([weight * norm], dtype="float32")
    # energy = np.array([ energy ])
    total_weight = 0.0 * weight
    total_energy = 0.0 * weight
    total_norm = 0.0 * weight
    # gradient = np.array(weight * gradient, dtype="float32")
    # lene_gradient = np.array(weight * lene_gradient, dtype="float32")

    # throw away zero norm vectors in the metric
    pos_ind = metric.diagonal().real > 0
    comm.barrier()
    pos_ind = comm.bcast(pos_ind, root=0)
    comm.barrier()
    metric = metric[pos_ind][:, pos_ind]
    h = h[pos_ind][:, pos_ind]
    prop = prop[pos_ind]

    prop = np.array(weight * prop)  # , dtype="float32")
    vector_prop = np.array(weight * vector_prop)  # , dtype="float32")
    metric = np.array(weight * metric)  # , dtype="complex64")
    h = np.array(weight * h)  # , dtype="complex64")
    # gradient = np.array(gradient)
    # lene_gradient = np.array(lene_gradient)
    total_metric = 0.0 * metric
    total_h = 0.0 * h

    comm.barrier()
    comm.Reduce([weight, MPI.FLOAT], [total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([energy, MPI.FLOAT], [total_energy, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([norm, MPI.FLOAT], [total_norm, MPI.FLOAT], op=MPI.SUM, root=0)
    total_prop = comm.reduce(prop, op=MPI.SUM, root=0)
    total_vector_prop = comm.reduce(vector_prop, op=MPI.SUM, root=0)
    total_metric = comm.reduce(metric, op=MPI.SUM, root=0)
    total_h = comm.reduce(h, op=MPI.SUM, root=0)
    comm.barrier()
    if rank == 0:
        total_energy /= total_weight
        total_norm /= total_weight
        assert total_prop is not None
        assert total_vector_prop is not None
        assert total_metric is not None
        assert total_h is not None
        total_prop /= total_weight
        total_vector_prop /= total_weight
        total_metric /= total_weight
        total_h /= total_weight
        print(f"energy: {total_energy / total_metric[0,0]}")
        # print(f"norm: {total_norm}")
        total_h = (total_h + total_h.T.conj()) / 2
        total_prop[:, 0] = total_prop[:, 0]  # / (total_norm) ** 0.5
        total_prop[:, 1] = total_prop[:, 1]  # / (total_norm) ** 0.5
        total_vector_prop = total_vector_prop / total_norm
        # total_prop = total_prop[:, :2]
        # total_prop_norm = total_prop_norm[:2] / total_norm

    comm.barrier()
    total_h = comm.bcast(total_h, root=0)
    total_metric = comm.bcast(total_metric, root=0)
    total_prop = comm.bcast(total_prop, root=0)
    total_vector_prop = comm.bcast(total_vector_prop, root=0)
    comm.barrier()

    return total_metric, total_h, total_prop, total_vector_prop
