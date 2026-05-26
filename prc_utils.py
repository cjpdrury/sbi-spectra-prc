## Util functions 
import time
from datetime import datetime

import jax
import ast
import sys
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel
from sklearn.linear_model import LinearRegression
from jaxspec.model.additive import Powerlaw, Blackbodyrad, Blackbody
from jaxspec.model.multiplicative import Tbabs, Phabs
from tabulate import tabulate


# Utility to print messages to the terminal in some format
def print_message(message):
    lines = message.split('\n')
    formatted_message="\n================================================================================\n"
    formatted_message+= '\n'.join("[SIXSA] " + line for line in lines)
    formatted_message+="\n================================================================================\n"
    print(formatted_message)


def summary_statistics_func(
    data: np.ndarray,
    energy_grid=None,
    energy_ref=None,
    with_basic_stats=True,
    with_sum=True,
    with_ratio=True,
    with_diff=True,
    with_energy_weighted=False,
):

    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, M)
    num_spectrum, num_bins = data.shape

    data_transformed_list = []
    labels = []

    if with_basic_stats:

        mean_x = np.mean(data, axis=1)
        std_x = np.std(data, axis=1, ddof=1)
        sum_x = np.sum(data, axis=1)

        data_transformed_list.append(mean_x)
        labels.append("Mean")
        data_transformed_list.append(std_x)
        labels.append("Std")
        data_transformed_list.append(sum_x)
        labels.append("Sum")

    if len(energy_grid) == 2:

        energies = energy_ref
        energy_bins_summary = np.append(energies[0], energies[1, -1])
        idx_low = np.searchsorted(energy_bins_summary, energy_grid.min())
        idx_high = np.searchsorted(energy_bins_summary, energy_grid.max())
        energy_bins_summary = energy_bins_summary[idx_low:idx_high + 1]

    else:
        energy_bins_summary = energy_grid

    counts = np.zeros((num_spectrum, len(energy_bins_summary),))
    energy_low_observation, energy_high_observation = energy_ref[:-1], energy_ref[1:]

    for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
        counts_in_bin = np.sum(data[:, (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)], axis=1)
        counts[:, i] += counts_in_bin

        if with_sum:

            data_transformed_list.append(counts_in_bin)
            labels.append(f"Sums in band {e_low_summary:.4f}-{e_high_summary:.4f}")

    epsilon = 1
    # Hardness ratios
    if with_ratio:
        hardness_ratios = counts[:, 1:] / (counts[:, :-1] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(hardness_ratios[:, i])
            labels.append(f"Hardness ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    # Differential ratios
    if with_diff:
        differential_ratios = (counts[:, :-1] - counts[:, 1:]) / (counts[:, :-1] + counts[:, 1:] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(differential_ratios[:, i])
            labels.append(f"Differential ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    if with_energy_weighted:
        for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
            idx = (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)
            average_counts = data[:, idx]

            if average_counts.sum() < len(average_counts):
                average_counts = np.ones_like(average_counts)

            average_energy = (energy_low_observation[idx] + energy_high_observation[idx])/2
            result = np.apply_along_axis(lambda x : np.average(average_energy, weights=x/x.sum()), 1, average_counts)
            data_transformed_list.append(result)
            labels.append(f"Weighted energy in {e_low_summary:.4f}-{e_high_summary:.4f}")

    data_transformed = np.column_stack(data_transformed_list)

    return data_transformed, labels




# ===============================================================================================================

def compute_x_sim( jaxspec_model_expression , parameter_states , thetas , pha_file , energy_min , energy_max ,
                free_parameter_prior_types , parameter_lower_bounds , apply_stat = True , verbose = False ) :
    """
    # compute_x_sim: compute the simulated spectra with jaxspec fakeit like command.
    # It is therefore dependent on jaxspec, which currently has a limited number of models implemented.
    # It is possible to generate simulated spectra with other software, such as XSPEC, as long as the output format
    # remains similar. The output format is an array of spectra in counts. jaxspec is really powerful in terms of speed.
    # More models will be implemented as time goes (see the jaxspec documentation for the synthax of the models).

    Args:
        jaxspec_model_expression (_type_): _description_
        parameter_states (_type_): _description_
        thetas (_type_): _description_
        pha_file (_type_): _description_
        energy_min (_type_): _description_
        energy_max (_type_): _description_
        free_parameter_prior_types (_type_): _description_
        parameter_lower_bounds (_type_): _description_
        apply_stat (bool, optional): _description_. Defaults to True.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    #
    # Apply the transformation if needed
    #
    thetas = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** thetas , thetas))

    # jaxspec_model = SpectralModel.from_string(jaxspec_model_expression)
    jaxspec_model = eval(jaxspec_model_expression)

    parameter_values = []
    index_theta = 0

    for i_param , param_state in enumerate(parameter_states) :
        if param_state == "free" :
            parameter_values.append([thetas[j][index_theta] for j in range(len(thetas))])
            index_theta += 1
            if verbose :
                print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

        elif param_state == "frozen" :
            parameter_values.append([parameter_lower_bounds[i_param] for j in range(len(thetas))])
            if verbose :
                print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

    params_to_set = jaxspec_model.params
    i_para = 0

    for l , param_set in params_to_set.items( ) :
        for param_name , _ in param_set.items( ) :
            upd_dict = {param_name : np.array(parameter_values[i_para])}
            param_set.update(upd_dict)
            i_para += 1

    folding_model = ObsConfiguration.from_pha_file(pha_file , energy_min , energy_max)

    if len(thetas) > 1 :
        print("Multiple thetas simulated -> parallelization with JAX required")
        start_time = time.perf_counter( )
        x = jax.jit(lambda s : fakeit_for_multiple_parameters(folding_model , jaxspec_model , s ,
                                                            apply_stat = apply_stat))(params_to_set)

        end_time = time.perf_counter( )
        duration_time = end_time - start_time
        print(f"It took just {duration_time:.1f} seconds for jax.jit to generate {len(thetas)} simulations")
    #    return torch.as_tensor(np.array(x).astype(np.float32))
    else :
        print("One single theta simulated -> parallelization with JAX not required")
        x = fakeit_for_multiple_parameters(folding_model , jaxspec_model , params_to_set , apply_stat = apply_stat)
    return torch.as_tensor(np.array(x).astype(np.float32))
